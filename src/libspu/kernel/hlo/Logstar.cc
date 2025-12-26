#include "libspu/kernel/hlo/Logstar.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "libspu/core/bit_utils.h"
#include "libspu/core/context.h"
#include "libspu/core/trace.h"
#include "libspu/core/vectorize.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/random.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hal/utils.h"
#include "libspu/kernel/hlo/permute.h"
#include "libspu/kernel/hlo/shuffle.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/spu.h"

namespace spu::kernel::hlo {

// ====================================================================================
//                Vectorized AggregateBrentKung with valid bits
// ====================================================================================

// 【优化2】更高效的索引扩展
// 避免反复 push_back，使用 resize + 直接内存访问
static spu::Index ExpandIndices(const spu::Index& rows, int64_t width) {
  if (width == 1) {
    return rows;
  }
  spu::Index expanded;
  size_t total_size = rows.size() * width;
  expanded.resize(total_size);

  int64_t* out_ptr = expanded.data();
  const int64_t* in_ptr = rows.data();
  size_t n_rows = rows.size();

  // 外层循环并行化收益不高，因为只是简单的整数加法，单线程顺序写反而缓存更友好
  for (size_t i = 0; i < n_rows; ++i) {
    int64_t base = in_ptr[i] * width;
    for (int64_t k = 0; k < width; ++k) {
      out_ptr[i * width + k] = base + k;
    }
  }
  return expanded;
}

spu::Value _2s(SPUContext* ctx, const Value& x) {
  if (x.isPublic()) {
    return kernel::hal::_p2s(ctx, x);
  } else if (x.isPrivate()) {
    return kernel::hal::_v2s(ctx, x);
  }
  return x;
}

static Value PrepareMutableCopy(SPUContext* ctx, const Value& input) {
  Value copy;
  if (!input.isSecret()) {
    copy = _2s(ctx, input.clone()).setDtype(input.dtype());
  } else {
    copy = input.clone();
  }
  return kernel::hal::_prefer_a(ctx, copy);
}

// 【优化3】移除冗余检查，假定调用方保证 P:[K, B], G:[K, 1]
static std::pair<Value, Value> VectorizedNoteFunc(SPUContext* ctx,
                                                  const Value& p1,
                                                  const Value& p2,
                                                  const Value& g1,
                                                  const Value& g2) {
  // 1. g3 = g1 * g2
  auto g3 = kernel::hal::mul(ctx, g1, g2);

  // 2. diff = p2 - p1
  auto diff = kernel::hal::sub(ctx, p2, p1);

  // 3. 强制广播 g1: [K, 1] -> [K, BlockSize]
  // 注意：p1/p2 形状是 [K, BlockSize]，如果 BlockSize=1，broadcast_to
  // 开销极低（仅改元数据）
  Value g1_broadcasted = kernel::hal::broadcast_to(ctx, g1, diff.shape());

  // 4. term = diff * g1
  auto term = kernel::hal::mul(ctx, diff, g1_broadcasted);

  // 5. p3 = p1 + term
  auto p3 = kernel::hal::add(ctx, p1, term);
  return {p3, g3};
}

// Down-Sweep 版本
static Value VectorizedNoteFunc(SPUContext* ctx, const Value& p1,
                                const Value& p2, const Value& g1) {
  auto diff = kernel::hal::sub(ctx, p2, p1);
  Value g1_broadcasted = kernel::hal::broadcast_to(ctx, g1, diff.shape());
  auto term = kernel::hal::mul(ctx, diff, g1_broadcasted);
  auto p3 = kernel::hal::add(ctx, p1, term);
  return p3;
}

std::pair<Value, Value> AggregateBrentKung(SPUContext* ctx, const Value& x_full,
                                           const Value& valid_full,
                                           const Value& g_full_in) {
  const int64_t n = x_full.shape()[0];
  const int64_t block_size = x_full.shape()[1];
  const int64_t total_block_size = block_size * 2;

  // 1. 数据准备
  Value p_full = kernel::hal::concatenate(ctx, {x_full, valid_full}, 1);
  p_full = kernel::hal::_prefer_a(ctx, p_full);
  Value g_full = PrepareMutableCopy(ctx, g_full_in);

  auto p_flat_ref = p_full.data().reshape({p_full.numel()});
  auto g_flat_ref = g_full.data().reshape({g_full.numel()});

  int depth = 0;
  if (n > 1) {
    depth = std::ceil(std::log2(n));
  }

  // 【优化1】预分配内存，循环外声明
  // 预估最大所需的 vector 大小约为 N/2
  size_t max_rows = (n / 2) + 1;
  spu::Index idx_right_rows;
  idx_right_rows.reserve(max_rows);
  spu::Index idx_left_rows;
  idx_left_rows.reserve(max_rows);

  // 这两个用于 Down-Sweep
  spu::Index idx_root_rows;
  idx_root_rows.reserve(max_rows);
  spu::Index idx_child_rows;
  idx_child_rows.reserve(max_rows);

  // ----------------------------------------------------
  // 2. Up-Sweep (Reduce)
  // ----------------------------------------------------
  for (int j = 0; j < depth; ++j) {
    int64_t step = 1LL << (j + 1);
    int64_t left_child_off = 1LL << j;

    // 清空而不释放内存
    idx_right_rows.clear();
    idx_left_rows.clear();

    for (int64_t i = step - 1; i < n; i += step) {
      idx_right_rows.push_back(i);
      idx_left_rows.push_back(i - left_child_off);
    }

    if (idx_right_rows.empty()) continue;

    // ExpandIndices 内部会分配新内存，但由于是临时变量且有 RVO，开销可控。
    // 如果极度追求性能，可以将 expanded
    // 向量也在循环外复用，但会增加代码复杂度(需改Expand接口)
    spu::Index idx_right_elems =
        ExpandIndices(idx_right_rows, total_block_size);
    spu::Index idx_left_elems = ExpandIndices(idx_left_rows, total_block_size);
    const spu::Index& idx_right_g = idx_right_rows;
    const spu::Index& idx_left_g = idx_left_rows;

    // Gather
    Value v_right_p(p_flat_ref.linear_gather(idx_right_elems), p_full.dtype());
    Value v_left_p(p_flat_ref.linear_gather(idx_left_elems), p_full.dtype());
    Value v_right_g(g_flat_ref.linear_gather(idx_right_g), g_full.dtype());
    Value v_left_g(g_flat_ref.linear_gather(idx_left_g), g_full.dtype());

    // Reshape
    int64_t k = idx_right_rows.size();
    v_right_p = kernel::hal::reshape(ctx, v_right_p, {k, total_block_size});
    v_left_p = kernel::hal::reshape(ctx, v_left_p, {k, total_block_size});
    v_right_g = kernel::hal::reshape(ctx, v_right_g, {k, 1});
    v_left_g = kernel::hal::reshape(ctx, v_left_g, {k, 1});

    // Compute
    auto [new_p, new_g] =
        VectorizedNoteFunc(ctx, v_right_p, v_left_p, v_right_g, v_left_g);

    // Scatter
    auto new_p_data = new_p.data().reshape({new_p.numel()});
    auto new_g_data = new_g.data().reshape({new_g.numel()});

    p_flat_ref.linear_scatter(new_p_data, idx_right_elems);
    g_flat_ref.linear_scatter(new_g_data, idx_right_g);
  }

  // ----------------------------------------------------
  // 3. Down-Sweep (Distribute)
  // ----------------------------------------------------
  for (int j = depth - 2; j >= 0; --j) {
    int64_t step = 1LL << (j + 1);
    int64_t dist = 1LL << j;

    idx_root_rows.clear();
    idx_child_rows.clear();

    for (int64_t i = step - 1; i < n; i += step) {
      int64_t target = i + dist;
      if (target < n) {
        idx_root_rows.push_back(i);
        idx_child_rows.push_back(target);
      }
    }

    if (idx_child_rows.empty()) continue;

    spu::Index idx_root_elems = ExpandIndices(idx_root_rows, total_block_size);
    spu::Index idx_child_elems =
        ExpandIndices(idx_child_rows, total_block_size);
    const spu::Index& idx_child_g = idx_child_rows;

    // Gather
    Value v_root_p(p_flat_ref.linear_gather(idx_root_elems), p_full.dtype());
    Value v_child_p(p_flat_ref.linear_gather(idx_child_elems), p_full.dtype());
    Value v_child_g(g_flat_ref.linear_gather(idx_child_g), g_full.dtype());

    // Reshape
    int64_t k = idx_child_rows.size();
    v_root_p = kernel::hal::reshape(ctx, v_root_p, {k, total_block_size});
    v_child_p = kernel::hal::reshape(ctx, v_child_p, {k, total_block_size});
    v_child_g = kernel::hal::reshape(ctx, v_child_g, {k, 1});

    // Compute
    auto new_p_child = VectorizedNoteFunc(ctx, v_child_p, v_root_p, v_child_g);

    // Scatter
    auto new_p_data = new_p_child.data().reshape({new_p_child.numel()});
    p_flat_ref.linear_scatter(new_p_data, idx_child_elems);
  }

  auto y_out = kernel::hal::slice(ctx, p_full, {0, 0}, {n, block_size}, {});
  auto valid_out =
      kernel::hal::slice(ctx, p_full, {0, block_size}, {n, 2 * block_size}, {});

  return {y_out, valid_out};
}
// ====================================================================================
//                    ExtraxtOrdered
// ====================================================================================
Value prefix_sum(SPUContext* ctx, const Value& x) {
  if (x.shape().ndim() == 1) {
    // reshape 1D -> [1, n]
    const int64_t n = x.numel();
    auto xr = hal::reshape(ctx, x, {1, n});
    return prefix_sum(ctx, xr);
  }

  SPU_ENFORCE(x.shape().ndim() == 2U && x.shape()[0] == 1,
              "x should be 1-row matrix");

  const int64_t n = x.numel();
  if (n == 0) {
    return x;
  }

  // 使用扫描算法手动实现前缀和
  std::vector<Value> parts;
  parts.reserve(n);

  // 获取第一个元素
  auto first = hal::slice(ctx, x, {0, 0}, {1, 1});
  parts.push_back(first);

  // 逐步累加
  for (int64_t i = 1; i < n; ++i) {
    auto current = hal::slice(ctx, x, {0, i}, {1, i + 1});
    auto prev_sum = parts.back();
    auto sum = hal::add(ctx, prev_sum, current);
    parts.push_back(sum);
  }

  // 水平拼接结果
  return hal::concatenate(ctx, parts, 1);
}

void extract_ordered_(SPUContext* ctx, const spu::Value& x,
                      const spu::Value& valids) {
  // 1.计算valids前缀和rho
  auto rho = prefix_sum(ctx, valids);
  // 2.洗牌x、valids、rho
  std::vector<spu::Value> inputs_to_shuffle = {x, valids, rho};
  // std::vector<spu::Value> shuffled_results = hlo::Shuffle(ctx,
  // inputs_to_shuffle, 1);
  auto [shuffled_results, pi] =
      hlo::shuffle_with_perm(ctx, inputs_to_shuffle, 1);
  auto sx = shuffled_results[0];
  auto svalids = shuffled_results[1];
  auto srho = shuffled_results[2];
  // ！！！问题：shuffle返回的置换pi是PShr的
  if (ctx->lctx()->Rank() == 0) {
    std::cout << ">> [Debug pi]" << std::endl;
    std::cout << "   Storage Type: " << pi.storage_type() << std::endl;
    std::cout << "   Data Type:    " << pi.dtype() << std::endl;
    std::cout << "   Visibility:   " << pi.vtype() << std::endl;
  }

  if (ctx->lctx()->Rank() == 0) {
    std::cout << ">> [Debug sx]" << std::endl;
    std::cout << "   Storage Type: " << sx.storage_type() << std::endl;
    std::cout << "   Data Type:    " << sx.dtype() << std::endl;
    std::cout << "   Visibility:   " << sx.vtype() << std::endl;
  }

  // 3.打开svalids
  auto svalids_open =
      hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx, svalids));
  auto sx_open = hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx, sx));
  auto srho_open = hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx, srho));
  if (ctx->lctx()->Rank() == 0) {
    std::cout << "svalids_open: " << svalids_open << std::endl;
    std::cout << "sx: " << sx_open << std::endl;
    std::cout << "srho: " << srho_open << std::endl;
  }

  // 4.计算公开置换p_hat
  int64_t numel = svalids_open.size();
  std::vector<int64_t> p_hat_indices(numel);
  // 双指针单次遍历填充
  int64_t left = 0;
  int64_t right = numel - 1;
  for (int64_t i = 0; i < numel; ++i) {
    if (svalids_open[i]) {
      p_hat_indices[left++] = i;  // 有效的放前面
    } else {
      p_hat_indices[right--] = i;  // 无效的放后面 (倒序填充)
    }
  }
  int64_t valid_count = left;

  if (ctx->lctx()->Rank() == 0) {
    std::cout << "p_hat_indices: " << xt::adapt(p_hat_indices) << std::endl;
  }
  if (ctx->lctx()->Rank() == 0) {
    std::cout << "valid_count: " << valid_count << std::endl;
  }

  // 5. 用公开的p_hat置换（sx, srho, pi）.
  // ！！！pi是PShr的置换它会报错，
  // 解决方案：（1）PShr的pi转换成AShr的（？）
  //          （2）用1比特radix_sort替换extract_ordered（开销貌似大一些）
  //          （3）不计算置换，把x改成多维的
  auto p_hat_xt = xt::adapt(p_hat_indices);
  spu::Value compact = hal::constant(
      ctx, p_hat_xt, spu::DT_I64, {static_cast<int64_t>(p_hat_indices.size())});
  std::vector<spu::Value> inputs_to_permute = {sx, srho, pi};
  std::vector<spu::Value> compacted_results =
      hlo::Permute(ctx, inputs_to_permute, compact, 1);

  auto y_compacted = compacted_results[0];
  auto rho_prime = compacted_results[1];
  auto rho_compose_pi = compacted_results[2];

  auto y_compacted_open =
      hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx, y_compacted));
  auto rho_prime_open =
      hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx, rho_prime));
  auto rho_compose_pi_open =
      hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx, rho_compose_pi));
  if (ctx->lctx()->Rank() == 0) {
    std::cout << "y_compacted: " << y_compacted_open << std::endl;
    std::cout << "rho_prime: " << rho_prime_open << std::endl;
    std::cout << "rho_compose_pi: " << rho_compose_pi_open << std::endl;
  }

  // // -----------------------------------------------------------------------
  // // Step 6: 截取前 c 个元素 (Slice)
  // // -----------------------------------------------------------------------

  // // 构造切片范围: [0, valid_count)
  // // 假设数据形状是 (1, N)，我们在第 1 维切片
  // std::vector<int64_t> start_indices(sx.shape().ndim(), 0);
  // std::vector<int64_t> end_indices = sx.shape();

  // // 设置切片的结束位置
  // if (sx.shape().ndim() == 1) {
  //     end_indices[0] = valid_count;
  // } else {
  //     // 假设第 0 维是 Batch(1)，第 1 维是数据
  //     end_indices[1] = valid_count;
  // }

  // // 执行切片，得到最终紧凑的 Secret Shared 结果
  // spu::Value y_final = hal::slice(ctx, y_compacted, start_indices,
  // end_indices, {});

  // // 返回结果
  // // 如果是 Extract Unordered，到这里就结束了，返回 y_final
  // // 如果是 Extract Ordered，你需要利用 rho_prime 继续进行后续的 Unshuffle
  // 操作 return std::vector<spu::Value>{y_final, rho_prime};
}

// ====================================================================================
//                备选方案
// ====================================================================================

// ====================================================================================
// Vectorized AggregateBrentKung without valid bits 备选
// // padding 方案，一维数组 In-place 迭代
// std::pair<Value, Value> AggregateBrentKung(SPUContext* ctx, const Value&
// x_full,
//                                            const Value& valid_full,
//                                            const Value& g_full) {
//   const int64_t n = x_full.shape()[0];
//   const int64_t block_size = x_full.shape()[1];

//   // 1. Padding 处理 (保持不变)
//   int64_t n_padded = NextPowerOfTwo(n);

//   Value x_use = x_full;
//   Value valid_use = valid_full;
//   Value g_use = g_full;

//   if (n_padded > n) {
//     int64_t pad_rows = n_padded - n;

//     auto padding_zeros_x = hal::constant(ctx, static_cast<int64_t>(0),
//     DT_I64,
//                                          {pad_rows, block_size});
//     std::vector<Value> vec_x;
//     vec_x.reserve(2);
//     vec_x.push_back(x_full);
//     vec_x.push_back(padding_zeros_x);
//     x_use = hal::concatenate(ctx, vec_x, 0);

//     auto padding_zeros_v = hal::constant(ctx, static_cast<int64_t>(0),
//     DT_I64,
//                                          {pad_rows, block_size});
//     std::vector<Value> vec_v;
//     vec_v.reserve(2);
//     vec_v.push_back(valid_full);
//     vec_v.push_back(padding_zeros_v);
//     valid_use = hal::concatenate(ctx, vec_v, 0);

//     auto padding_zeros_g =
//         hal::constant(ctx, static_cast<int64_t>(0), DT_I64, {pad_rows, 1});
//     std::vector<Value> vec_g;
//     vec_g.reserve(2);
//     vec_g.push_back(g_full);
//     vec_g.push_back(padding_zeros_g);
//     g_use = hal::concatenate(ctx, vec_g, 0);
//   }

//   const int64_t current_n = n_padded;
//   const int64_t total_block_size = block_size * 2;
//   const int64_t logn = std::floor(std::log2(current_n));

//   // 拼接 payload
//   std::vector<Value> vec_payload;
//   vec_payload.reserve(2);
//   vec_payload.push_back(x_use);
//   vec_payload.push_back(valid_use);
//   Value payload_full = hal::concatenate(ctx, vec_payload, 1);

//   // 维护一维数组
//   std::vector<Value> p_rows(current_n);
//   std::vector<Value> g_rows(current_n);

//   for (int i = 0; i < current_n; ++i) {
//     p_rows[i] =
//         hal::slice(ctx, payload_full, {i, 0}, {i + 1, total_block_size}, {});
//     g_rows[i] = hal::slice(ctx, g_use, {i, 0}, {i + 1, 1}, {});
//     p_rows[i] = hal::reshape(ctx, p_rows[i], {1, total_block_size});
//     g_rows[i] = hal::reshape(ctx, g_rows[i], {1, 1});
//   }

//   // ========================================================
//   // 1. Up-Sweep
//   // ========================================================
//   for (int j = 0; j < logn; ++j) {
//     int64_t step = 1LL << (j + 1);
//     int64_t left_off = 1LL << j;

//     std::vector<Value> right_p, left_p, right_g, left_g;
//     std::vector<int> target_indices;

//     size_t est_size = (current_n / step) + 1;
//     right_p.reserve(est_size);
//     left_p.reserve(est_size);
//     right_g.reserve(est_size);
//     left_g.reserve(est_size);
//     target_indices.reserve(est_size);

//     for (int64_t i = step - 1; i < current_n; i += step) {
//       int64_t left_idx = i - left_off;

//       // i 是当前节点(Right), left_idx 是左侧节点(Left)
//       right_p.push_back(p_rows[i]);
//       left_p.push_back(p_rows[left_idx]);
//       right_g.push_back(g_rows[i]);
//       left_g.push_back(g_rows[left_idx]);
//       target_indices.push_back(i);
//     }

//     if (!target_indices.empty()) {
//       auto v_right_p = hal::concatenate(ctx, right_p, 0);
//       auto v_left_p = hal::concatenate(ctx, left_p, 0);
//       auto v_right_g = hal::concatenate(ctx, right_g, 0);
//       auto v_left_g = hal::concatenate(ctx, left_g, 0);

//       // [FIX 1] 参数顺序修正：(Right, Left, Right_G, Left_G)
//       auto [new_p, new_g] =
//           VectorizedNoteFunc(ctx, v_right_p, v_left_p, v_right_g, v_left_g);

//       for (size_t k = 0; k < target_indices.size(); ++k) {
//         int idx = target_indices[k];
//         p_rows[idx] =
//             hal::slice(ctx, new_p, {static_cast<int64_t>(k), 0},
//                        {static_cast<int64_t>(k + 1), total_block_size}, {});
//         g_rows[idx] = hal::slice(ctx, new_g, {static_cast<int64_t>(k), 0},
//                                  {static_cast<int64_t>(k + 1), 1}, {});
//       }
//     }
//   }

//   // ========================================================
//   // 2. Down-Sweep
//   // ========================================================
//   for (int j = logn - 2; j >= 0; --j) {
//     int64_t step = 1LL << (j + 1);
//     int64_t right_off = 1LL << j;

//     std::vector<Value> root_p, child_p, child_g;  // 不需要 root_g
//     std::vector<int> target_indices;

//     size_t est_size = (current_n / step) + 1;
//     root_p.reserve(est_size);
//     child_p.reserve(est_size);
//     child_g.reserve(est_size);
//     target_indices.reserve(est_size);

//     for (int64_t i = step - 1; i < current_n; i += step) {
//       int64_t target = i + right_off;

//       if (target < current_n) {
//         // i 是 Root (Left), target 是 Child (Right)
//         root_p.push_back(p_rows[i]);
//         child_p.push_back(p_rows[target]);
//         child_g.push_back(g_rows[target]);
//         target_indices.push_back(target);
//       }
//     }

//     if (!target_indices.empty()) {
//       auto v_root_p = hal::concatenate(ctx, root_p, 0);
//       auto v_child_p = hal::concatenate(ctx, child_p, 0);
//       auto v_child_g = hal::concatenate(ctx, child_g, 0);

//       // child_new = child + root * child_g
//       auto new_p = VectorizedNoteFunc(ctx, v_child_p, v_root_p, v_child_g);

//       for (size_t k = 0; k < target_indices.size(); ++k) {
//         int idx = target_indices[k];
//         p_rows[idx] =
//             hal::slice(ctx, new_p, {static_cast<int64_t>(k), 0},
//                        {static_cast<int64_t>(k + 1), total_block_size}, {});
//       }
//     }
//   }

//   // --- 3. 输出后处理 ---
//   auto final_payload = hal::concatenate(ctx, p_rows, 0);

//   auto y_out_padded =
//       hal::slice(ctx, final_payload, {0, 0}, {current_n, block_size}, {});
//   auto valid_out_padded = hal::slice(ctx, final_payload, {0, block_size},
//                                      {current_n, total_block_size}, {});

//   // 4. 去除 Padding
//   if (n_padded > n) {
//     y_out_padded = hal::slice(ctx, y_out_padded, {0, 0}, {n, block_size},
//     {}); valid_out_padded =
//         hal::slice(ctx, valid_out_padded, {0, 0}, {n, block_size}, {});
//   }

//   return {y_out_padded, valid_out_padded};
// }

// // 最差：无法处理非2的幂的情况
// std::pair<Value, Value> AggregateBrentKung(SPUContext* ctx,
//   const Value& x_full,
//   const Value& valid_full,
//   const Value& g_full) {

//   // 1. 获取维度信息
//   const int64_t n = x_full.shape()[0];
//   const int64_t block_size = x_full.shape()[1]; // B

//   // 检查 valid 维度是否匹配 (Debug模式下很有用，Release可省略)
//   if (valid_full.shape()[0] != n || valid_full.shape()[1] != block_size) {
//   // 实际代码中建议加 SPUENFORCE 或类似检查
//   }

//   // 2. 数据拼接 (Pre-process)
//   // 将 x (N, B) 和 valid (N, B) 在列维度拼接 -> (N, 2*B)
//   Value payload_full = hal::concatenate(ctx, {x_full, valid_full}, 1);

//   // 更新后续逻辑使用的 block_size
//   const int64_t total_block_size = block_size * 2;
//   const int64_t logn = std::floor(std::log2(n));

//   // --- 下面的逻辑与原版完全一致，只是操作的是 payload_full ---

//   std::vector<Value> p_rows(n);
//   std::vector<Value> g_rows(n);

//   for (int i = 0; i < n; ++i) {
//   // Slice 出一行: [1, 2B]
//   p_rows[i] = hal::slice(ctx, payload_full, {i, 0}, {i + 1,
//   total_block_size}, {}); g_rows[i] = hal::slice(ctx, g_full, {i, 0}, {i + 1,
//   1}, {});

//   // 保持 Rank=2 以便拼接
//   p_rows[i] = hal::reshape(ctx, p_rows[i], {1, total_block_size});
//   g_rows[i] = hal::reshape(ctx, g_rows[i], {1, 1});
//   }

//   std::vector<std::vector<Value>> p_tree(n, std::vector<Value>(logn));
//   std::vector<std::vector<Value>> g_tree(n, std::vector<Value>(logn));
//   std::vector<Value> res(n);

//   // --------------------------------------------------------
//   // Up-Sweep
//   // --------------------------------------------------------
//   {
//   std::vector<Value> p1_batch, p2_batch, g1_batch, g2_batch;
//   std::vector<int> target_indices;

//   for (int i = 1; i < n; i += 2) {
//   p1_batch.push_back(p_rows[i]);
//   p2_batch.push_back(p_rows[i - 1]);
//   g1_batch.push_back(g_rows[i]);
//   g2_batch.push_back(g_rows[i - 1]);
//   target_indices.push_back(i);
//   }

//   if (!target_indices.empty()) {
//   auto p1_vec = hal::concatenate(ctx, p1_batch, 0);
//   auto p2_vec = hal::concatenate(ctx, p2_batch, 0);
//   auto g1_vec = hal::concatenate(ctx, g1_batch, 0);
//   auto g2_vec = hal::concatenate(ctx, g2_batch, 0);

//   // 这里的计算会自动带上 valid 部分
//   auto [p3_vec, g3_vec] = VectorizedNoteFunc(ctx, p1_vec, p2_vec, g1_vec,
//   g2_vec);

//   for (size_t k = 0; k < target_indices.size(); ++k) {
//   int idx = target_indices[k];
//   p_tree[idx][0] = hal::slice(ctx, p3_vec, {static_cast<int64_t>(k), 0},
//   {static_cast<int64_t>(k + 1), total_block_size}, {});
//   g_tree[idx][0] = hal::slice(ctx, g3_vec, {static_cast<int64_t>(k), 0},
//   {static_cast<int64_t>(k + 1), 1}, {});
//   }
//   }
//   }

//   for (int j = 1; j < logn; ++j) {
//   int step = 1 << (j + 1);
//   std::vector<Value> p1_batch, p2_batch, g1_batch, g2_batch;
//   std::vector<int> target_indices;

//   for (int i = step - 1; i < n; i += step) {
//   int prev_idx = i - (1 << j);
//   p1_batch.push_back(p_tree[i][j - 1]);
//   p2_batch.push_back(p_tree[prev_idx][j - 1]);
//   g1_batch.push_back(g_tree[i][j - 1]);
//   g2_batch.push_back(g_tree[prev_idx][j - 1]);
//   target_indices.push_back(i);
//   }

//   if (!target_indices.empty()) {
//   auto p1_vec = hal::concatenate(ctx, p1_batch, 0);
//   auto p2_vec = hal::concatenate(ctx, p2_batch, 0);
//   auto g1_vec = hal::concatenate(ctx, g1_batch, 0);
//   auto g2_vec = hal::concatenate(ctx, g2_batch, 0);

//   auto [p3_vec, g3_vec] = VectorizedNoteFunc(ctx, p1_vec, p2_vec, g1_vec,
//   g2_vec);

//   for (size_t k = 0; k < target_indices.size(); ++k) {
//   int idx = target_indices[k];
//   p_tree[idx][j] = hal::slice(ctx, p3_vec, {static_cast<int64_t>(k), 0},
//   {static_cast<int64_t>(k + 1), total_block_size}, {});
//   g_tree[idx][j] = hal::slice(ctx, g3_vec, {static_cast<int64_t>(k), 0},
//   {static_cast<int64_t>(k + 1), 1}, {});
//   }
//   }
//   }

//   // --------------------------------------------------------
//   // Down-Sweep
//   // --------------------------------------------------------
//   res[0] = p_rows[0];

//   for (int j = 0; j < logn; ++j) {
//   int idx = (1 << (j + 1)) - 1;
//   if (idx < n) {
//   res[idx] = p_tree[idx][j];
//   }
//   }

//   for (int j = logn - 3; j >= 0; --j) {
//   int step = 1 << (j + 2);
//   int half_step = 1 << (j + 1);

//   std::vector<Value> p1_batch, p2_batch, g1_batch;
//   std::vector<int> target_indices;

//   for (int k = 1; k < n / step + 1; ++k) {
//   int idx_curr = n - half_step * (2 * k - 1) - 1;
//   int idx_prev = n - half_step * 2 * k - 1;

//   if (idx_curr >= 0 && idx_prev >= 0 && idx_curr < n) {
//   p1_batch.push_back(p_tree[idx_curr][j]);
//   p2_batch.push_back(res[idx_prev]);
//   g1_batch.push_back(g_tree[idx_curr][j]);
//   target_indices.push_back(idx_curr);
//   }
//   }

//   if (!target_indices.empty()) {
//   auto p1_vec = hal::concatenate(ctx, p1_batch, 0);
//   auto p2_vec = hal::concatenate(ctx, p2_batch, 0);
//   auto g1_vec = hal::concatenate(ctx, g1_batch, 0);

//   auto p3_vec = VectorizedNoteFunc(ctx, p1_vec, p2_vec, g1_vec);

//   for (size_t k = 0; k < target_indices.size(); ++k) {
//   res[target_indices[k]] = hal::slice(ctx, p3_vec, {static_cast<int64_t>(k),
//   0},
//     {static_cast<int64_t>(k + 1), total_block_size}, {});
//   }
//   }
//   }

//   {
//   std::vector<Value> p1_batch, p2_batch, g1_batch;
//   std::vector<int> target_indices;

//   for (int k = 1; k < n / 2 + 1; ++k) {
//   int idx_curr = n - 2 * k;
//   int idx_prev = n - 2 * k - 1;

//   if (idx_curr >= 0 && idx_prev >= 0 && idx_curr < n) {
//   p1_batch.push_back(p_rows[idx_curr]);
//   p2_batch.push_back(res[idx_prev]);
//   g1_batch.push_back(g_rows[idx_curr]);
//   target_indices.push_back(idx_curr);
//   }
//   }

//   if (!target_indices.empty()) {
//   auto p1_vec = hal::concatenate(ctx, p1_batch, 0);
//   auto p2_vec = hal::concatenate(ctx, p2_batch, 0);
//   auto g1_vec = hal::concatenate(ctx, g1_batch, 0);

//   auto p3_vec = VectorizedNoteFunc(ctx, p1_vec, p2_vec, g1_vec);

//   for (size_t k = 0; k < target_indices.size(); ++k) {
//   res[target_indices[k]] = hal::slice(ctx, p3_vec, {static_cast<int64_t>(k),
//   0},
//     {static_cast<int64_t>(k + 1), total_block_size}, {});
//   }
//   }
//   }

//   // --- 3. 输出后处理 ---
//   // final_payload Shape: [N, 2*B]
//   auto final_payload = hal::concatenate(ctx, res, 0);

//   // 切分回 x 和 valid
//   // x:     [0 .. B)
//   // valid: [B .. 2B)
//   auto y_out = hal::slice(ctx, final_payload, {0, 0}, {n, block_size}, {});
//   auto valid_out = hal::slice(ctx, final_payload, {0, block_size}, {n, 2 *
//   block_size}, {});

//   return {y_out, valid_out};
// }

// // 第二差：padding 方案，2D数组存储树（空间复杂度太高，数据量大可能崩溃）
// std::pair<Value, Value> AggregateBrentKung(SPUContext* ctx,
//   const Value& x_full,
//   const Value& valid_full,
//   const Value& g_full) {

//   const int64_t n = x_full.shape()[0];
//   const int64_t block_size = x_full.shape()[1];

//   int64_t n_padded = NextPowerOfTwo(n);

//   Value x_use = x_full;
//   Value valid_use = valid_full;
//   Value g_use = g_full;

//   // 1. 如果需要 Padding
//   if (n_padded > n) {
//       int64_t pad_rows = n_padded - n;

//       // [Fix 1] 使用 DT_I64，并确保 0 是 int64_t 类型
//       auto padding_zeros_x = hal::constant(ctx, static_cast<int64_t>(0),
//       DT_I64, {pad_rows, block_size});

//       // [Fix 2] 使用 push_back 代替
//       {initializer_list}，避免编译器的隐式推导错误 std::vector<Value> vec_x;
//       vec_x.reserve(2);
//       vec_x.push_back(x_full);
//       vec_x.push_back(padding_zeros_x);
//       x_use = hal::concatenate(ctx, vec_x, 0);

//       auto padding_zeros_v = hal::constant(ctx, static_cast<int64_t>(0),
//       DT_I64, {pad_rows, block_size}); std::vector<Value> vec_v;
//       vec_v.reserve(2);
//       vec_v.push_back(valid_full);
//       vec_v.push_back(padding_zeros_v);
//       valid_use = hal::concatenate(ctx, vec_v, 0);

//       auto padding_zeros_g = hal::constant(ctx, static_cast<int64_t>(0),
//       DT_I64, {pad_rows, 1}); std::vector<Value> vec_g; vec_g.reserve(2);
//       vec_g.push_back(g_full);
//       vec_g.push_back(padding_zeros_g);
//       g_use = hal::concatenate(ctx, vec_g, 0);
//   }

//   // --- 主逻辑使用 n_padded (current_n) ---
//   const int64_t current_n = n_padded;
//   const int64_t total_block_size = block_size * 2;
//   const int64_t logn = std::floor(std::log2(current_n));

//   // 拼接 x 和 valid
//   std::vector<Value> vec_payload;
//   vec_payload.reserve(2);
//   vec_payload.push_back(x_use);
//   vec_payload.push_back(valid_use);
//   Value payload_full = hal::concatenate(ctx, vec_payload, 1);

//   std::vector<Value> p_rows(current_n);
//   std::vector<Value> g_rows(current_n);

//   for (int i = 0; i < current_n; ++i) {
//       p_rows[i] = hal::slice(ctx, payload_full, {i, 0}, {i + 1,
//       total_block_size}, {}); g_rows[i] = hal::slice(ctx, g_use, {i, 0}, {i +
//       1, 1}, {});

//       p_rows[i] = hal::reshape(ctx, p_rows[i], {1, total_block_size});
//       g_rows[i] = hal::reshape(ctx, g_rows[i], {1, 1});
//   }

//   std::vector<std::vector<Value>> p_tree(current_n,
//   std::vector<Value>(logn)); std::vector<std::vector<Value>>
//   g_tree(current_n, std::vector<Value>(logn)); std::vector<Value>
//   res(current_n);

//   // --------------------------------------------------------
//   // Up-Sweep
//   // --------------------------------------------------------
//   {
//       std::vector<Value> p1_batch, p2_batch, g1_batch, g2_batch;
//       std::vector<int> target_indices;

//       for (int i = 1; i < current_n; i += 2) {
//           p1_batch.push_back(p_rows[i]);
//           p2_batch.push_back(p_rows[i - 1]);
//           g1_batch.push_back(g_rows[i]);
//           g2_batch.push_back(g_rows[i - 1]);
//           target_indices.push_back(i);
//       }

//       if (!target_indices.empty()) {
//           auto p1_vec = hal::concatenate(ctx, p1_batch, 0);
//           auto p2_vec = hal::concatenate(ctx, p2_batch, 0);
//           auto g1_vec = hal::concatenate(ctx, g1_batch, 0);
//           auto g2_vec = hal::concatenate(ctx, g2_batch, 0);

//           auto [p3_vec, g3_vec] = VectorizedNoteFunc(ctx, p1_vec, p2_vec,
//           g1_vec, g2_vec);

//           for (size_t k = 0; k < target_indices.size(); ++k) {
//               int idx = target_indices[k];
//               p_tree[idx][0] = hal::slice(ctx, p3_vec,
//               {static_cast<int64_t>(k), 0},
//                                           {static_cast<int64_t>(k + 1),
//                                           total_block_size}, {});
//               g_tree[idx][0] = hal::slice(ctx, g3_vec,
//               {static_cast<int64_t>(k), 0},
//                                           {static_cast<int64_t>(k + 1), 1},
//                                           {});
//           }
//       }
//   }

//   for (int j = 1; j < logn; ++j) {
//       int step = 1 << (j + 1);
//       std::vector<Value> p1_batch, p2_batch, g1_batch, g2_batch;
//       std::vector<int> target_indices;

//       for (int i = step - 1; i < current_n; i += step) {
//           int prev_idx = i - (1 << j);
//           p1_batch.push_back(p_tree[i][j - 1]);
//           p2_batch.push_back(p_tree[prev_idx][j - 1]);
//           g1_batch.push_back(g_tree[i][j - 1]);
//           g2_batch.push_back(g_tree[prev_idx][j - 1]);
//           target_indices.push_back(i);
//       }

//       if (!target_indices.empty()) {
//           auto p1_vec = hal::concatenate(ctx, p1_batch, 0);
//           auto p2_vec = hal::concatenate(ctx, p2_batch, 0);
//           auto g1_vec = hal::concatenate(ctx, g1_batch, 0);
//           auto g2_vec = hal::concatenate(ctx, g2_batch, 0);

//           auto [p3_vec, g3_vec] = VectorizedNoteFunc(ctx, p1_vec, p2_vec,
//           g1_vec, g2_vec);

//           for (size_t k = 0; k < target_indices.size(); ++k) {
//               int idx = target_indices[k];
//               p_tree[idx][j] = hal::slice(ctx, p3_vec,
//               {static_cast<int64_t>(k), 0},
//                                           {static_cast<int64_t>(k + 1),
//                                           total_block_size}, {});
//               g_tree[idx][j] = hal::slice(ctx, g3_vec,
//               {static_cast<int64_t>(k), 0},
//                                           {static_cast<int64_t>(k + 1), 1},
//                                           {});
//           }
//       }
//   }

//   // --------------------------------------------------------
//   // Down-Sweep
//   // --------------------------------------------------------
//   res[0] = p_rows[0];

//   for (int j = 0; j < logn; ++j) {
//       int idx = (1 << (j + 1)) - 1;
//       if (idx < current_n) {
//           res[idx] = p_tree[idx][j];
//       }
//   }

//   for (int j = logn - 3; j >= 0; --j) {
//       int step = 1 << (j + 2);
//       int half_step = 1 << (j + 1);

//       std::vector<Value> p1_batch, p2_batch, g1_batch;
//       std::vector<int> target_indices;

//       for (int k = 1; k < current_n / step + 1; ++k) {
//           int idx_curr = current_n - half_step * (2 * k - 1) - 1;
//           int idx_prev = current_n - half_step * 2 * k - 1;

//           if (idx_curr >= 0 && idx_prev >= 0 && idx_curr < current_n) {
//               p1_batch.push_back(p_tree[idx_curr][j]);
//               p2_batch.push_back(res[idx_prev]);
//               g1_batch.push_back(g_tree[idx_curr][j]);
//               target_indices.push_back(idx_curr);
//           }
//       }

//       if (!target_indices.empty()) {
//           auto p1_vec = hal::concatenate(ctx, p1_batch, 0);
//           auto p2_vec = hal::concatenate(ctx, p2_batch, 0);
//           auto g1_vec = hal::concatenate(ctx, g1_batch, 0);

//           auto p3_vec = VectorizedNoteFunc(ctx, p1_vec, p2_vec, g1_vec);

//           for (size_t k = 0; k < target_indices.size(); ++k) {
//               res[target_indices[k]] = hal::slice(ctx, p3_vec,
//               {static_cast<int64_t>(k), 0},
//                   {static_cast<int64_t>(k + 1), total_block_size}, {});
//           }
//       }
//   }

//   {
//       std::vector<Value> p1_batch, p2_batch, g1_batch;
//       std::vector<int> target_indices;

//       for (int k = 1; k < current_n / 2 + 1; ++k) {
//           int idx_curr = current_n - 2 * k;
//           int idx_prev = current_n - 2 * k - 1;

//           if (idx_curr >= 0 && idx_prev >= 0 && idx_curr < current_n) {
//               p1_batch.push_back(p_rows[idx_curr]);
//               p2_batch.push_back(res[idx_prev]);
//               g1_batch.push_back(g_rows[idx_curr]);
//               target_indices.push_back(idx_curr);
//           }
//       }

//       if (!target_indices.empty()) {
//           auto p1_vec = hal::concatenate(ctx, p1_batch, 0);
//           auto p2_vec = hal::concatenate(ctx, p2_batch, 0);
//           auto g1_vec = hal::concatenate(ctx, g1_batch, 0);

//           auto p3_vec = VectorizedNoteFunc(ctx, p1_vec, p2_vec, g1_vec);

//           for (size_t k = 0; k < target_indices.size(); ++k) {
//               res[target_indices[k]] = hal::slice(ctx, p3_vec,
//               {static_cast<int64_t>(k), 0},
//                   {static_cast<int64_t>(k + 1), total_block_size}, {});
//           }
//       }
//   }

//   // --- 3. 输出后处理 ---
//   auto final_payload = hal::concatenate(ctx, res, 0);

//   // 切分回 x 和 valid
//   auto y_out_padded = hal::slice(ctx, final_payload, {0, 0}, {current_n,
//   block_size}, {}); auto valid_out_padded = hal::slice(ctx, final_payload,
//   {0, block_size}, {current_n, total_block_size}, {});

//   // 4. 去除 Padding
//   if (n_padded > n) {
//       y_out_padded = hal::slice(ctx, y_out_padded, {0, 0}, {n, block_size},
//       {}); valid_out_padded = hal::slice(ctx, valid_out_padded, {0, 0}, {n,
//       block_size}, {});
//   }

//   return {y_out_padded, valid_out_padded};
// }
// ------------------------------------------------------------------------------------

// ====================================================================================
// Vectorized AggregateBrentKung without valid bits
Value AggregateBrentKung_without_valids(SPUContext* ctx, const Value& x_full,
                                        const Value& g_full) {
  const int64_t n = x_full.shape()[0];
  const int64_t block_size = x_full.shape()[1];
  const int64_t logn = std::floor(std::log2(n));

  // 预处理：切分输入行，准备 Level 0
  std::vector<Value> x_rows(n);
  std::vector<Value> g_rows(n);

  for (int i = 0; i < n; ++i) {
    // Slice 出来是 [1, Block] 和 [1, 1]
    x_rows[i] = hal::slice(ctx, x_full, {i, 0}, {i + 1, block_size}, {});
    g_rows[i] = hal::slice(ctx, g_full, {i, 0}, {i + 1, 1}, {});
    // Reshape 成 [1, Block] 和 [1, 1] 以便后续 Batch 拼接
    // 注意：原代码 reshape 成了 [Block]，这里为了 batch concatenate 方便，保持
    // rank=2 更好， 即 [1, Block]。所有的 NoteFunc 输入都预期是 [Batch,
    // Block]。
    x_rows[i] = hal::reshape(ctx, x_rows[i], {1, block_size});
    g_rows[i] = hal::reshape(ctx, g_rows[i], {1, 1});
  }

  std::vector<std::vector<Value>> p_tree(n, std::vector<Value>(logn));
  std::vector<std::vector<Value>> g_tree(n, std::vector<Value>(logn));
  std::vector<Value> res(n);

  // --------------------------------------------------------
  // Up-Sweep (Parallelized)
  // --------------------------------------------------------

  // Level 0: 处理 x_rows
  {
    std::vector<Value> p1_batch, p2_batch, g1_batch, g2_batch;
    std::vector<int> target_indices;

    for (int i = 1; i < n; i += 2) {
      p1_batch.push_back(x_rows[i]);
      p2_batch.push_back(x_rows[i - 1]);
      g1_batch.push_back(g_rows[i]);
      g2_batch.push_back(g_rows[i - 1]);
      target_indices.push_back(i);
    }

    if (!target_indices.empty()) {
      auto p1_vec = hal::concatenate(ctx, p1_batch, 0);
      auto p2_vec = hal::concatenate(ctx, p2_batch, 0);
      auto g1_vec = hal::concatenate(ctx, g1_batch, 0);
      auto g2_vec = hal::concatenate(ctx, g2_batch, 0);

      // 并行计算
      auto [p3_vec, g3_vec] =
          VectorizedNoteFunc(ctx, p1_vec, p2_vec, g1_vec, g2_vec);

      // 拆分结果存回 Tree
      for (size_t k = 0; k < target_indices.size(); ++k) {
        int idx = target_indices[k];
        // Slice: [k, k+1]
        p_tree[idx][0] =
            hal::slice(ctx, p3_vec, {static_cast<int64_t>(k), 0},
                       {static_cast<int64_t>(k + 1), block_size}, {});
        g_tree[idx][0] = hal::slice(ctx, g3_vec, {static_cast<int64_t>(k), 0},
                                    {static_cast<int64_t>(k + 1), 1}, {});
      }
    }
  }

  // Levels 1 to logn-1
  for (int j = 1; j < logn; ++j) {
    int step = 1 << (j + 1);
    std::vector<Value> p1_batch, p2_batch, g1_batch, g2_batch;
    std::vector<int> target_indices;

    for (int i = step - 1; i < n; i += step) {
      int prev_idx = i - (1 << j);
      p1_batch.push_back(p_tree[i][j - 1]);
      p2_batch.push_back(p_tree[prev_idx][j - 1]);
      g1_batch.push_back(g_tree[i][j - 1]);
      g2_batch.push_back(g_tree[prev_idx][j - 1]);
      target_indices.push_back(i);
    }

    if (!target_indices.empty()) {
      auto p1_vec = hal::concatenate(ctx, p1_batch, 0);
      auto p2_vec = hal::concatenate(ctx, p2_batch, 0);
      auto g1_vec = hal::concatenate(ctx, g1_batch, 0);
      auto g2_vec = hal::concatenate(ctx, g2_batch, 0);

      auto [p3_vec, g3_vec] =
          VectorizedNoteFunc(ctx, p1_vec, p2_vec, g1_vec, g2_vec);

      for (size_t k = 0; k < target_indices.size(); ++k) {
        int idx = target_indices[k];
        p_tree[idx][j] =
            hal::slice(ctx, p3_vec, {static_cast<int64_t>(k), 0},
                       {static_cast<int64_t>(k + 1), block_size}, {});
        g_tree[idx][j] = hal::slice(ctx, g3_vec, {static_cast<int64_t>(k), 0},
                                    {static_cast<int64_t>(k + 1), 1}, {});
      }
    }
  }

  // --------------------------------------------------------
  // Down-Sweep (Parallelized)
  // --------------------------------------------------------

  // 初始化 res[0]
  res[0] = x_rows[0];

  // Copy computed roots
  for (int j = 0; j < logn; ++j) {
    int idx = (1 << (j + 1)) - 1;
    if (idx < n) {
      res[idx] = p_tree[idx][j];
    }
  }

  // Phase 1 of Down-Sweep (Internal Nodes)
  for (int j = logn - 3; j >= 0; --j) {
    int step = 1 << (j + 2);
    int half_step = 1 << (j + 1);

    std::vector<Value> p1_batch, p2_batch, g1_batch;
    std::vector<int> target_indices;

    for (int k = 1; k < n / step + 1; ++k) {
      int idx_curr = n - half_step * (2 * k - 1) - 1;
      int idx_prev = n - half_step * 2 * k - 1;

      if (idx_curr >= 0 && idx_prev >= 0 && idx_curr < n) {
        // NoteFunc: res[curr] = NoteFunc(tree[curr], res[prev], tree_g[curr])
        // 注意参数对应关系：p1=tree[curr], p2=res[prev], g1=tree_g[curr]
        p1_batch.push_back(p_tree[idx_curr][j]);
        p2_batch.push_back(res[idx_prev]);
        g1_batch.push_back(g_tree[idx_curr][j]);
        target_indices.push_back(idx_curr);
      }
    }

    if (!target_indices.empty()) {
      auto p1_vec = hal::concatenate(ctx, p1_batch, 0);
      auto p2_vec = hal::concatenate(ctx, p2_batch, 0);
      auto g1_vec = hal::concatenate(ctx, g1_batch, 0);

      // 调用 3参数版本的 VectorizedNoteFunc
      auto p3_vec = VectorizedNoteFunc(ctx, p1_vec, p2_vec, g1_vec);

      for (size_t k = 0; k < target_indices.size(); ++k) {
        res[target_indices[k]] =
            hal::slice(ctx, p3_vec, {static_cast<int64_t>(k), 0},
                       {static_cast<int64_t>(k + 1), block_size}, {});
      }
    }
  }

  // Phase 2 of Down-Sweep (Leaves)
  {
    std::vector<Value> p1_batch, p2_batch, g1_batch;
    std::vector<int> target_indices;

    for (int k = 1; k < n / 2 + 1; ++k) {
      int idx_curr = n - 2 * k;
      int idx_prev = n - 2 * k - 1;

      if (idx_curr >= 0 && idx_prev >= 0 && idx_curr < n) {
        p1_batch.push_back(x_rows[idx_curr]);
        p2_batch.push_back(res[idx_prev]);
        g1_batch.push_back(g_rows[idx_curr]);
        target_indices.push_back(idx_curr);
      }
    }

    if (!target_indices.empty()) {
      auto p1_vec = hal::concatenate(ctx, p1_batch, 0);
      auto p2_vec = hal::concatenate(ctx, p2_batch, 0);
      auto g1_vec = hal::concatenate(ctx, g1_batch, 0);

      auto p3_vec = VectorizedNoteFunc(ctx, p1_vec, p2_vec, g1_vec);

      for (size_t k = 0; k < target_indices.size(); ++k) {
        res[target_indices[k]] =
            hal::slice(ctx, p3_vec, {static_cast<int64_t>(k), 0},
                       {static_cast<int64_t>(k + 1), block_size}, {});
      }
    }
  }

  // --- Reshape & Concatenate Output ---
  // 此时 res 中的每个元素已经是 [1, BlockSize] 形状
  return hal::concatenate(ctx, res, 0);
}
// ------------------------------------------------------------------------------------

// ====================================================================================
// NonVectorized AggregateBrentKung without valid bits
// NoteFunc with input g2
static std::pair<Value, Value> NoteFunc(SPUContext* ctx, const Value& p1,
                                        const Value& p2, const Value& g1,
                                        const Value& g2) {
  // g3 = g1 * g2
  auto g3 = hal::mul(ctx, g1, g2);

  // p3 = p1 + (p2 - p1) * g1
  auto diff = hal::sub(ctx, p2, p1);

  Value g1_broadcasted = g1;
  if (g1.shape() != diff.shape()) {
    g1_broadcasted = hal::broadcast_to(ctx, g1, diff.shape());
  }

  auto term = hal::mul(ctx, diff, g1_broadcasted);
  auto p3 = hal::add(ctx, p1, term);

  return std::make_pair(p3, g3);
}
// NoteFunc without input g2
static Value NoteFunc(SPUContext* ctx, const Value& p1, const Value& p2,
                      const Value& g1) {
  auto diff = hal::sub(ctx, p2, p1);

  Value g1_broadcasted = g1;
  if (g1.shape() != diff.shape()) {
    g1_broadcasted = hal::broadcast_to(ctx, g1, diff.shape());
  }

  auto term = hal::mul(ctx, diff, g1_broadcasted);
  auto p3 = hal::add(ctx, p1, term);
  return p3;
}
Value AggregateBrentKung_NonVectorized(SPUContext* ctx, const Value& x_full,
                                       const Value& g_full) {
  const int64_t n = x_full.shape()[0];
  const int64_t block_size = x_full.shape()[1];
  const int64_t logn = std::floor(std::log2(n));

  std::vector<Value> x_rows(n);
  std::vector<Value> g_rows(n);

  for (int i = 0; i < n; ++i) {
    x_rows[i] = hal::slice(ctx, x_full, {i, 0}, {i + 1, block_size}, {});
    g_rows[i] = hal::slice(ctx, g_full, {i, 0}, {i + 1, 1}, {});

    x_rows[i] = hal::reshape(ctx, x_rows[i], {block_size});
    g_rows[i] = hal::reshape(ctx, g_rows[i], {1});
  }

  std::vector<std::vector<Value>> p_tree(n, std::vector<Value>(logn));
  std::vector<std::vector<Value>> g_tree(n, std::vector<Value>(logn));
  std::vector<Value> res(n);

  // --- Up-Sweep ---
  for (int i = 1; i < n; i += 2) {
    auto result =
        NoteFunc(ctx, x_rows[i], x_rows[i - 1], g_rows[i], g_rows[i - 1]);
    p_tree[i][0] = result.first;
    g_tree[i][0] = result.second;
  }

  for (int j = 1; j < logn; ++j) {
    int step = 1 << (j + 1);
    for (int i = step - 1; i < n; i += step) {
      int prev_idx = i - (1 << j);
      auto result = NoteFunc(ctx, p_tree[i][j - 1], p_tree[prev_idx][j - 1],
                             g_tree[i][j - 1], g_tree[prev_idx][j - 1]);
      p_tree[i][j] = result.first;
      g_tree[i][j] = result.second;
    }
  }

  // --- Down-Sweep ---
  res[0] = x_rows[0];

  for (int j = 0; j < logn; ++j) {
    int idx = (1 << (j + 1)) - 1;
    if (idx < n) {
      res[idx] = p_tree[idx][j];
    }
  }

  for (int j = logn - 3; j >= 0; --j) {
    int step = 1 << (j + 2);
    int half_step = 1 << (j + 1);

    for (int k = 1; k < n / step + 1; ++k) {
      int idx_curr = n - half_step * (2 * k - 1) - 1;
      int idx_prev = n - half_step * 2 * k - 1;

      if (idx_curr >= 0 && idx_prev >= 0 && idx_curr < n) {
        res[idx_curr] = NoteFunc(ctx, p_tree[idx_curr][j], res[idx_prev],
                                 g_tree[idx_curr][j]);
      }
    }
  }

  for (int k = 1; k < n / 2 + 1; ++k) {
    int idx_curr = n - 2 * k;
    int idx_prev = n - 2 * k - 1;

    if (idx_curr >= 0 && idx_prev >= 0 && idx_curr < n) {
      res[idx_curr] =
          NoteFunc(ctx, x_rows[idx_curr], res[idx_prev], g_rows[idx_curr]);
    }
  }

  // --- Reshape & Concatenate ---
  std::vector<Value> res_reshaped;
  for (auto& v : res) {
    res_reshaped.push_back(hal::reshape(ctx, v, {1, block_size}));
  }

  return hal::concatenate(ctx, res_reshaped, 0);
}
// ------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------
// // 支持多行 x 的 extract_ordered
// std::pair<std::vector<spu::Value>, int64_t> extract_ordered(SPUContext* ctx,
// const spu::Value& x_in, const spu::Value& valids) {
//   // 兼容性处理：如果 x 不是 2D，就把它 reshape 成 2D（1 行）
//   spu::Value x = x_in;
//   if (x.shape().ndim() == 1) {
//     const int64_t n1 = x.numel();
//     x = hal::reshape(ctx, x, {1, n1});
//   } else if (x.shape().ndim() == 0) {
//     // scalar -> 1x1
//     x = hal::reshape(ctx, x, {1, 1});
//   }
//   SPU_ENFORCE(x.shape().ndim() == 2, "x should be 2D array");
//   SPU_ENFORCE(valids.shape().ndim() == 2 && valids.shape()[0] == 1,
//               "valids should be 1-row matrix");

//   const int64_t num_arrays = x.shape()[0];  // x行数
//   const int64_t n = x.shape()[1];           // x长度

//   SPU_ENFORCE(valids.shape()[1] == n,
//               "valids length must match x's second dimension");

//   // 1. 计算valids前缀和rho（保持一维）
//   auto rho = prefix_sum(ctx, valids);

//   // 2. 将所有 x 的行分离出来，与 valids 一起洗牌
//   std::vector<spu::Value> inputs_to_shuffle;
//   for (int64_t i = 0; i < num_arrays; ++i) {
//     auto x_row = hal::slice(ctx, x, {i, 0}, {i + 1, n}, {});
//     inputs_to_shuffle.push_back(x_row);
//   }
//   inputs_to_shuffle.push_back(valids);
//   inputs_to_shuffle.push_back(rho);

//   auto shuffled_results = hlo::Shuffle(ctx, inputs_to_shuffle, 1);

//   // 分离洗牌后的结果
//   std::vector<spu::Value> sx_rows(num_arrays);
//   for (int64_t i = 0; i < num_arrays; ++i) {
//     sx_rows[i] = shuffled_results[i];
//   }
//   auto svalids = shuffled_results[num_arrays];
//   auto srho = shuffled_results[num_arrays + 1];

//   // 3. 打开 svalids
//   auto svalids_open = hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx,
//   svalids));

//   // 4. 计算公开置换 compact，使得 compact(svalids) = [1, 1, ..., 0, 0]
//   int64_t numel = svalids_open.size();
//   std::vector<int64_t> p_hat_indices(numel);

//   int64_t left = 0;
//   int64_t right = numel - 1;
//   for (int64_t i = 0; i < numel; ++i) {
//     if (svalids_open[i]) {
//       p_hat_indices[left++] = i;  // valid放前面
//     } else {
//       p_hat_indices[right--] = i; // dummy放后面
//     }
//   }
//   int64_t valid_count = left;

//   // 5. 用公开的 compact 置换 sx_rows 和 srho
//   auto p_hat_xt = xt::adapt(p_hat_indices);
//   spu::Value compact = hal::constant(ctx, p_hat_xt, spu::DT_I64,
//     {static_cast<int64_t>(p_hat_indices.size())});

//   std::vector<spu::Value> inputs_to_permute;
//   for (auto& sx_row : sx_rows) {
//     inputs_to_permute.push_back(sx_row);
//   }
//   inputs_to_permute.push_back(srho);

//   std::vector<spu::Value> compacted_results = hlo::Permute(ctx,
//   inputs_to_permute, compact, 1);

//   // 分离结果
//   std::vector<spu::Value> x_prime_rows(num_arrays);
//   for (int64_t i = 0; i < num_arrays; ++i) {
//     x_prime_rows[i] = compacted_results[i];
//   }
//   auto rho_prime = compacted_results[num_arrays];

//   // 6.
//   // rho_prime_processed = Open( rho_prime[0，valid_count] ) || [valid_count,
//   n]
//   // rho_prime_processed 逆置换 x_prime_rows
//   xt::xarray<int64_t> rho_prime_processed;
//   if (valid_count > 0) {
//     auto rho_prime_slice = hal::slice(ctx, rho_prime, {0, 0}, {1,
//     valid_count}, {}); auto rho_prime_slice_open =
//     hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx, rho_prime_slice));

//     rho_prime_slice_open = rho_prime_slice_open - 1;
//     auto flatted = xt::ravel(rho_prime_slice_open);
//     auto tail = xt::arange<int64_t>(valid_count, n);
//     rho_prime_processed = xt::concatenate(xt::xtuple(flatted, tail));
//   } else {
//     rho_prime_processed = xt::arange<int64_t>(n);
//   }
//   spu::Value rho_prime_constant = hal::constant(ctx, rho_prime_processed,
//   spu::DT_I64, {n}); std::vector<spu::Value> y = hlo::InvPermute(ctx,
//   x_prime_rows, rho_prime_constant, 1);

//   return {y, valid_count};
// }
// ------------------------------------------------------------------------------------

}  // namespace spu::kernel::hlo