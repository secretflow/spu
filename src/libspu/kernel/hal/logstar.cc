#include "libspu/kernel/hal/logstar.h"

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

namespace spu::kernel::hal {

// Flatting the indices.
static spu::Index flat_indices(const spu::Index& blocks, int64_t width) {
  if (width == 1) {
    return blocks;
  }
  spu::Index expanded;
  size_t total_size = blocks.size() * width;
  expanded.resize(total_size);

  int64_t* out_ptr = expanded.data();
  const int64_t* in_ptr = blocks.data();
  size_t n_blocks = blocks.size();

  for (size_t i = 0; i < n_blocks; ++i) {
    int64_t base = in_ptr[i] * width;
    for (int64_t k = 0; k < width; ++k) {
      out_ptr[i * width + k] = base + k;
    }
  }
  return expanded;
}

spu::Value _2s(SPUContext* ctx, const Value& data) {
  if (data.isPublic()) {
    return kernel::hal::_p2s(ctx, data);
  } else if (data.isPrivate()) {
    return kernel::hal::_v2s(ctx, data);
  }
  return data;
}

static Value mutable_copy(SPUContext* ctx, const Value& input) {
  Value copy;
  if (!input.isSecret()) {
    copy = _2s(ctx, input.clone()).setDtype(input.dtype());
  } else {
    copy = input.clone();
  }
  return kernel::hal::_prefer_a(ctx, copy);
}

/**
 * @brief The associative operator for the Up-Sweep phase.
 *
 * Logic:
 *   g3 = g1 * g2,
 *   p3 = p1 + (p2 - p1) * g1  (Multiplexer)
 */
static std::pair<Value, Value> node_func(SPUContext* ctx, const Value& p1,
                                         const Value& p2, const Value& g1,
                                         const Value& g2) {
  auto g3 = kernel::hal::mul(ctx, g1, g2);
  auto diff = kernel::hal::sub(ctx, p2, p1);
  Value g1_broadcasted = kernel::hal::broadcast_to(ctx, g1, diff.shape());
  auto term = kernel::hal::mul(ctx, diff, g1_broadcasted);
  auto p3 = kernel::hal::add(ctx, p1, term);
  return {p3, g3};
}

/**
 * @brief The associative operator for the Down-Sweep phase (Compute 'p' only).
 *
 * Logic:
 *   p3 = p1 + (p2 - p1) * g1  (Multiplexer)
 */
// TODO: 乘法有待优化
static Value node_func(SPUContext* ctx, const Value& p1, const Value& p2,
                       const Value& g1) {
  auto diff = kernel::hal::sub(ctx, p2, p1);
  Value g1_broadcasted = kernel::hal::broadcast_to(ctx, g1, diff.shape());
  auto term = kernel::hal::mul(ctx, diff, g1_broadcasted);
  auto p3 = kernel::hal::add(ctx, p1, term);
  return p3;
}

/**
 * @brief The Duplication step of Logstar based on Brent-Kung network. Duplicate
 * the inputs according to group signals.
 *
 * Logic:
 *   For i = 1, ..., n-1:
 *      x_out[i] = x[i], valids_out[i] = valids[i],                if g[i] == 0;
 *      x_out[i] = x_out[i-1], valids_out[i] = valids_out[i-1],    if g[i] == 1.
 *
 * @param x Input data blocks [n, block_size].
 * @param valids Valid bits of each data [n, block_size].
 * @param g_in Group signals of each block [n, 1].
 * @return std::pair<Value, Value> Duplicated inputs {x_out, valids_out}.
 */
std::pair<Value, Value> duplicate_brent_kung(SPUContext* ctx, const Value& x,
                                             const Value& valids,
                                             const Value& g_in) {
  const int64_t n = x.shape()[0];
  const int64_t block_size = x.shape()[1];
  const int64_t total_block_size = block_size * 2;

  // p: [n, 2 * block_size]
  Value p = kernel::hal::concatenate(ctx, {x, valids}, 1);
  p = kernel::hal::_prefer_a(ctx, p);
  Value g = mutable_copy(ctx, g_in);

  // Get flattened views of the data for scatter/gather operations
  auto p_flat_ref = p.data().reshape({p.numel()});
  auto g_flat_ref = g.data().reshape({g.numel()});

  int depth = Log2Ceil(n);
  size_t max_blocks = (n / 2) + 1;
  spu::Index idx_right_blocks;
  idx_right_blocks.reserve(max_blocks);
  spu::Index idx_left_blocks;
  idx_left_blocks.reserve(max_blocks);
  spu::Index idx_root_blocks;
  idx_root_blocks.reserve(max_blocks);
  spu::Index idx_child_blocks;
  idx_child_blocks.reserve(max_blocks);

  // --- 1. Up-Sweep (Reduce Phase) ---
  // This phase builds a tree from leaves to root.
  for (int j = 0; j < depth; ++j) {
    int64_t step = 1LL << (j + 1);      // Distance between nodes to update
    int64_t left_child_off = 1LL << j;  // Distance to the left child

    idx_right_blocks.clear();
    idx_left_blocks.clear();

    // Identify which blocks (nodes) participate in this level
    for (int64_t i = step - 1; i < n; i += step) {
      idx_right_blocks.push_back(i);
      idx_left_blocks.push_back(i - left_child_off);
    }

    if (idx_right_blocks.empty()) continue;

    // Flatting block indices
    spu::Index idx_right_elems =
        flat_indices(idx_right_blocks, total_block_size);
    spu::Index idx_left_elems = flat_indices(idx_left_blocks, total_block_size);
    const spu::Index& idx_right_g = idx_right_blocks;
    const spu::Index& idx_left_g = idx_left_blocks;

    // Gather values for participating nodes into temporary compact tensors
    Value v_right_p(p_flat_ref.linear_gather(idx_right_elems), p.dtype());
    Value v_left_p(p_flat_ref.linear_gather(idx_left_elems), p.dtype());
    Value v_right_g(g_flat_ref.linear_gather(idx_right_g), g.dtype());
    Value v_left_g(g_flat_ref.linear_gather(idx_left_g), g.dtype());

    // Restore shapes for vectorized computation
    int64_t k = idx_right_blocks.size();
    v_right_p = kernel::hal::reshape(ctx, v_right_p, {k, total_block_size});
    v_left_p = kernel::hal::reshape(ctx, v_left_p, {k, total_block_size});
    v_right_g = kernel::hal::reshape(ctx, v_right_g, {k, 1});
    v_left_g = kernel::hal::reshape(ctx, v_left_g, {k, 1});

    // Apply node function in batch
    auto [new_p, new_g] =
        node_func(ctx, v_right_p, v_left_p, v_right_g, v_left_g);

    // Write results back to the main memory buffer
    auto new_p_data = new_p.data().reshape({new_p.numel()});
    auto new_g_data = new_g.data().reshape({new_g.numel()});
    p_flat_ref.linear_scatter(new_p_data, idx_right_elems);
    g_flat_ref.linear_scatter(new_g_data, idx_right_g);
  }

  // --- 2. Down-Sweep (Distribute Phase) ---
  // Traverse back down the tree.
  for (int j = depth - 2; j >= 0; --j) {
    int64_t step = 1LL << (j + 1);
    int64_t dist = 1LL << j;

    idx_root_blocks.clear();
    idx_child_blocks.clear();

    // Identify nodes: 'Root' pushes its value to 'Child'
    for (int64_t i = step - 1; i < n; i += step) {
      int64_t target = i + dist;
      if (target < n) {
        idx_root_blocks.push_back(i);
        idx_child_blocks.push_back(target);
      }
    }

    if (idx_child_blocks.empty()) continue;

    spu::Index idx_root_elems = flat_indices(idx_root_blocks, total_block_size);
    spu::Index idx_child_elems =
        flat_indices(idx_child_blocks, total_block_size);
    const spu::Index& idx_child_g = idx_child_blocks;

    Value v_root_p(p_flat_ref.linear_gather(idx_root_elems), p.dtype());
    Value v_child_p(p_flat_ref.linear_gather(idx_child_elems), p.dtype());
    Value v_child_g(g_flat_ref.linear_gather(idx_child_g), g.dtype());

    int64_t k = idx_child_blocks.size();
    v_root_p = kernel::hal::reshape(ctx, v_root_p, {k, total_block_size});
    v_child_p = kernel::hal::reshape(ctx, v_child_p, {k, total_block_size});
    v_child_g = kernel::hal::reshape(ctx, v_child_g, {k, 1});

    auto new_p_child = node_func(ctx, v_child_p, v_root_p, v_child_g);

    auto new_p_data = new_p_child.data().reshape({new_p_child.numel()});
    p_flat_ref.linear_scatter(new_p_data, idx_child_elems);
  }

  // Slice the concatenated 'p' back into outputs
  auto x_out = kernel::hal::slice(ctx, p, {0, 0}, {n, block_size}, {});
  auto valids_out =
      kernel::hal::slice(ctx, p, {0, block_size}, {n, 2 * block_size}, {});

  return {x_out, valids_out};
}

/**
 * @brief Extract the values with conditions = 1 to the front of the vector
 * while keeping the order of the values. The number of values that satisfy
 * condition = 1 will be revealed.
 *
 * @param x Input values.
 * @param conditions Extracting conditions.
 * @return Output values and the number of values that satisfy condition = 1.
 */
std::pair<std::vector<spu::Value>, int64_t> extract_ordered(
    SPUContext* ctx, const spu::Value& x_in, const spu::Value& conditions) {
  // Compatibility handling: if x is not 2D, reshape it to 2D (1 line)
  spu::Value x = x_in;
  if (x.shape().ndim() == 1) {
    const int64_t n1 = x.numel();
    x = hal::reshape(ctx, x, {1, n1});
  } else if (x.shape().ndim() == 0) {
    // scalar -> 1x1
    x = hal::reshape(ctx, x, {1, 1});
  }
  SPU_ENFORCE(x.shape().ndim() == 2, "x should be 2D array");
  SPU_ENFORCE(conditions.shape().ndim() == 2 && conditions.shape()[0] == 1,
              "conditions should be 1-row matrix");

  const int64_t num_arrays = x.shape()[0];
  const int64_t n = x.shape()[1];

  SPU_ENFORCE(conditions.shape()[1] == n,
              "conditions length must match x's second dimension");

  // Prefix sum of conditions
  auto rho = hal::associative_scan(hal::add, ctx, conditions);
  // Shuffle x, conditions, and rho
  std::vector<spu::Value> inputs_to_shuffle;
  inputs_to_shuffle.reserve(num_arrays + 2);
  for (int64_t i = 0; i < num_arrays; ++i) {
    auto x_row = hal::slice(ctx, x, {i, 0}, {i + 1, n}, {});
    inputs_to_shuffle.push_back(x_row);
  }
  inputs_to_shuffle.push_back(conditions);
  inputs_to_shuffle.push_back(rho);

  auto shuffled_results = hlo::Shuffle(ctx, inputs_to_shuffle, 1);

  // Separate the shuffling results
  std::vector<spu::Value> sx_rows(num_arrays);
  for (int64_t i = 0; i < num_arrays; ++i) {
    sx_rows[i] = shuffled_results[i];
  }
  auto scondition = shuffled_results[num_arrays];
  auto srho = shuffled_results[num_arrays + 1];

  // Open scondition
  auto scondition_open =
      hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx, scondition));

  // Compute the public permutation 'compact', such that compact(scondition) =
  // [1, 1, ..., 0, 0]
  int64_t numel = scondition_open.size();
  std::vector<int64_t> p_hat_indices(numel);

  int64_t left = 0;
  int64_t right = numel - 1;
  for (int64_t i = 0; i < numel; ++i) {
    if (scondition_open[i] != 0) {
      p_hat_indices[left++] = i;
    } else {
      p_hat_indices[right--] = i;
    }
  }
  int64_t valid_count = left;

  // Permute sx_rows and srho using 'compact'
  auto p_hat_xt = xt::adapt(p_hat_indices);
  spu::Value compact = hal::constant(
      ctx, p_hat_xt, spu::DT_I64, {static_cast<int64_t>(p_hat_indices.size())});

  std::vector<spu::Value> inputs_to_permute;
  inputs_to_permute.reserve(sx_rows.size() + 1);
  for (auto& sx_row : sx_rows) {
    inputs_to_permute.push_back(sx_row);
  }
  inputs_to_permute.push_back(srho);

  std::vector<spu::Value> compacted_results =
      hlo::Permute(ctx, inputs_to_permute, compact, 1);

  // Seperate the permuting results
  std::vector<spu::Value> x_prime_rows(num_arrays);
  for (int64_t i = 0; i < num_arrays; ++i) {
    x_prime_rows[i] = compacted_results[i];
  }
  auto rho_prime = compacted_results[num_arrays];

  // rho_prime_processed = Open( rho_prime[0，valid_count] ) || [valid_count,n]
  // Inversely permute x_prime_rows using rho_prime_processed
  xt::xarray<int64_t> rho_prime_processed;
  if (valid_count > 0) {
    auto rho_prime_slice =
        hal::slice(ctx, rho_prime, {0, 0}, {1, valid_count}, {});
    auto rho_prime_slice_open =
        hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx, rho_prime_slice));

    rho_prime_slice_open = rho_prime_slice_open - 1;
    auto flatted = xt::ravel(rho_prime_slice_open);
    auto tail = xt::arange<int64_t>(valid_count, n);
    rho_prime_processed = xt::concatenate(xt::xtuple(flatted, tail));
  } else {
    rho_prime_processed = xt::arange<int64_t>(n);
  }
  spu::Value rho_prime_constant =
      hal::constant(ctx, rho_prime_processed, spu::DT_I64, {n});
  std::vector<spu::Value> y =
      hlo::InvPermute(ctx, x_prime_rows, rho_prime_constant, 1);

  return {y, valid_count};
}

// spu::Value ComputeMedians(SPUContext* ctx, const spu::Value& arr, const int
// k,
//                           const int m) {
//   // a. b := im (隐含在 reshape 中)
//   auto reshaped = hal::reshape(ctx, arr, {k, m, arr.shape()[1]});

//   // 分离 key 和 valid (假设最后一列是 valid bit)
//   auto key = hal::slice(ctx, reshaped, {0, 0, 0}, {k, m, 1}, {});
//   auto valid = hal::slice(ctx, reshaped, {0, 0, 1}, {k, m, 2}, {});

//   // 计算 f[i, j]
//   // f[i, 0] = valid[i, 0]
//   // f[i, j] = valid[i, j] && !valid[i, j-1]
//   // 我们可以通过在 m 维度上将 valid 向右移位来实现

//   // shift right: [0, v0, v1, ..., vm-2]
//   auto v_prev_slice =
//       hal::slice(ctx, valid, {0, 0, 0}, {k, m - 1, 1}, {});  // 前 m-1 个
//   auto zeros = hal::constant(ctx, 0, valid.dtype(), {k, 1, 1});
//   zeros = hal::seal(ctx, zeros);
//   auto valid_prev = hal::concatenate(ctx, {zeros, v_prev_slice}, 1);

//   // !valid_prev = 1 - valid_prev
//   auto ones = hal::constant(ctx, 1.0, valid.dtype(), valid.shape());
//   ones = hal::seal(ctx, ones);
//   auto not_valid_prev = hal::sub(ctx, ones, valid_prev);

//   // f = valid * (1 - valid_prev)
//   auto f = hal::mul(ctx, valid, not_valid_prev);

//   // e. Z[i] = sum(X[b+j] * f[i, j])
//   // 广播 f 到数据维度
//   // auto f_bcast = hal::broadcast_to(ctx, f, {k, m, width});
//   auto term = hal::mul(ctx, key, f);

//   // 在维度 1 (m) 上求和。由于 m 通常较小 (logN)，直接用循环累加即可，
//   // 或者使用 reduce_sum 如果 HAL 支持。这里使用切片累加确保兼容性。
//   auto z = hal::slice(ctx, term, {0, 0, 0}, {k, 1, 1}, {});
//   z = hal::reshape(ctx, z, {k});

//   for (int64_t j = 1; j < m; ++j) {
//     auto slice_j = hal::slice(ctx, term, {0, j, 0}, {k, j + 1, 1}, {});
//     slice_j = hal::reshape(ctx, slice_j, {k});
//     z = hal::add(ctx, z, slice_j);
//   }
//   return z;
// }

// spu::Value LogstarRecursive(SPUContext* ctx, const spu::Value& x,
//                             const spu::Value& y) {
//   const int64_t nx = x.shape()[0];
//   const int64_t ny = y.shape()[0];
//   auto list_id_x = hal::seal(ctx, hal::constant(ctx, 0, DT_I1, {nx, 1}));
//   auto list_id_y = hal::seal(ctx, hal::constant(ctx, 1, DT_I1, {ny, 1}));

//   // block parameters
//   int64_t basic_size = 1;
//   auto m = Log2Floor(std::max(nx, ny));
//   if (m == 0) m = 1;
//   // k_x = ceil(nx / m)
//   const int k_x = (nx + m - 1) / m;
//   const int k_y = (ny + m - 1) / m;

//   // pad x and y to make its length a multiple of m
//   spu::Value x_pad = x;
//   spu::Value y_pad = y;
//   if (nx % m != 0) {
//     int64_t padding_len = k_x * m - nx;
//     spu::Value padding =
//         hal::constant(ctx, 0, x.dtype(), {padding_len, x.shape()[1]});
//     padding = hal::seal(ctx, padding);
//     x_pad = hal::concatenate(ctx, {x, padding}, 0);
//   }
//   if (ny % m != 0) {
//     int64_t padding_len = k_y * m - ny;
//     spu::Value padding =
//         hal::constant(ctx, 0, y.dtype(), {padding_len, y.shape()[1]});
//     padding = hal::seal(ctx, padding);
//     y_pad = hal::concatenate(ctx, {y, padding}, 0);
//   }

//   auto revealed1 = hal::dump_public_as<float>(ctx, hal::reveal(ctx, x_pad));
//   auto revealed2 = hal::dump_public_as<float>(ctx, hal::reveal(ctx, y_pad));
//   if (ctx->lctx()->Rank() == 0) {
//     std::cout << "x_pad: " << revealed1 << std::endl;
//     std::cout << "y_pad: " << revealed2 << std::endl;
//   }

//   if (nx <= basic_size) {
//     // TODO
//   } else {
//     auto median_x = ComputeMedians(ctx, x_pad, k_x, m);
//     auto median_y = ComputeMedians(ctx, y_pad, k_y, m);
//     auto revealed1 =
//         hal::dump_public_as<float>(ctx, hal::reveal(ctx, median_x));
//     auto revealed2 =
//         hal::dump_public_as<float>(ctx, hal::reveal(ctx, median_y));
//     if (ctx->lctx()->Rank() == 0) {
//       std::cout << "median_x: " << revealed1 << std::endl;
//       std::cout << "median_y: " << revealed2 << std::endl;
//     }

//     //
//     TODO：对每个块并行调用LogstarRecursive。解法1：直接串行，解法2：把块拼起来做向量计算。
//   }

//   return x;
// }

// spu::Value logstar(SPUContext* ctx, const spu::Value& key_x,
//                    const spu::Value& key_y) {
//   const int64_t nx = key_x.shape()[0];
//   const int64_t ny = key_y.shape()[0];
//   auto dtayp = key_x.dtype();
//   auto valid_x = hal::seal(ctx, hal::constant(ctx, 1.0, dtayp, {nx, 1}));
//   auto valid_y = hal::seal(ctx, hal::constant(ctx, 1.0, dtayp, {ny, 1}));
//   // xt::xarray<int64_t> x_iota = xt::arange<int64_t>(nx);
//   // auto idx_x = hal::seal(ctx, hal::constant(ctx, x_iota, dtayp, {nx, 1}));
//   // xt::xarray<int64_t> y_iota = xt::arange<int64_t>(nx, nx + ny);
//   // auto idx_y = hal::seal(ctx, hal::constant(ctx, y_iota, dtayp, {ny, 1}));

//   auto x = hal::concatenate(ctx, {reshape(ctx, key_x, {nx, 1}), valid_x}, 1);
//   auto y = hal::concatenate(ctx, {reshape(ctx, key_y, {ny, 1}), valid_y}, 1);

//   if (ctx->lctx()->Rank() == 0) {
//     std::cout << "x.shape(): " << x.shape() << std::endl;
//   }

//   return LogstarRecursive(ctx, x, y);

//   // hal::dump_public_as<float>(ctx, hal::reveal(ctx, list_id_x));
//   // auto c_idx_x = hal::dump_public_as<float>(ctx, hal::reveal(ctx, idx_y));
//   // if (ctx->lctx()->Rank() == 0) {
//   //   std::cout << "c_idx_x: " << c_idx_x << std::endl;
//   // }
// }

spu::Value ComputeMedians(SPUContext* ctx, const spu::Value& arr, const int k,
                          const int m) {
  const int64_t batch_size = arr.shape()[0];
  const int64_t n_attr = arr.shape()[2];

  // a. b := im (隐含在 reshape 中)
  auto reshaped = hal::reshape(ctx, arr, {batch_size, k, m, n_attr});

  // 分离 key 和 valid (假设最后一列是 valid bit)
  auto key = hal::slice(ctx, reshaped, {0, 0, 0, 0}, {batch_size, k, m, 1}, {});
  auto valid =
      hal::slice(ctx, reshaped, {0, 0, 0, 1}, {batch_size, k, m, 2}, {});

  // 计算 f[i, j]
  // f[i, 0] = valid[i, 0]
  // f[i, j] = valid[i, j] && !valid[i, j-1]
  // 我们可以通过在 m 维度上将 valid 向右移位来实现

  // shift right: [0, v0, v1, ..., vm-2]
  auto v_prev_slice = hal::slice(ctx, valid, {0, 0, 0, 0},
                                 {batch_size, k, m - 1, 1}, {});  // 前 m-1 个
  auto zeros = hal::constant(ctx, 0, valid.dtype(), {batch_size, k, 1, 1});
  zeros = hal::seal(ctx, zeros);
  auto valid_prev = hal::concatenate(ctx, {zeros, v_prev_slice}, 2);

  // !valid_prev = 1 - valid_prev
  auto ones = hal::constant(ctx, 1.0, valid.dtype(), valid.shape());
  ones = hal::seal(ctx, ones);
  auto not_valid_prev = hal::sub(ctx, ones, valid_prev);

  // f = valid * (1 - valid_prev)
  auto f = hal::mul(ctx, valid, not_valid_prev);

  // e. Z[i] = sum(X[b+j] * f[i, j])
  // 广播 f 到数据维度
  // auto f_bcast = hal::broadcast_to(ctx, f, {batch_size, k, m, width});
  auto term = hal::mul(ctx, key, f);

  // 在维度 2 (m) 上求和。由于 m 通常较小 (logN)，直接用循环累加即可，
  // 或者使用 reduce_sum 如果 HAL 支持。这里使用切片累加确保兼容性。
  auto z = hal::slice(ctx, term, {0, 0, 0, 0}, {batch_size, k, 1, 1}, {});
  z = hal::reshape(ctx, z, {batch_size, k});

  for (int64_t j = 1; j < m; ++j) {
    auto slice_j =
        hal::slice(ctx, term, {0, 0, j, 0}, {batch_size, k, j + 1, 1}, {});
    slice_j = hal::reshape(ctx, slice_j, {batch_size, k});
    z = hal::add(ctx, z, slice_j);
  }
  return z;
}

spu::Value LogstarRecursive(SPUContext* ctx, const spu::Value& x,
                            const spu::Value& y) {
  const int64_t batch_size = x.shape()[0];
  const int64_t nx = x.shape()[1];
  const int64_t ny = y.shape()[1];
  const int64_t n_attr = x.shape()[2];

  // block parameters
  int64_t basic_size = 1;
  auto m = Log2Floor(std::max(nx, ny));
  if (m == 0) m = 1;
  // k_x = ceil(nx / m)
  const int k_x = (nx + m - 1) / m;
  const int k_y = (ny + m - 1) / m;

  // pad x and y to make its length a multiple of m
  spu::Value x_pad = x;
  spu::Value y_pad = y;
  if (nx % m != 0) {
    int64_t padding_len = k_x * m - nx;
    auto max_key = hal::slice(ctx, x, {0, nx - 1, 0}, {batch_size, nx, 1}, {});
    spu::Value padding_key =
        hal::broadcast_to(ctx, max_key, {batch_size, padding_len, 1});
    spu::Value padding_valid = seal(
        ctx, hal::constant(ctx, 0, x.dtype(), {batch_size, padding_len, 1}));
    // spu::Value padding_list_id =
    //     hal::constant(ctx, 0, x.dtype(), {batch_size, padding_len, 1});

    auto orig_list_id = hal::slice(ctx, x, {0, 0, 2}, {batch_size, 1, 3}, {});
    spu::Value padding_list_id =
        hal::broadcast_to(ctx, orig_list_id, {batch_size, padding_len, 1});

    auto padding =
        hal::concatenate(ctx, {padding_key, padding_valid, padding_list_id}, 2);
    x_pad = hal::concatenate(ctx, {x, padding}, 1);
  }
  if (ny % m != 0) {
    int64_t padding_len = k_y * m - ny;
    // spu::Value padding_key =
    //     seal(ctx, hal::constant(ctx, 1000000000, x.dtype(),
    //                             {batch_size, padding_len, 1}));
    auto max_key = hal::slice(ctx, y, {0, ny - 1, 0}, {batch_size, ny, 1}, {});
    spu::Value padding_key =
        hal::broadcast_to(ctx, max_key, {batch_size, padding_len, 1});

    spu::Value padding_valid = seal(
        ctx, hal::constant(ctx, 0, x.dtype(), {batch_size, padding_len, 1}));
    // spu::Value padding_list_id =
    //     hal::constant(ctx, 0, x.dtype(), {batch_size, padding_len, 1});

    auto orig_list_id = hal::slice(ctx, y, {0, 0, 2}, {batch_size, 1, 3}, {});
    spu::Value padding_list_id =
        hal::broadcast_to(ctx, orig_list_id, {batch_size, padding_len, 1});

    auto padding =
        hal::concatenate(ctx, {padding_key, padding_valid, padding_list_id}, 2);
    y_pad = hal::concatenate(ctx, {y, padding}, 1);
  }

  auto revealed1 = hal::dump_public_as<float>(ctx, hal::reveal(ctx, x_pad));
  auto revealed2 = hal::dump_public_as<float>(ctx, hal::reveal(ctx, y_pad));
  if (ctx->lctx()->Rank() == 0) {
    std::cout << "x_pad: " << revealed1 << std::endl;
    std::cout << "y_pad: " << revealed2 << std::endl;
  }

  if (nx <= basic_size) {
    // TODO
  } else {
    // 2.a.
    auto median_x = ComputeMedians(ctx, x_pad, k_x, m);
    auto median_y = ComputeMedians(ctx, y_pad, k_y, m);

    // 打印2.a.结果
    auto revealed1 =
        hal::dump_public_as<float>(ctx, hal::reveal(ctx, median_x));
    auto revealed2 =
        hal::dump_public_as<float>(ctx, hal::reveal(ctx, median_y));
    if (ctx->lctx()->Rank() == 0) {
      std::cout << "median_x: " << revealed1 << std::endl;
      std::cout << "median_y: " << revealed2 << std::endl;
    }

    // 2.cd. 将每块的第一个 x 和 y 值赋值为对应的 median 值
    // 更新 x_pad
    auto reshaped_x = hal::reshape(ctx, x_pad, {batch_size, k_x, m, n_attr});
    auto median_x_reshaped =
        hal::reshape(ctx, median_x, {batch_size, k_x, 1, 1});
    auto first_rest_attr_x = hal::slice(ctx, reshaped_x, {0, 0, 0, 1},
                                        {batch_size, k_x, 1, n_attr}, {});
    auto new_first_x =
        hal::concatenate(ctx, {median_x_reshaped, first_rest_attr_x}, 3);

    if (m > 1) {
      auto rest_m_x = hal::slice(ctx, reshaped_x, {0, 0, 1, 0},
                                 {batch_size, k_x, m, n_attr}, {});
      reshaped_x = hal::concatenate(ctx, {new_first_x, rest_m_x}, 2);
    } else {
      reshaped_x = new_first_x;
    }
    x_pad = hal::reshape(ctx, reshaped_x, {batch_size, k_x * m, n_attr});

    // 更新 y_pad
    auto reshaped_y = hal::reshape(ctx, y_pad, {batch_size, k_y, m, n_attr});
    auto median_y_reshaped =
        hal::reshape(ctx, median_y, {batch_size, k_y, 1, 1});
    auto first_rest_attr_y = hal::slice(ctx, reshaped_y, {0, 0, 0, 1},
                                        {batch_size, k_y, 1, n_attr}, {});
    auto new_first_y =
        hal::concatenate(ctx, {median_y_reshaped, first_rest_attr_y}, 3);

    if (m > 1) {
      auto rest_m_y = hal::slice(ctx, reshaped_y, {0, 0, 1, 0},
                                 {batch_size, k_y, m, n_attr}, {});
      reshaped_y = hal::concatenate(ctx, {new_first_y, rest_m_y}, 2);
    } else {
      reshaped_y = new_first_y;
    }
    y_pad = hal::reshape(ctx, reshaped_y, {batch_size, k_y * m, n_attr});

    // 打印2.cd.结果
    auto revealed3 = hal::dump_public_as<float>(ctx, hal::reveal(ctx, x_pad));
    auto revealed4 = hal::dump_public_as<float>(ctx, hal::reveal(ctx, y_pad));
    if (ctx->lctx()->Rank() == 0) {
      std::cout << "x_pad: " << revealed3 << std::endl;
      std::cout << "y_pad: " << revealed4 << std::endl;
    }

    // 2.be. Merge by medians
    // 构造 keys: [batch_size, k_x + k_y]
    auto keys = hal::concatenate(ctx, {median_x, median_y}, 1);

    // // ****************  使用不支持多维payload的odd_evev_merge时：
    // 构造 payloads: 将每个块内的 m * n_attr 个元素拆分成独立的 payload
    // std::vector<spu::Value> merge_inputs;
    // merge_inputs.reserve(1 + m * n_attr);
    // merge_inputs.push_back(keys);

    // for (int64_t j = 0; j < m; ++j) {
    //   for (int64_t l = 0; l < n_attr; ++l) {
    //     // 提取 x 的 payload
    //     auto px = hal::slice(ctx, reshaped_x, {0, 0, j, l},
    //                          {batch_size, k_x, j + 1, l + 1}, {});
    //     px = hal::reshape(ctx, px, {batch_size, k_x});

    //     // 提取 y 的 payload
    //     auto py = hal::slice(ctx, reshaped_y, {0, 0, j, l},
    //                          {batch_size, k_y, j + 1, l + 1}, {});
    //     py = hal::reshape(ctx, py, {batch_size, k_y});

    //     // 拼接成完整的 payload: [batch_size, k_x + k_y]
    //     auto p_concat = hal::concatenate(ctx, {px, py}, 1);
    //     merge_inputs.push_back(p_concat);
    //   }
    // }

    // // ****************  使用支持多维payload的odd_evev_merge时：
    // 构造单一的巨型 Payload: [batch_size, k_x + k_y, m, n_attr]
    auto payloads = hal::concatenate(ctx, {reshaped_x, reshaped_y}, 1);
    std::vector<spu::Value> merge_inputs = {keys, payloads};

    // 调用 merge1d (内部会执行 odd_even_merge)
    // 注意：如果 SPU 的 merge1d 严格校验 1D，对于 batch_size > 1 的情况可能需要
    // vmap
    auto merged_results =
        hal::merge1d(ctx, merge_inputs, true, k_x, SortDirection::Ascending,
                     Visibility::VIS_SECRET, false);

    // // ****************  使用不支持多维payload的odd_evev_merge时：
    // // 重组 merged_blocks
    // std::vector<spu::Value> merged_payloads;
    // merged_payloads.reserve(m * n_attr);
    // for (size_t i = 1; i < merged_results.size(); ++i) {
    //   // 将每个 payload 扩展一个维度以便拼接: [batch_size, k_x + k_y, 1]
    //   auto reshaped_p =
    //       hal::reshape(ctx, merged_results[i], {batch_size, k_x + k_y, 1});
    //   merged_payloads.push_back(reshaped_p);
    // }

    // // 拼接所有 payload: [batch_size, k_x + k_y, m * n_attr]
    // auto flat_payloads = hal::concatenate(ctx, merged_payloads, 2);

    // // 恢复成块的形状: [batch_size, k_x + k_y, m, n_attr]
    // auto merged_blocks =
    //     hal::reshape(ctx, flat_payloads, {batch_size, k_x + k_y, m, n_attr});

    // 打印2.be.结果
    auto revealed5 =
        hal::dump_public_as<float>(ctx, hal::reveal(ctx, merged_results[1]));
    if (ctx->lctx()->Rank() == 0) {
      std::cout << "merged_results: " << revealed5 << std::endl;
    }

    // TODO：对每个块并行调用LogstarRecursive。解法1：直接串行，解法2：把块拼起来做向量计算。
  }

  return x;
}

spu::Value logstar(SPUContext* ctx, const spu::Value& key_x,
                   const spu::Value& key_y) {
  const int64_t nx = key_x.shape()[0];
  const int64_t ny = key_y.shape()[0];
  auto dtayp = key_x.dtype();
  auto valid_x = hal::seal(ctx, hal::constant(ctx, 1.0, dtayp, {1, nx, 1}));
  auto valid_y = hal::seal(ctx, hal::constant(ctx, 1.0, dtayp, {1, ny, 1}));
  // xt::xarray<int64_t> x_iota = xt::arange<int64_t>(nx);
  // auto idx_x = hal::seal(ctx, hal::constant(ctx, x_iota, dtayp, {nx, 1}));
  // xt::xarray<int64_t> y_iota = xt::arange<int64_t>(nx, nx + ny);
  // auto idx_y = hal::seal(ctx, hal::constant(ctx, y_iota, dtayp, {ny, 1}));
  auto list_id_x = hal::seal(ctx, hal::constant(ctx, 0, dtayp, {1, nx, 1}));
  auto list_id_y = hal::seal(ctx, hal::constant(ctx, 1, dtayp, {1, nx, 1}));

  auto x = hal::concatenate(
      ctx, {reshape(ctx, key_x, {1, nx, 1}), valid_x, list_id_x}, 2);
  auto y = hal::concatenate(
      ctx, {reshape(ctx, key_y, {1, ny, 1}), valid_y, list_id_y}, 2);

  if (ctx->lctx()->Rank() == 0) {
    std::cout << "x.shape(): " << x.shape() << std::endl;
  }

  return LogstarRecursive(ctx, x, y);

  // hal::dump_public_as<float>(ctx, hal::reveal(ctx, list_id_x));
  // auto c_idx_x = hal::dump_public_as<float>(ctx, hal::reveal(ctx, idx_y));
  // if (ctx->lctx()->Rank() == 0) {
  //   std::cout << "c_idx_x: " << c_idx_x << std::endl;
  // }
}

}  // namespace spu::kernel::hal