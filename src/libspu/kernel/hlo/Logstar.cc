#include "libspu/kernel/hlo/Logstar.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "libspu/core/bit_utils.h"
#include "libspu/core/context.h"
#include "libspu/core/trace.h"
#include "libspu/core/vectorize.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/random.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hal/utils.h"
#include "libspu/kernel/hlo/shuffle.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/spu.h"
#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hlo/permute.h"

namespace spu::kernel::hlo {

// ==========================================
// 1. 定义向量化的 NoteFunc
// ==========================================
// 输入的 p1, p2 维度为 [Batch, BlockSize]
// 输入的 g1, g2 维度为 [Batch, 1]
static std::pair<Value, Value> VectorizedNoteFunc(SPUContext* ctx,
                                                  const Value& p1,
                                                  const Value& p2,
                                                  const Value& g1,
                                                  const Value& g2) {
  // 1. g3 = g1 * g2 (Element-wise mul)
  auto g3 = hal::mul(ctx, g1, g2);

  // 2. diff = p2 - p1
  auto diff = hal::sub(ctx, p2, p1);

  // 3. 处理广播: g1 是 [Batch, 1], diff 是 [Batch, BlockSize]
  // SPU 的 mul 通常支持自动广播，但显式广播更安全
  Value g1_broadcasted = g1;
  if (diff.shape().size() > 0 && g1.shape() != diff.shape()) {
    g1_broadcasted = hal::broadcast_to(ctx, g1, diff.shape());
  }

  // 4. term = diff * g1
  auto term = hal::mul(ctx, diff, g1_broadcasted);

  // 5. p3 = p1 + term
  auto p3 = hal::add(ctx, p1, term);

  return {p3, g3};
}

// 重载版本：不需要输入的 g2 (用于 Down-Sweep 的最后阶段)
static Value VectorizedNoteFunc(SPUContext* ctx, const Value& p1,
                                const Value& p2, const Value& g1) {
  auto diff = hal::sub(ctx, p2, p1);

  Value g1_broadcasted = g1;
  if (diff.shape().size() > 0 && g1.shape() != diff.shape()) {
    g1_broadcasted = hal::broadcast_to(ctx, g1, diff.shape());
  }

  auto term = hal::mul(ctx, diff, g1_broadcasted);
  auto p3 = hal::add(ctx, p1, term);
  return p3;
}

// ====================================================================================
//                Vectorized AggregateBrentKung without valid bits
// ====================================================================================
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

// ====================================================================================
//                Vectorized AggregateBrentKung with valid bits
// ====================================================================================
std::pair<Value, Value> AggregateBrentKung(SPUContext* ctx, 
  const Value& x_full,
  const Value& valid_full, 
  const Value& g_full) {

  // 1. 获取维度信息
  const int64_t n = x_full.shape()[0];
  const int64_t block_size = x_full.shape()[1]; // B

  // 检查 valid 维度是否匹配 (Debug模式下很有用，Release可省略)
  if (valid_full.shape()[0] != n || valid_full.shape()[1] != block_size) {
  // 实际代码中建议加 SPUENFORCE 或类似检查
  }

  // 2. 数据拼接 (Pre-process)
  // 将 x (N, B) 和 valid (N, B) 在列维度拼接 -> (N, 2*B)
  Value payload_full = hal::concatenate(ctx, {x_full, valid_full}, 1);

  // 更新后续逻辑使用的 block_size
  const int64_t total_block_size = block_size * 2; 
  const int64_t logn = std::floor(std::log2(n));

  // --- 下面的逻辑与原版完全一致，只是操作的是 payload_full ---

  std::vector<Value> p_rows(n); 
  std::vector<Value> g_rows(n);

  for (int i = 0; i < n; ++i) {
  // Slice 出一行: [1, 2B]
  p_rows[i] = hal::slice(ctx, payload_full, {i, 0}, {i + 1, total_block_size}, {});
  g_rows[i] = hal::slice(ctx, g_full, {i, 0}, {i + 1, 1}, {});

  // 保持 Rank=2 以便拼接
  p_rows[i] = hal::reshape(ctx, p_rows[i], {1, total_block_size});
  g_rows[i] = hal::reshape(ctx, g_rows[i], {1, 1});
  }

  std::vector<std::vector<Value>> p_tree(n, std::vector<Value>(logn));
  std::vector<std::vector<Value>> g_tree(n, std::vector<Value>(logn));
  std::vector<Value> res(n);

  // --------------------------------------------------------
  // Up-Sweep
  // --------------------------------------------------------
  {
  std::vector<Value> p1_batch, p2_batch, g1_batch, g2_batch;
  std::vector<int> target_indices;

  for (int i = 1; i < n; i += 2) {
  p1_batch.push_back(p_rows[i]);
  p2_batch.push_back(p_rows[i - 1]);
  g1_batch.push_back(g_rows[i]);
  g2_batch.push_back(g_rows[i - 1]);
  target_indices.push_back(i);
  }

  if (!target_indices.empty()) {
  auto p1_vec = hal::concatenate(ctx, p1_batch, 0);
  auto p2_vec = hal::concatenate(ctx, p2_batch, 0);
  auto g1_vec = hal::concatenate(ctx, g1_batch, 0);
  auto g2_vec = hal::concatenate(ctx, g2_batch, 0);

  // 这里的计算会自动带上 valid 部分
  auto [p3_vec, g3_vec] = VectorizedNoteFunc(ctx, p1_vec, p2_vec, g1_vec, g2_vec);

  for (size_t k = 0; k < target_indices.size(); ++k) {
  int idx = target_indices[k];
  p_tree[idx][0] = hal::slice(ctx, p3_vec, {static_cast<int64_t>(k), 0},
  {static_cast<int64_t>(k + 1), total_block_size}, {});
  g_tree[idx][0] = hal::slice(ctx, g3_vec, {static_cast<int64_t>(k), 0},
  {static_cast<int64_t>(k + 1), 1}, {});
  }
  }
  }

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

  auto [p3_vec, g3_vec] = VectorizedNoteFunc(ctx, p1_vec, p2_vec, g1_vec, g2_vec);

  for (size_t k = 0; k < target_indices.size(); ++k) {
  int idx = target_indices[k];
  p_tree[idx][j] = hal::slice(ctx, p3_vec, {static_cast<int64_t>(k), 0},
  {static_cast<int64_t>(k + 1), total_block_size}, {});
  g_tree[idx][j] = hal::slice(ctx, g3_vec, {static_cast<int64_t>(k), 0},
  {static_cast<int64_t>(k + 1), 1}, {});
  }
  }
  }

  // --------------------------------------------------------
  // Down-Sweep
  // --------------------------------------------------------
  res[0] = p_rows[0];

  for (int j = 0; j < logn; ++j) {
  int idx = (1 << (j + 1)) - 1;
  if (idx < n) {
  res[idx] = p_tree[idx][j];
  }
  }

  for (int j = logn - 3; j >= 0; --j) {
  int step = 1 << (j + 2);
  int half_step = 1 << (j + 1);

  std::vector<Value> p1_batch, p2_batch, g1_batch;
  std::vector<int> target_indices;

  for (int k = 1; k < n / step + 1; ++k) {
  int idx_curr = n - half_step * (2 * k - 1) - 1;
  int idx_prev = n - half_step * 2 * k - 1;

  if (idx_curr >= 0 && idx_prev >= 0 && idx_curr < n) {
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

  auto p3_vec = VectorizedNoteFunc(ctx, p1_vec, p2_vec, g1_vec);

  for (size_t k = 0; k < target_indices.size(); ++k) {
  res[target_indices[k]] = hal::slice(ctx, p3_vec, {static_cast<int64_t>(k), 0},
    {static_cast<int64_t>(k + 1), total_block_size}, {});
  }
  }
  }

  {
  std::vector<Value> p1_batch, p2_batch, g1_batch;
  std::vector<int> target_indices;

  for (int k = 1; k < n / 2 + 1; ++k) {
  int idx_curr = n - 2 * k;
  int idx_prev = n - 2 * k - 1;

  if (idx_curr >= 0 && idx_prev >= 0 && idx_curr < n) {
  p1_batch.push_back(p_rows[idx_curr]);
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
  res[target_indices[k]] = hal::slice(ctx, p3_vec, {static_cast<int64_t>(k), 0},
    {static_cast<int64_t>(k + 1), total_block_size}, {});
  }
  }
  }

  // --- 3. 输出后处理 ---
  // final_payload Shape: [N, 2*B]
  auto final_payload = hal::concatenate(ctx, res, 0); 

  // 切分回 x 和 valid
  // x:     [0 .. B)
  // valid: [B .. 2B)
  auto y_out = hal::slice(ctx, final_payload, {0, 0}, {n, block_size}, {});
  auto valid_out = hal::slice(ctx, final_payload, {0, block_size}, {n, 2 * block_size}, {});

  return {y_out, valid_out};
}

// ====================================================================================
//                NonVectorized AggregateBrentKung without valid bits
// ====================================================================================
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







// ====================================================================================
//                    ExtraxtOrdered
// ====================================================================================
Value prefix_sum(SPUContext *ctx, const Value &x) {
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

void extract_ordered(SPUContext* ctx, const spu::Value& x, const spu::Value& valids) {
  // 1.计算valids前缀和rho
  auto rho = prefix_sum(ctx, valids);
  // 2.洗牌x、valids、rho
  std::vector<spu::Value> inputs_to_shuffle = {x, valids, rho};
  // std::vector<spu::Value> shuffled_results = hlo::Shuffle(ctx, inputs_to_shuffle, 1);
  auto [shuffled_results, pi] = hlo::shuffle_with_perm(ctx, inputs_to_shuffle, 1);
  auto sx   = shuffled_results[0];
  auto svalids   = shuffled_results[1];
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
  auto svalids_open = hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx, svalids));
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
          p_hat_indices[right--] = i; // 无效的放后面 (倒序填充)
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
  spu::Value p_hat = hal::constant(ctx, p_hat_xt, spu::DT_I64, 
    {static_cast<int64_t>(p_hat_indices.size())});
  std::vector<spu::Value> inputs_to_permute = {sx, srho, pi};
  std::vector<spu::Value> compacted_results = hlo::Permute(ctx, inputs_to_permute, p_hat, 1);

  auto y_compacted = compacted_results[0];
  auto rho_compacted = compacted_results[1];
  auto rho_compose_pi = compacted_results[2];

      auto y_compacted_open = hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx, y_compacted));
      auto rho_compacted_open = hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx, rho_compacted));
      auto rho_compose_pi_open = hal::dump_public_as<int64_t>(ctx, hal::reveal(ctx, rho_compose_pi));
      if (ctx->lctx()->Rank() == 0) {
        std::cout << "y_compacted: " << y_compacted_open << std::endl;
        std::cout << "rho_compacted: " << rho_compacted_open << std::endl;
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
  // spu::Value y_final = hal::slice(ctx, y_compacted, start_indices, end_indices, {});
  
  // // 返回结果
  // // 如果是 Extract Unordered，到这里就结束了，返回 y_final
  // // 如果是 Extract Ordered，你需要利用 rho_compacted 继续进行后续的 Unshuffle 操作
  // return std::vector<spu::Value>{y_final, rho_compacted};
}




}  // namespace spu::kernel::hlo