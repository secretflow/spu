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

//  Vectorized NoteFunc
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
  // 3. 广播: g1 是 [Batch, 1], diff 是 [Batch, BlockSize]
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
// 辅助函数：计算下一个2的幂
int64_t NextPowerOfTwo(int64_t n) {
  if (n <= 0) return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n + 1;
}
// // 最优：非 padding 方案，额外开销低
std::pair<Value, Value> AggregateBrentKung(SPUContext* ctx, const Value& x_full,
                                           const Value& valid_full,
                                           const Value& g_full) {
  const int64_t n = x_full.shape()[0];
  const int64_t block_size = x_full.shape()[1];
  const int64_t total_block_size = block_size * 2;

  // 1. 预处理
  Value payload_full = hal::concatenate(ctx, {x_full, valid_full}, 1);

  std::vector<Value> p_curr(n);
  std::vector<Value> g_curr(n);

  for (int i = 0; i < n; ++i) {
    p_curr[i] =
        hal::slice(ctx, payload_full, {i, 0}, {i + 1, total_block_size}, {});
    g_curr[i] = hal::slice(ctx, g_full, {i, 0}, {i + 1, 1}, {});
    p_curr[i] = hal::reshape(ctx, p_curr[i], {1, total_block_size});
    g_curr[i] = hal::reshape(ctx, g_curr[i], {1, 1});
  }

  int depth = 0;
  if (n > 1) {
    depth = std::ceil(std::log2(n));
  }

  // ========================================================
  // 1. Up-Sweep
  // ========================================================
  for (int j = 0; j < depth; ++j) {
    int64_t step = 1LL << (j + 1);
    int64_t left_child_off = 1LL << j;

    std::vector<Value> left_p;
    std::vector<Value> right_p;
    std::vector<Value> left_g;
    std::vector<Value> right_g;
    std::vector<int> target_indices;

    size_t estimated_size = (n / step) + 1;
    left_p.reserve(estimated_size);
    right_p.reserve(estimated_size);
    left_g.reserve(estimated_size);
    right_g.reserve(estimated_size);
    target_indices.reserve(estimated_size);

    for (int64_t i = step - 1; i < n; i += step) {
      int64_t left_idx = i - left_child_off;

      right_p.push_back(p_curr[i]);        // Right
      left_p.push_back(p_curr[left_idx]);  // Left
      right_g.push_back(g_curr[i]);
      left_g.push_back(g_curr[left_idx]);
      target_indices.push_back(i);
    }

    if (!target_indices.empty()) {
      auto v_left_p = hal::concatenate(ctx, left_p, 0);
      auto v_right_p = hal::concatenate(ctx, right_p, 0);
      auto v_left_g = hal::concatenate(ctx, left_g, 0);
      auto v_right_g = hal::concatenate(ctx, right_g, 0);
      auto [new_p, new_g] =
          VectorizedNoteFunc(ctx, v_right_p, v_left_p, v_right_g, v_left_g);

      for (size_t k = 0; k < target_indices.size(); ++k) {
        int idx = target_indices[k];
        p_curr[idx] =
            hal::slice(ctx, new_p, {static_cast<int64_t>(k), 0},
                       {static_cast<int64_t>(k + 1), total_block_size}, {});
        g_curr[idx] = hal::slice(ctx, new_g, {static_cast<int64_t>(k), 0},
                                 {static_cast<int64_t>(k + 1), 1}, {});
      }
    }
  }

  // ========================================================
  // 2. Down-Sweep
  // ========================================================
  for (int j = depth - 2; j >= 0; --j) {
    int64_t step = 1LL << (j + 1);
    int64_t dist = 1LL << j;

    std::vector<Value> root_p;
    std::vector<Value> child_p;
    std::vector<Value> child_g;
    std::vector<int> target_indices;

    size_t estimated_size = (n / step) + 1;
    root_p.reserve(estimated_size);
    child_p.reserve(estimated_size);
    child_g.reserve(estimated_size);
    target_indices.reserve(estimated_size);

    for (int64_t i = step - 1; i < n; i += step) {
      int64_t target = i + dist;

      if (target < n) {
        root_p.push_back(p_curr[i]);        // Root (Left)
        child_p.push_back(p_curr[target]);  // Child (Right)
        child_g.push_back(g_curr[target]);  // Child_G
        target_indices.push_back(target);
      }
    }

    if (!target_indices.empty()) {
      auto v_root_p = hal::concatenate(ctx, root_p, 0);
      auto v_child_p = hal::concatenate(ctx, child_p, 0);
      auto v_child_g = hal::concatenate(ctx, child_g, 0);
      auto new_p = VectorizedNoteFunc(ctx, v_child_p, v_root_p, v_child_g);

      for (size_t k = 0; k < target_indices.size(); ++k) {
        int idx = target_indices[k];
        p_curr[idx] =
            hal::slice(ctx, new_p, {static_cast<int64_t>(k), 0},
                       {static_cast<int64_t>(k + 1), total_block_size}, {});
      }
    }
  }

  // 3. 输出
  auto final_payload = hal::concatenate(ctx, p_curr, 0);
  auto y_out = hal::slice(ctx, final_payload, {0, 0}, {n, block_size}, {});
  auto valid_out =
      hal::slice(ctx, final_payload, {0, block_size}, {n, 2 * block_size}, {});

  return {y_out, valid_out};
}

}  // namespace spu::kernel::hlo