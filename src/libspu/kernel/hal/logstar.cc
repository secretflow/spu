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
 * @return Output values.
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

}  // namespace spu::kernel::hal