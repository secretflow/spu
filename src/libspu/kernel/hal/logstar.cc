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
#include "libspu/kernel/test_util.h"
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
 *      x_out[i] = x[i],                if g[i] == 0;
 *      x_out[i] = x_out[i-1],          if g[i] == 1.
 *
 * @param x Input data blocks [batch_size, K, m, n_attr].
 * @param g_in Group signals (transition flags) [batch_size, K].
 * @return spu::Value Duplicated inputs with the same shape as x.
 */
spu::Value duplicate_brent_kung(SPUContext* ctx, const spu::Value& x,
                                const spu::Value& g_in) {
  const int64_t batch_size = x.shape()[0];
  const int64_t K = x.shape()[1];
  const int64_t m = x.shape()[2];
  const int64_t n_attr = x.shape()[3];
  const int64_t block_size = m * n_attr;

  Value p = kernel::hal::reshape(ctx, x, {batch_size * K, block_size});
  p = mutable_copy(ctx, p);
  Value g = kernel::hal::reshape(ctx, g_in, {batch_size * K, 1});
  g = mutable_copy(ctx, g);

  auto p_flat_ref = p.data().reshape({p.numel()});
  auto g_flat_ref = g.data().reshape({g.numel()});

  int depth = Log2Ceil(K);
  size_t max_blocks = (K / 2) + 1;
  spu::Index idx_right_blocks;
  idx_right_blocks.reserve(max_blocks);
  spu::Index idx_left_blocks;
  idx_left_blocks.reserve(max_blocks);
  spu::Index idx_root_blocks;
  idx_root_blocks.reserve(max_blocks);
  spu::Index idx_child_blocks;
  idx_child_blocks.reserve(max_blocks);

  // Auxiliary function: Broadcast the index of a single sequence to all batches
  auto get_batched_indices = [&](const spu::Index& blocks) {
    spu::Index batched;
    batched.reserve(blocks.size() * batch_size);
    for (int64_t b = 0; b < batch_size; ++b) {
      int64_t offset = b * K;
      for (int64_t idx : blocks) {
        batched.push_back(offset + idx);
      }
    }
    return batched;
  };

  // --- 1. Up-Sweep (Reduce Phase) ---
  for (int j = 0; j < depth; ++j) {
    int64_t step = 1LL << (j + 1);
    int64_t left_child_off = 1LL << j;

    idx_right_blocks.clear();
    idx_left_blocks.clear();

    for (int64_t i = step - 1; i < K; i += step) {
      idx_right_blocks.push_back(i);
      idx_left_blocks.push_back(i - left_child_off);
    }

    if (idx_right_blocks.empty()) continue;

    spu::Index batched_right = get_batched_indices(idx_right_blocks);
    spu::Index batched_left = get_batched_indices(idx_left_blocks);

    spu::Index idx_right_elems = flat_indices(batched_right, block_size);
    spu::Index idx_left_elems = flat_indices(batched_left, block_size);

    Value v_right_p(p_flat_ref.linear_gather(idx_right_elems), p.dtype());
    Value v_left_p(p_flat_ref.linear_gather(idx_left_elems), p.dtype());
    Value v_right_g(g_flat_ref.linear_gather(batched_right), g.dtype());
    Value v_left_g(g_flat_ref.linear_gather(batched_left), g.dtype());

    int64_t num_nodes = batched_right.size();
    v_right_p = kernel::hal::reshape(ctx, v_right_p, {num_nodes, block_size});
    v_left_p = kernel::hal::reshape(ctx, v_left_p, {num_nodes, block_size});
    v_right_g = kernel::hal::reshape(ctx, v_right_g, {num_nodes, 1});
    v_left_g = kernel::hal::reshape(ctx, v_left_g, {num_nodes, 1});

    auto [new_p, new_g] =
        node_func(ctx, v_right_p, v_left_p, v_right_g, v_left_g);

    p_flat_ref.linear_scatter(new_p.data().reshape({new_p.numel()}),
                              idx_right_elems);
    g_flat_ref.linear_scatter(new_g.data().reshape({new_g.numel()}),
                              batched_right);
  }

  // --- 2. Down-Sweep (Distribute Phase) ---
  for (int j = depth - 2; j >= 0; --j) {
    int64_t step = 1LL << (j + 1);
    int64_t dist = 1LL << j;

    idx_root_blocks.clear();
    idx_child_blocks.clear();

    for (int64_t i = step - 1; i < K; i += step) {
      int64_t target = i + dist;
      if (target < K) {
        idx_root_blocks.push_back(i);
        idx_child_blocks.push_back(target);
      }
    }

    if (idx_child_blocks.empty()) continue;

    spu::Index batched_root = get_batched_indices(idx_root_blocks);
    spu::Index batched_child = get_batched_indices(idx_child_blocks);

    spu::Index idx_root_elems = flat_indices(batched_root, block_size);
    spu::Index idx_child_elems = flat_indices(batched_child, block_size);

    Value v_root_p(p_flat_ref.linear_gather(idx_root_elems), p.dtype());
    Value v_child_p(p_flat_ref.linear_gather(idx_child_elems), p.dtype());
    Value v_child_g(g_flat_ref.linear_gather(batched_child), g.dtype());

    int64_t num_nodes = batched_child.size();
    v_root_p = kernel::hal::reshape(ctx, v_root_p, {num_nodes, block_size});
    v_child_p = kernel::hal::reshape(ctx, v_child_p, {num_nodes, block_size});
    v_child_g = kernel::hal::reshape(ctx, v_child_g, {num_nodes, 1});

    auto new_p_child = node_func(ctx, v_child_p, v_root_p, v_child_g);

    p_flat_ref.linear_scatter(new_p_child.data().reshape({new_p_child.numel()}),
                              idx_child_elems);
  }
  return kernel::hal::reshape(ctx, p, {batch_size, K, m, n_attr});
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

spu::Value ComputeMedians(SPUContext* ctx, SortDirection direction,
                          const spu::Value& arr, const int k, const int m) {
  const int64_t batch_size = arr.shape()[0];
  const int64_t n_attr = arr.shape()[2];

  auto reshaped = hal::reshape(ctx, arr, {batch_size, k, m, n_attr});
  auto key = hal::slice(ctx, reshaped, {0, 0, 0, 0}, {batch_size, k, m, 1}, {});
  auto valid =
      hal::slice(ctx, reshaped, {0, 0, 0, 1}, {batch_size, k, m, 2}, {});

  auto v_prev_slice =
      hal::slice(ctx, valid, {0, 0, 0, 0}, {batch_size, k, m - 1, 1}, {});
  auto zeros = hal::constant(ctx, 0.0F, valid.dtype(), {batch_size, k, 1, 1});
  if (arr.isSecret()) zeros = hal::seal(ctx, zeros);
  auto valid_prev = hal::concatenate(ctx, {zeros, v_prev_slice}, 2);

  auto ones = hal::constant(ctx, 1.0F, valid.dtype(), valid.shape());
  if (arr.isSecret()) ones = hal::seal(ctx, ones);
  auto not_valid_prev = hal::sub(ctx, ones, valid_prev);

  auto f = hal::mul(ctx, valid, not_valid_prev);
  auto term = hal::mul(ctx, key, f);

  auto z = hal::slice(ctx, term, {0, 0, 0, 0}, {batch_size, k, 1, 1}, {});
  z = hal::reshape(ctx, z, {batch_size, k});

  auto sum_f = hal::slice(ctx, f, {0, 0, 0, 0}, {batch_size, k, 1, 1}, {});
  sum_f = hal::reshape(ctx, sum_f, {batch_size, k});

  for (int64_t j = 1; j < m; ++j) {
    auto slice_j =
        hal::slice(ctx, term, {0, 0, j, 0}, {batch_size, k, j + 1, 1}, {});
    slice_j = hal::reshape(ctx, slice_j, {batch_size, k});
    z = hal::add(ctx, z, slice_j);

    auto f_j = hal::slice(ctx, f, {0, 0, j, 0}, {batch_size, k, j + 1, 1}, {});
    f_j = hal::reshape(ctx, f_j, {batch_size, k});
    sum_f = hal::add(ctx, sum_f, f_j);
  }

  auto first_key =
      hal::slice(ctx, key, {0, 0, 0, 0}, {batch_size, k, 1, 1}, {});
  first_key = hal::reshape(ctx, first_key, {batch_size, k});

  // z = first_key + sum_f * (z - first_key)
  z = hal::add(ctx, first_key,
               hal::mul(ctx, sum_f, hal::sub(ctx, z, first_key)));
  return z;
}

static thread_local int call_count = 0;

spu::Value LogstarRecursive(SPUContext* ctx, SortDirection direction,
                            const spu::Value& x_in, const spu::Value& y_in) {
  const int64_t batch_size = x_in.shape()[0];
  const int64_t nx = x_in.shape()[1];
  const int64_t ny = y_in.shape()[1];
  const int64_t n_attr = x_in.shape()[2];
  ++call_count;

  // Reset opposite list_id for x and y
  auto dtayp = x_in.dtype();
  auto list_id_0 = hal::constant(ctx, 0.0F, dtayp, {batch_size, nx, 1});
  if (x_in.isSecret()) list_id_0 = hal::seal(ctx, list_id_0);
  auto x_attr02 = hal::slice(ctx, x_in, {0, 0, 0}, {batch_size, nx, 2}, {});
  spu::Value x;
  if (n_attr > 3) {
    auto x_attr_rest =
        hal::slice(ctx, x_in, {0, 0, 3}, {batch_size, nx, n_attr}, {});
    x = hal::concatenate(ctx, {x_attr02, list_id_0, x_attr_rest}, 2);
  } else {
    x = hal::concatenate(ctx, {x_attr02, list_id_0}, 2);
  }

  auto list_id_1 = hal::constant(ctx, 1.0F, dtayp, {batch_size, ny, 1});
  if (y_in.isSecret()) list_id_1 = hal::seal(ctx, list_id_1);
  auto y_attr02 = hal::slice(ctx, y_in, {0, 0, 0}, {batch_size, ny, 2}, {});
  spu::Value y;
  if (n_attr > 3) {
    auto y_attr_rest =
        hal::slice(ctx, y_in, {0, 0, 3}, {batch_size, ny, n_attr}, {});
    y = hal::concatenate(ctx, {y_attr02, list_id_1, y_attr_rest}, 2);
  } else {
    y = hal::concatenate(ctx, {y_attr02, list_id_1}, 2);
  }

  // block parameters
  int64_t basic_size = 5;
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

    auto orig_list_id = hal::slice(ctx, x, {0, 0, 2}, {batch_size, 1, 3}, {});
    spu::Value padding_list_id =
        hal::broadcast_to(ctx, orig_list_id, {batch_size, padding_len, 1});

    auto padding =
        hal::concatenate(ctx, {padding_key, padding_valid, padding_list_id}, 2);
    x_pad = hal::concatenate(ctx, {x, padding}, 1);
  }
  if (ny % m != 0) {
    int64_t padding_len = k_y * m - ny;
    auto max_key = hal::slice(ctx, y, {0, ny - 1, 0}, {batch_size, ny, 1}, {});
    spu::Value padding_key =
        hal::broadcast_to(ctx, max_key, {batch_size, padding_len, 1});

    spu::Value padding_valid = seal(
        ctx, hal::constant(ctx, 0, x.dtype(), {batch_size, padding_len, 1}));

    auto orig_list_id = hal::slice(ctx, y, {0, 0, 2}, {batch_size, 1, 3}, {});
    spu::Value padding_list_id =
        hal::broadcast_to(ctx, orig_list_id, {batch_size, padding_len, 1});

    auto padding =
        hal::concatenate(ctx, {padding_key, padding_valid, padding_list_id}, 2);
    y_pad = hal::concatenate(ctx, {y, padding}, 1);
  }

  if (nx <= basic_size) {
    auto key_x = hal::reshape(
        ctx, hal::slice(ctx, x, {0, 0, 0}, {batch_size, nx, 1}, {}),
        {batch_size, nx});
    auto key_y = hal::reshape(
        ctx, hal::slice(ctx, y, {0, 0, 0}, {batch_size, ny, 1}, {}),
        {batch_size, ny});
    auto keys = hal::concatenate(ctx, {key_x, key_y}, 1);
    auto payloads =
        hal::concatenate(ctx, {x, y}, 1);  // [batch_size, nx + ny, n_attr]

    std::vector<spu::Value> merge_inputs = {keys, payloads};

    auto merged_base = hal::merge1d(ctx, merge_inputs, true, nx, direction,
                                    Visibility::VIS_SECRET, false);
    return merged_base[1];
  } else {
    // 2.a.
    auto median_x = ComputeMedians(ctx, direction, x_pad, k_x, m);
    auto median_y = ComputeMedians(ctx, direction, y_pad, k_y, m);

    // 2.cd.
    auto reshaped_x = hal::reshape(ctx, x_pad, {batch_size, k_x, m, n_attr});
    auto reshaped_y = hal::reshape(ctx, y_pad, {batch_size, k_y, m, n_attr});

    // 2.be. Merge by medians
    auto keys = hal::concatenate(ctx, {median_x, median_y}, 1);
    const int64_t K = k_x + k_y;

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    // CompositeKey = Keys * K + iota（只支持整数）
    xt::xarray<float> iota_arr = xt::arange<float>(K);
    auto iota = hal::constant(ctx, iota_arr, keys.dtype(), {batch_size, K});
    if (keys.isSecret()) iota = hal::seal(ctx, iota);

    auto K_val = hal::constant(ctx, static_cast<float>(K), keys.dtype(),
                               {batch_size, K});
    if (keys.isSecret()) K_val = hal::seal(ctx, K_val);

    auto scaled_keys = hal::mul(ctx, keys, K_val);
    auto composite_keys = hal::add(ctx, scaled_keys, iota);

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    // // CompositeKey = (Keys << iota_bits) + bitcast(iota)
    // //（MPC的概率截断会导致出错）
    // int64_t iota_bits = static_cast<int64_t>(std::ceil(std::log2(K)) + 1);
    // if (iota_bits == 0) iota_bits = 1;

    // spu::Sizes shift_amount = {iota_bits};
    // auto shifted_keys = hal::left_shift(ctx, keys, shift_amount);

    // xt::xarray<int64_t> iota_arr = xt::arange<int64_t>(K);
    // auto iota_val = hal::constant(ctx, iota_arr, spu::DT_I64, {batch_size,
    // K});

    // if (keys.isSecret()) {
    //   iota_val = hal::seal(ctx, iota_val);
    // }

    // auto iota_aligned = hal::bitcast(ctx, iota_val, keys.dtype());
    // auto composite_keys = hal::add(ctx, shifted_keys, iota_aligned);

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    // // CompositeKey = Keys * 2^iota_bits + iota / 2^f_bits
    // //（MPC的概率截断会导致出错）
    // int64_t iota_bits = static_cast<int64_t>(std::ceil(std::log2(K)));
    // if (iota_bits == 0) iota_bits = 1;

    // auto multiplier = static_cast<float>(1ULL << iota_bits);
    // auto multiplier_val =
    //     hal::constant(ctx, multiplier, keys.dtype(), {batch_size, K});
    // if (keys.isSecret()) multiplier_val = hal::seal(ctx, multiplier_val);
    // auto multiplied_keys = hal::mul(ctx, keys, multiplier_val);

    // auto revealed1 = hal::dump_public_as<float>(ctx, hal::reveal(ctx, keys));
    // auto revealed2 =
    //     hal::dump_public_as<float>(ctx, hal::reveal(ctx, multiplied_keys));
    // if (ctx->lctx()->Rank() == 0) {
    //   std::cout << "keys: " << revealed1 << std::endl;
    //   std::cout << "multiplied_keys: " << revealed2 << std::endl;
    // }

    // int64_t f_bits = ctx->config().fxp_fraction_bits;
    // float denominator = std::exp2(static_cast<float>(f_bits));  // 2^f_bits

    // xt::xarray<float> iota_arr = xt::arange<float>(K);
    // iota_arr = iota_arr / denominator;

    // auto iota_val = hal::constant(ctx, iota_arr, keys.dtype(), {batch_size,
    // K}); if (keys.isSecret()) iota_val = hal::seal(ctx, iota_val);

    // auto revealed5 =
    //     hal::dump_public_as<float>(ctx, hal::reveal(ctx, iota_val));
    // if (ctx->lctx()->Rank() == 0) {
    //   std::cout << "iota_val: " << revealed5 << std::endl;
    // }

    // auto composite_keys = hal::add(ctx, multiplied_keys, iota_val);
    // auto revealed4 =
    //     hal::dump_public_as<float>(ctx, hal::reveal(ctx, composite_keys));
    // if (ctx->lctx()->Rank() == 0) {
    //   std::cout << std::fixed << std::setprecision(18);
    //   std::cout << "composite_keys: " << revealed4 << std::endl;
    // }

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    auto payloads = hal::concatenate(ctx, {reshaped_x, reshaped_y}, 1);
    std::vector<spu::Value> merge_inputs = {composite_keys, keys, payloads};

    auto merged_results =
        hal::merge1d(ctx, merge_inputs, true, k_x, SortDirection::Ascending,
                     Visibility::VIS_SECRET, false);

    auto merged_blocks = merged_results[2];  // B
    auto merged_medians = merged_results[1];

    // auto revealed5 =
    //     hal::dump_public_as<float>(ctx, hal::reveal(ctx, merged_medians));
    // if (ctx->lctx()->Rank() == 0) {
    //   std::cout << "merged_medians: " << revealed5 << std::endl;
    // }
    // auto revealed6 =
    //     hal::dump_public_as<float>(ctx, hal::reveal(ctx, merged_blocks));
    // if (ctx->lctx()->Rank() == 0) {
    //   std::cout << "merged_blocks: " << revealed6 << std::endl;
    // }

    // merged_blocks shape: [batch_size, K, m, n_attr]
    auto list_ids =
        hal::slice(ctx, merged_blocks, {0, 0, 0, 2}, {batch_size, K, 1, 3}, {});
    list_ids = hal::reshape(ctx, list_ids, {batch_size, K});

    // Get B_i and B_{i-1}
    auto b_curr = hal::slice(ctx, list_ids, {0, 1}, {batch_size, K}, {});
    auto b_prev = hal::slice(ctx, list_ids, {0, 0}, {batch_size, K - 1}, {});

    // xor_val = B_i.ListId ⊕ B_{i-1}.ListId
    auto diff = hal::sub(ctx, b_curr, b_prev);
    auto xor_val = hal::mul(ctx, diff, diff);

    // c_i = ¬(xor_val) = 1 - xor_val
    auto ones = hal::seal(
        ctx, hal::constant(ctx, 1.0F, list_ids.dtype(), {batch_size, K - 1}));
    auto c_rest = hal::sub(ctx, ones, xor_val);

    auto c_0 = hal::seal(
        ctx, hal::constant(ctx, 0.0F, list_ids.dtype(), {batch_size, 1}));
    auto c = hal::concatenate(ctx, {c_0, c_rest}, 1);

    // auto revealed_c = hal::dump_public_as<float>(ctx, hal::reveal(ctx, c));
    // if (ctx->lctx()->Rank() == 0) {
    //   std::cout << "transition flag c: " << revealed_c << std::endl;
    // }

    // 2.g. Duplicate blocks using Brent-Kung network
    auto B_shifted = hal::slice(ctx, merged_blocks, {0, 0, 0, 0},
                                {batch_size, K - 1, m, n_attr}, {});

    // Construct a Dummy block and set its list_id to the opposite of B_0
    auto first_list_id =
        hal::slice(ctx, merged_blocks, {0, 0, 0, 2}, {batch_size, 1, 1, 3}, {});

    auto ones_id =
        hal::constant(ctx, 1.0F, merged_blocks.dtype(), {batch_size, 1, 1, 1});
    if (merged_blocks.isSecret()) ones_id = hal::seal(ctx, ones_id);
    auto opposite_id = hal::sub(ctx, ones_id, first_list_id);

    auto dummy_list_id =
        hal::broadcast_to(ctx, opposite_id, {batch_size, 1, m, 1});

    auto dummy_attr01 =
        hal::constant(ctx, 0.0F, merged_blocks.dtype(), {batch_size, 1, m, 2});
    if (merged_blocks.isSecret()) dummy_attr01 = hal::seal(ctx, dummy_attr01);

    spu::Value dummy_block;
    if (n_attr > 3) {
      auto dummy_attr_rest = hal::constant(ctx, 0.0F, merged_blocks.dtype(),
                                           {batch_size, 1, m, n_attr - 3});
      if (merged_blocks.isSecret())
        dummy_attr_rest = hal::seal(ctx, dummy_attr_rest);
      dummy_block = hal::concatenate(
          ctx, {dummy_attr01, dummy_list_id, dummy_attr_rest}, 3);
    } else {
      dummy_block = hal::concatenate(ctx, {dummy_attr01, dummy_list_id}, 3);
    }

    // B_input_to_tree = [dummy_block, B_0, B_1, ..., B_{K-2}]
    auto B_input_to_tree = hal::concatenate(ctx, {dummy_block, B_shifted}, 1);
    auto S = duplicate_brent_kung(ctx, B_input_to_tree, c);

    // auto revealed_S_ = hal::dump_public_as<float>(ctx, hal::reveal(ctx, S));
    // if (ctx->lctx()->Rank() == 0) {
    //   std::cout << "duplicated blocks S: " << revealed_S_ << std::endl;
    // }

    // 2.h. Update IsReal for B (merged_blocks) and S
    auto B_key = hal::reshape(
        ctx,
        hal::slice(ctx, merged_blocks, {0, 0, 0, 0}, {batch_size, K, m, 1}, {}),
        {batch_size, K, m});
    auto B_valid = hal::reshape(
        ctx,
        hal::slice(ctx, merged_blocks, {0, 0, 0, 1}, {batch_size, K, m, 2}, {}),
        {batch_size, K, m});

    auto S_key = hal::reshape(
        ctx, hal::slice(ctx, S, {0, 0, 0, 0}, {batch_size, K, m, 1}, {}),
        {batch_size, K, m});
    auto S_valid = hal::reshape(
        ctx, hal::slice(ctx, S, {0, 0, 0, 1}, {batch_size, K, m, 2}, {}),
        {batch_size, K, m});

    auto B_curr_first = hal::reshape(ctx, merged_medians, {batch_size, K, 1});
    auto B_first_shifted =
        hal::slice(ctx, B_curr_first, {0, 1, 0}, {batch_size, K, 1},
                   {});  // [batch_size, K-1, 1]

    spu::FieldType field = ctx->config().field;
    int64_t total_bits = spu::SizeOf(field) * 8;
    int64_t f_bits = ctx->config().fxp_fraction_bits;
    int64_t i_bits = total_bits - f_bits - 1;

    float extremum;
    if (direction == SortDirection::Ascending) {
      extremum = std::exp2(static_cast<float>(i_bits)) - 1;
    } else {
      extremum = -std::exp2(static_cast<float>(i_bits));
    }
    auto inf_pad = hal::constant(
        ctx, extremum, B_key.dtype(),
        {batch_size, 1, 1});  // Set the next of key the last block to extremum
    if (B_key.isSecret()) inf_pad = hal::seal(ctx, inf_pad);
    auto B_next_first = hal::concatenate(ctx, {B_first_shifted, inf_pad},
                                         1);  // [batch_size, K, 1]

    auto B_curr_first_bcast =
        hal::broadcast_to(ctx, B_curr_first, {batch_size, K, m});
    auto B_next_first_bcast =
        hal::broadcast_to(ctx, B_next_first, {batch_size, K, m});

    // i. B_{i,j}.IsReal := B_{i,j}.IsReal ∧ (B_{i,j} <= B_{i+1, 0})
    auto cond_B = hal::less_equal(ctx, B_key, B_next_first_bcast);
    auto new_B_valid = hal::mul(ctx, B_valid, cond_B);

    // ii. S_{i,j}.IsReal := S_{i,j}.IsReal ∧ (S_{i,j} > B_{i, 0}) ∧ (S_{i,j} <=
    // B_{i+1, 0})
    auto cond_S1 = hal::greater(ctx, S_key, B_curr_first_bcast);
    auto cond_S2 = hal::less_equal(ctx, S_key, B_next_first_bcast);
    auto cond_S = hal::mul(ctx, cond_S1, cond_S2);
    auto new_S_valid = hal::mul(ctx, S_valid, cond_S);

    auto B_attr0 =
        hal::slice(ctx, merged_blocks, {0, 0, 0, 0}, {batch_size, K, m, 1}, {});
    auto B_attr2 = hal::slice(ctx, merged_blocks, {0, 0, 0, 2},
                              {batch_size, K, m, n_attr}, {});
    auto new_B_valid_reshaped =
        hal::reshape(ctx, new_B_valid, {batch_size, K, m, 1});
    merged_blocks =
        hal::concatenate(ctx, {B_attr0, new_B_valid_reshaped, B_attr2}, 3);

    auto S_attr0 = hal::slice(ctx, S, {0, 0, 0, 0}, {batch_size, K, m, 1}, {});
    auto S_attr2 =
        hal::slice(ctx, S, {0, 0, 0, 2}, {batch_size, K, m, n_attr}, {});
    auto new_S_valid_reshaped =
        hal::reshape(ctx, new_S_valid, {batch_size, K, m, 1});
    S = hal::concatenate(ctx, {S_attr0, new_S_valid_reshaped, S_attr2}, 3);

    // auto revealed_B =
    //     hal::dump_public_as<float>(ctx, hal::reveal(ctx, merged_blocks));
    // auto revealed_S = hal::dump_public_as<float>(ctx, hal::reveal(ctx, S));
    // if (ctx->lctx()->Rank() == 0) {
    //   std::cout << "merged_blocks.shape: " << merged_blocks.shape()
    //             << std::endl;
    //   std::cout << "processed blocks B: " << revealed_B << std::endl;
    //   std::cout << "processed blocks S: " << revealed_S << std::endl;
    // }

    // 2.i. parallel-for i \in [2k]: [[I_i]] := LogstarRecursive([[S_i]],
    // [[B_i]])
    auto S_flat = hal::reshape(ctx, S, {batch_size * K, m, n_attr});
    auto B_flat = hal::reshape(ctx, merged_blocks, {batch_size * K, m, n_attr});
    auto I_flat = LogstarRecursive(ctx, direction, S_flat, B_flat);

    const int64_t L = I_flat.shape()[1];
    // I_flat: [batch_size * K, L, n_attr]
    auto I = hal::reshape(ctx, I_flat, {batch_size, K * L, n_attr});
    return I;
  }
}

spu::Value logstar(SPUContext* ctx, SortDirection direction,
                   const spu::Value& key_x, const spu::Value& key_y) {
  const int64_t nx = key_x.shape()[0];
  const int64_t ny = key_y.shape()[0];
  auto dtayp = key_x.dtype();
  auto valid_x = hal::seal(ctx, hal::constant(ctx, 1.0, dtayp, {1, nx, 1}));
  auto valid_y = hal::seal(ctx, hal::constant(ctx, 1.0, dtayp, {1, ny, 1}));
  auto list_id_x = hal::seal(ctx, hal::constant(ctx, 0, dtayp, {1, nx, 1}));
  auto list_id_y = hal::seal(ctx, hal::constant(ctx, 1.0, dtayp, {1, ny, 1}));

  auto x = hal::concatenate(
      ctx, {reshape(ctx, key_x, {1, nx, 1}), valid_x, list_id_x}, 2);
  auto y = hal::concatenate(
      ctx, {reshape(ctx, key_y, {1, ny, 1}), valid_y, list_id_y}, 2);

  auto I = LogstarRecursive(ctx, direction, x, y);

  if (ctx->lctx()->Rank() == 0) {
    std::cout << "Number of recursive calls: " << call_count - 1 << std::endl;
  }

  const int64_t B = I.shape()[0];          // batch_size (通常為 1)
  const int64_t total_len = I.shape()[1];  // 包含 Dummy 的總長度
  SPU_ENFORCE(
      B == 1,
      "batchsize of the outermost recursion must be 1, but it is now: {}", B);

  auto valids = hal::slice(ctx, I, {0, 0, 1}, {B, total_len, 2}, {});
  valids = hal::reshape(ctx, valids, {1, total_len});
  auto keys = hal::slice(ctx, I, {0, 0, 0}, {B, total_len, 1}, {});
  keys = hal::reshape(ctx, keys, {1, total_len});

  auto [y_rows, valid_count] = extract_ordered(ctx, keys, valids);

  int64_t target_len = nx + ny;
  auto final_keys_2d = hal::slice(ctx, y_rows[0], {0, 0}, {1, target_len}, {});
  auto final_keys = hal::reshape(ctx, final_keys_2d, {target_len});
  return final_keys;
}

}  // namespace spu::kernel::hal