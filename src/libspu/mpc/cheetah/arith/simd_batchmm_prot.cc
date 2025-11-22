// Copyright 2024 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/cheetah/arith/simd_batchmm_prot.h"

#include "seal/seal.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "seal/util/scalingvariant.h"
#include "absl/numeric/bits.h"
#include "yacl/utils/parallel.h"
#include "yacl/utils/platform_utils.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/rlwe/types.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"


namespace spu::mpc::cheetah {

SIMDBatchMMProt::SIMDBatchMMProt(uint64_t simd_lane, uint64_t prime_modulus)
    : simd_lane_(simd_lane), prime_modulus_(prime_modulus) {
  SPU_ENFORCE(prime_modulus_.is_prime(), "modulus {} is not a prime",
              prime_modulus);
  SPU_ENFORCE(absl::has_single_bit(simd_lane), "invalid simd lane {}",
              simd_lane_);
  SPU_ENFORCE(prime_modulus % (2 * simd_lane) == 1);

  row_size_ = simd_lane_ / 2;
  encode_tabl_ = std::make_unique<seal::util::NTTTables>(
      absl::bit_width(simd_lane) - 1, prime_modulus_);

  // populate matrix_reps_index_map_
  matrix_reps_index_map_ = std::make_unique<std::vector<size_t>>();
  PopulateMatrixRepsIndexMap(simd_lane_, *matrix_reps_index_map_);

}

// Same as seal::BatchEncoder::populate_matrix_reps_index_map
void SIMDBatchMMProt::PopulateMatrixRepsIndexMap(uint64_t simd_lane, std::vector<size_t>& matrix_reps_index_map) {
  matrix_reps_index_map.resize(simd_lane);
  size_t logn = seal::util::get_power_of_two(simd_lane);

  // Copy from the matrix to the value vectors
  size_t row_size = simd_lane >> 1;
  size_t m = simd_lane << 1;
  uint64_t gen = 3;
  uint64_t pos = 1;
  for (size_t i = 0; i < row_size; i++)
  {
      // Position in normal bit order
      uint64_t index1 = (pos - 1) >> 1;
      uint64_t index2 = (m - pos - 1) >> 1;
      // Set the bit-reversed locations
      matrix_reps_index_map[i] = seal::util::reverse_bits(index1, logn);
      matrix_reps_index_map[row_size | i] = seal::util::reverse_bits(index2, logn);
      // Next primitive root
      pos *= gen;
      pos &= (m - 1);
  }
}


SIMDBatchMMProt::~SIMDBatchMMProt() = default;

// SIMDMulProt::EncodeSingle + matrix position arrangement from seal::BatchEncoder
void SIMDBatchMMProt::EncodeSingle(absl::Span<const uint64_t> array, RLWEPt &out) const {
  SPU_ENFORCE_LE(array.size(), (size_t)simd_lane_);

  SPU_ENFORCE(
      std::all_of(array.cbegin(), array.cend(),
                  [&](uint64_t x) { return x < prime_modulus_.value(); }),
      "array value out-of-range to encode");

  out.parms_id() = seal::parms_id_zero;
  out.resize(simd_lane_);
  std::copy_n(array.data(), array.size(), out.data());
  std::fill_n(out.data() + array.size(), simd_lane_ - array.size(), 0);

  // First write the values to destination coefficients.
  // Read in top row, then bottom row.
  for (size_t i = 0; i < array.size(); i++) {
      *(out.data() + (*matrix_reps_index_map_)[i]) = array[i];
  }
  for (size_t i = array.size(); i < simd_lane_; i++) {
      *(out.data() + (*matrix_reps_index_map_)[i]) = 0;
  }

  seal::util::inverse_ntt_negacyclic_harvey(out.data(), *encode_tabl_);
}

// SIMDMulProt::DecodeSingle + matrix position arrangement from seal::BatchEncoder
void SIMDBatchMMProt::DecodeSingle(const RLWEPt &poly, absl::Span<uint64_t> array) const {
  SPU_ENFORCE_EQ(poly.coeff_count(), (size_t)simd_lane_);
  SPU_ENFORCE_LE(array.size(), poly.coeff_count());
  if (array.empty()) {
    return;
  }

  std::vector<uint64_t> temp_dest(simd_lane_, 0);
  if (array.size() == (size_t)simd_lane_) {
    // SIMD encode is doing intt
    // inplace ntt
    std::copy_n(poly.data(), simd_lane_, temp_dest.data());
    seal::util::ntt_negacyclic_harvey(temp_dest.data(), *encode_tabl_);

    // Read top row, then bottom row
    for (size_t i = 0; i < array.size(); i++) {
        array[i] = temp_dest[(*matrix_reps_index_map_)[i]];
    }
  } else {
    // only take the front part
    std::vector<uint64_t> tmp(simd_lane_);
    std::copy_n(poly.data(), simd_lane_, tmp.data());
    seal::util::ntt_negacyclic_harvey(tmp.data(), *encode_tabl_);
    std::copy_n(tmp.data(), simd_lane_, temp_dest.data());

    // Read top row, then bottom row
    for (size_t i = 0; i < array.size(); i++) {
        array[i] = temp_dest[(*matrix_reps_index_map_)[i]];
    }
  }
}


void SIMDBatchMMProt::EncodeBatch(absl::Span<const uint64_t> array,
                                  absl::Span<RLWEPt> batch_out, 
                                  seal::BatchEncoder& encoder) const {
  SPU_ENFORCE_EQ(batch_out.size(), CeilDiv(array.size(), (size_t)simd_lane_));
  yacl::parallel_for(0, batch_out.size(), [&](int64_t bgn, int64_t end) {
    for (int64_t i = bgn; i < end; ++i) {
      int64_t slice_bgn = i * simd_lane_;
      int64_t slice_n = std::min<int64_t>(simd_lane_, array.size() - slice_bgn);
      EncodeSingle(array.subspan(slice_bgn, slice_n), batch_out[i]);
    }
  });
}

void SIMDBatchMMProt::DecodeBatch(absl::Span<const RLWEPt> polys,
                                  absl::Span<uint64_t> array,
                                  seal::BatchEncoder& encoder) const {
  SPU_ENFORCE_EQ(polys.size(), CeilDiv(array.size(), (size_t)simd_lane_));
  yacl::parallel_for(0, polys.size(), [&](int64_t bgn, int64_t end) {
    for (int64_t i = bgn; i < end; ++i) {
      int64_t slice_bgn = i * simd_lane_;
      int64_t slice_n = std::min<int64_t>(simd_lane_, array.size() - slice_bgn);
      std::vector<uint64_t> plain_vec(simd_lane_, 0);
      // encoder.decode(polys[i], plain_vec);
      DecodeSingle(polys[i], absl::MakeSpan(plain_vec));
      std::copy_n(plain_vec.data(), slice_n, array.data() + slice_bgn);
    }
  });
}

// Same as SIMDMulProt::SymEncrypt
void SIMDBatchMMProt::SymEncrypt(absl::Span<const RLWEPt> polys,
                                 const RLWESecretKey &secret_key,
                                 const seal::SEALContext &context, bool save_seed,
                                 absl::Span<RLWECt> out) const {
  SPU_ENFORCE_EQ(polys.size(), out.size());

  yacl::parallel_for(0, polys.size(), [&](int64_t bgn, int64_t end) {
    for (int64_t i = bgn; i < end; ++i) {
      seal::util::encrypt_zero_symmetric(secret_key, context,
                                         context.first_parms_id(), false,
                                         save_seed, out[i]);
      seal::util::multiply_add_plain_with_scaling_variant(
          polys[i], *context.first_context_data(),
          seal::util::RNSIter{out[i].data(), out[i].poly_modulus_degree()});
    }
  });

  // seal::Encryptor encryptor(context, secret_key);
  // // seal::Evaluator evaluator(context);

  // yacl::parallel_for(0, polys.size(), [&](int64_t bgn, int64_t end) {
  //   for (int64_t i = bgn; i < end; ++i) {
  //     encryptor.encrypt_symmetric(polys[i], out[i]);
  //   }
  // });
}


Shape2D SIMDBatchMMProt::ComputeInShape(const Meta& meta) {
  // input (batch * dims[0]) * dims[1]
  size_t in_rows = meta.batch * meta.dims[0];
  size_t eff_row_size = absl::bit_ceil(in_rows);
  size_t block_size = 0;
  if (eff_row_size <= row_size_) {
    block_size = row_size_ / eff_row_size;
  } else {
    block_size = 1;
  }
  // eff_row_size, block_size are power of 2
  // eff_row_size can > simd_lane_ 
  Shape2D in_shape = {static_cast<int64_t>(eff_row_size), static_cast<int64_t>(block_size)};
  return in_shape;
}

size_t SIMDBatchMMProt::ComputeInputCtNum(const Meta& meta, Shape2D in_shape) const {
  size_t block_size = in_shape[1];
  size_t num_row_blocks = CeilDiv((size_t)meta.dims[1], block_size);
  size_t input_groups = CeilDiv((uint64_t)in_shape[0], simd_lane_);
  return input_groups * num_row_blocks;
}

size_t SIMDBatchMMProt::ComputeWeightPtNum(const Meta& meta, Shape2D in_shape) const {
  size_t block_size = in_shape[1];
  size_t num_row_blocks = CeilDiv((size_t)meta.dims[1], block_size);
  size_t num_col_blocks = CeilDiv((size_t)meta.dims[2], block_size);
  if ((uint64_t)in_shape[0] <= row_size_) {
    // input_groups == 1
    return CeilDiv(num_col_blocks, (size_t)2) * num_row_blocks * block_size;
  } else {
    size_t input_groups = CeilDiv((uint64_t)in_shape[0], simd_lane_);
    return input_groups * num_col_blocks * num_row_blocks;
  }
}

size_t SIMDBatchMMProt::ComputeOutputCtNum(const Meta& meta, Shape2D in_shape) const {
  size_t block_size = in_shape[1];
  size_t num_col_blocks = CeilDiv((size_t)meta.dims[2], block_size);
  size_t input_groups = CeilDiv((uint64_t)in_shape[0], simd_lane_);
  if ((uint64_t)in_shape[0] <= row_size_) {
    // input_groups == 1
    return CeilDiv(num_col_blocks, (size_t)2);
  } else {
    return input_groups * num_col_blocks;
  }

}

void SIMDBatchMMProt::PrepareWeightVector(const Meta& meta, Shape2D in_shape,
                                                absl::Span<const uint64_t> weight,
                                                absl::Span<uint64_t> weight_vec) const {
  // weight (batch * dims[1]) * dims[2]
  // in_shape = {eff_row_size, block_size};
  size_t block_size = in_shape[1];
  size_t num_row_blocks = CeilDiv((size_t)meta.dims[1], block_size);
  size_t num_col_blocks = CeilDiv((size_t)meta.dims[2], block_size);
  // input_groups > 1 if and only if eff_row_size > simd_lane_ <-> meta.dims[0] > simd_lane_
  size_t input_groups = CeilDiv((uint64_t)in_shape[0], simd_lane_);


  // Fill weight_vec with 0
  std::fill(weight_vec.begin(), weight_vec.end(), 0);

  // Very similar with BOLT BSGS weight preparation
  // Use weight from different batch to align with different input in {eff_row_size}

  SPU_ENFORCE_EQ(weight.size(), meta.batch * meta.dims[1] * meta.dims[2]);
  if ((uint64_t)in_shape[0] <= row_size_) { 
    // Compute baby_step
    size_t baby_step = absl::bit_ceil(
      static_cast<uint64_t>(std::sqrt(block_size * meta.dims[2] / (double)meta.dims[1])));
    baby_step = std::min(baby_step, block_size);
    

    // compute 2 blocks in each poly
    // input_groups == 1
    SPU_ENFORCE_EQ(input_groups, (size_t)1);
    SPU_ENFORCE_EQ(weight_vec.size(), num_row_blocks * CeilDiv(num_col_blocks, (size_t)2) * block_size * simd_lane_);

    size_t job_number = num_row_blocks * CeilDiv(num_col_blocks, (size_t)2);
    yacl::parallel_for(0, job_number, [&](int64_t bgn, int64_t end) {
      for (int64_t job_id = bgn; job_id < end; ++job_id) {
        size_t rb = job_id / CeilDiv(num_col_blocks, (size_t)2);
        size_t cb2 = job_id % CeilDiv(num_col_blocks, (size_t)2); // for each block pair of 2 cols
        size_t block_start = (rb * CeilDiv(num_col_blocks, (size_t)2) + cb2) * block_size * simd_lane_;

        for (size_t d = 0; d < block_size; ++d) { // for each block diag
          size_t diag_start = block_start + d * simd_lane_;
          // BSGS plaintext rotate right step
          size_t diag_step = (size_t)(d / baby_step) * baby_step;

          for (size_t k = 0; k < block_size; ++k) { // for each diag element
            // diag d: (r, c) where r - c = d mod block_size
            // r = rb * block_size + r_in_block
            size_t src_row_idx1 = rb * block_size + (k + d - diag_step + block_size) % block_size; 
            // c = cb2 * block_size + c_in_block
            size_t src_col_idx1 = (2*cb2) * block_size + (k - diag_step + block_size) % block_size;

            // check boundary
            if (src_row_idx1 < (size_t)meta.dims[1] && src_col_idx1 < (size_t)meta.dims[2]) {
              for (size_t r = 0; r < (size_t)in_shape[0]; ++r) {
                  size_t batch_idx = r / meta.dims[0];
                  if (batch_idx >= meta.batch) {
                      break;
                  }

                  uint64_t value = weight[batch_idx * meta.dims[1] * meta.dims[2] + src_row_idx1 * meta.dims[2] + src_col_idx1];
                  weight_vec[diag_start + k * in_shape[0] + r] = value;
              }
            }

            // second col of the block pair
            size_t src_row_idx2 = src_row_idx1;
            size_t src_col_idx2 = (2*cb2+1) * block_size + (k - diag_step + block_size) % block_size;
            if (src_row_idx2 < (size_t)meta.dims[1] && src_col_idx2 < (size_t)meta.dims[2]) {
              for (size_t r = 0; r < (size_t)in_shape[0]; ++r) {
                  size_t batch_idx = r / meta.dims[0];
                  if (batch_idx >= meta.batch) {
                      break;
                  }
                  
                  uint64_t value = weight[batch_idx * meta.dims[1] * meta.dims[2] + src_row_idx2 * meta.dims[2] + src_col_idx2];
                  weight_vec[diag_start + k * in_shape[0] + r + row_size_] = value;
              }
            }
            // else pad 0
          }
        }
      }
    });
  } else {
    // block_size == 1
    SPU_ENFORCE_EQ(block_size, (size_t)1);
    SPU_ENFORCE_EQ(weight_vec.size(), input_groups * num_row_blocks * num_col_blocks * simd_lane_);
    // this case does not need rotation

    for (size_t i = 0; i < input_groups; ++i) { // for each group of input

      for (size_t rb = 0; rb < num_row_blocks; ++rb) { 
        for (size_t cb = 0; cb < num_col_blocks; ++cb) { // for each block
          size_t block_start = ((i * num_row_blocks + rb) * num_col_blocks + cb) * simd_lane_;

          // for (size_t r = 0; r < (size_t)in_shape[0]; ++r) { // for each input col element
          for (size_t r = 0; r < simd_lane_; ++r) { // for each input col element
            size_t batch_idx = (i * simd_lane_ + r) / meta.dims[0];
            if (batch_idx >= meta.batch) {
                break;
            }
            size_t src_row_idx = batch_idx * meta.dims[1] + rb;
            size_t src_col_idx = cb;
            if (src_row_idx <  meta.batch * meta.dims[1] && src_col_idx < (size_t)meta.dims[2]) {
              weight_vec[block_start + r] =
                  weight[src_row_idx * meta.dims[2] + src_col_idx];
            }
            // else pad 0
          }
        }
      }
    }

  }
}

NdArrayRef SIMDBatchMMProt::PrepareWeightVector(const Meta& meta, Shape2D in_shape,
                                                const NdArrayRef &weight) const {
  const Type& eltype = weight.eltype();
  SPU_ENFORCE(eltype.isa<Ring2k>(), "must be ring_type, got={}", eltype);

  size_t block_size = in_shape[1];
  size_t num_row_blocks = CeilDiv((size_t)meta.dims[1], block_size);
  size_t num_col_blocks = CeilDiv((size_t)meta.dims[2], block_size);
  // input_groups > 1 if and only if eff_row_size > simd_lane_ <-> meta.dims[0] > simd_lane_
  size_t input_groups = CeilDiv((uint64_t)in_shape[0], simd_lane_);

  SPU_ENFORCE_EQ((size_t)weight.numel(), (size_t)meta.batch * meta.dims[1] * meta.dims[2]);

  const auto field = eltype.as<Ring2k>()->field();
  NdArrayRef weight_vec;

  DISPATCH_ALL_FIELDS(field, [&]() {
    using uT = std::make_unsigned<ring2k_t>::type;
    if ((uint64_t)in_shape[0] <= row_size_) { 
      size_t baby_step = absl::bit_ceil(
        static_cast<uint64_t>(std::sqrt(block_size * meta.dims[2] / (double)meta.dims[1])));
      baby_step = std::min(baby_step, block_size);
      
      SPU_ENFORCE_EQ(input_groups, (size_t)1);
      weight_vec = ring_zeros(field, {static_cast<int64_t>(num_row_blocks *
                                                CeilDiv(num_col_blocks, (size_t)2) * block_size * simd_lane_)});

      size_t job_number = num_row_blocks * CeilDiv(num_col_blocks, (size_t)2);
      yacl::parallel_for(0, job_number, [&](int64_t bgn, int64_t end) {
        for (int64_t job_id = bgn; job_id < end; ++job_id) {
          size_t rb = job_id / CeilDiv(num_col_blocks, (size_t)2);
          size_t cb2 = job_id % CeilDiv(num_col_blocks, (size_t)2); // for each block pair of 2 cols
          size_t block_start = (rb * CeilDiv(num_col_blocks, (size_t)2) + cb2) * block_size * simd_lane_;

          for (size_t d = 0; d < block_size; ++d) { // for each block diag
            size_t diag_start = block_start + d * simd_lane_;
            size_t diag_step = (size_t)(d / baby_step) * baby_step;

            for (size_t k = 0; k < block_size; ++k) { // for each diag element
              size_t src_row_idx1 = rb * block_size + (k + d - diag_step + block_size) % block_size; 
              size_t src_col_idx1 = (2*cb2) * block_size + (k - diag_step + block_size) % block_size;

              if (src_row_idx1 < (size_t)meta.dims[1] && src_col_idx1 < (size_t)meta.dims[2]) {
                // std::cout << "k: " << k << " src_row_idx1: " << src_row_idx1 << " src_col_idx1: " << src_col_idx1 << std::endl;
                for (size_t r = 0; r < (size_t)in_shape[0]; ++r) {
                  size_t batch_idx = r / meta.dims[0];
                  if (batch_idx >= meta.batch) {
                      break;
                  }
                  uint64_t value = weight.at<uT>(batch_idx * meta.dims[1] * meta.dims[2] + src_row_idx1 * meta.dims[2] + src_col_idx1);
                  weight_vec.at<uT>(diag_start + k * in_shape[0] + r) = value;
                }
              }

              size_t src_row_idx2 = src_row_idx1;
              size_t src_col_idx2 = (2*cb2+1) * block_size + (k - diag_step + block_size) % block_size;
              if (src_row_idx2 < (size_t)meta.dims[1] && src_col_idx2 < (size_t)meta.dims[2]) {
                for (size_t r = 0; r < (size_t)in_shape[0]; ++r) {
                  size_t batch_idx = r / meta.dims[0];
                  if (batch_idx >= meta.batch) {
                      break;
                  }
                  uint64_t value = weight.at<uT>(batch_idx * meta.dims[1] * meta.dims[2] + src_row_idx2 * meta.dims[2] + src_col_idx2);
                  weight_vec.at<uT>(diag_start + k * in_shape[0] + r + row_size_) = value;
                }
              }
            }
          }
        }
      });
    } else {
      SPU_ENFORCE_EQ(block_size, (size_t)1);
      weight_vec = ring_zeros(field, {static_cast<int64_t>(input_groups * num_row_blocks * num_col_blocks * simd_lane_)});

      for (size_t i = 0; i < input_groups; ++i) { // for each group of input

        for (size_t rb = 0; rb < num_row_blocks; ++rb) { 
          for (size_t cb = 0; cb < num_col_blocks; ++cb) { // for each block
            size_t block_start = ((i * num_row_blocks + rb) * num_col_blocks + cb) * simd_lane_;

            for (size_t r = 0; r < simd_lane_; ++r) { // for each input col element
              size_t batch_idx = (i * simd_lane_ + r) / meta.dims[0];
              if (batch_idx >= meta.batch) {
                  break;
              }
              size_t src_row_idx = batch_idx * meta.dims[1] + rb;
              size_t src_col_idx = cb;
              if (src_row_idx <  meta.batch * meta.dims[1] && src_col_idx < (size_t)meta.dims[2]) {
                uint64_t value = weight.at<uT>(src_row_idx * meta.dims[2] + src_col_idx);
                weight_vec.at<uT>(block_start + r) = value;
              }
            }
          }
        }
      }
    }
  });
  return weight_vec;
}

void SIMDBatchMMProt::PrepareInputVector(const Meta& meta, Shape2D in_shape,
                                         absl::Span<const uint64_t> input,
                                         absl::Span<uint64_t> input_vec) const {
  // input (batch * dims[0]) * dims[1]
  // in_shape = {eff_row_size, block_size};
  size_t block_size = in_shape[1];
  size_t number_blocks = CeilDiv((size_t)meta.dims[1], block_size);
  // input_groups > 1 if and only if eff_row_size > simd_lane_ <-> meta.dims[0] > simd_lane_
  size_t input_groups = CeilDiv((uint64_t)in_shape[0], simd_lane_);

  SPU_ENFORCE_EQ(input.size(), meta.batch * meta.dims[0] * meta.dims[1]);
  SPU_ENFORCE_EQ(input_vec.size(), input_groups * number_blocks * simd_lane_);

  // Fill input_vec with 0
  std::fill(input_vec.begin(), input_vec.end(), 0);

  for (size_t i = 0; i < input_groups; ++i) { // for each group of input
    for (size_t b = 0; b < number_blocks; ++b) { // for each block
      size_t block_start = (i * number_blocks + b) * simd_lane_;

      for (size_t c = 0; c < block_size; ++c) { // for each col of the block
        size_t col_start = c * in_shape[0];

        for (size_t r = 0; r < std::min((size_t)in_shape[0], simd_lane_); ++r) { // for each row of the block
          size_t src_row_idx = i * simd_lane_ + r;
          size_t src_col_idx = b * block_size + c;

          if (src_row_idx < meta.batch * meta.dims[0] && src_col_idx < (size_t)meta.dims[1]) {
            input_vec[block_start + col_start + r] =
                input[src_row_idx * meta.dims[1] + src_col_idx];
            
            // if eff_row_size <= row_size_, we copy input into the second half of simd poly
            if ((size_t)in_shape[0] <= row_size_) {
                input_vec[block_start + col_start + r + row_size_] =
                    input[src_row_idx * meta.dims[1] + src_col_idx];
            }
          }
          // else pad 0
        }
      }
    }
  }
}

NdArrayRef SIMDBatchMMProt::PrepareInputVector(const Meta& meta, Shape2D in_shape,
                                               const NdArrayRef &input) const {
  const Type& eltype = input.eltype();
  SPU_ENFORCE(eltype.isa<Ring2k>(), "must be ring_type, got={}", eltype);

  size_t block_size = in_shape[1];
  size_t number_blocks = CeilDiv((size_t)meta.dims[1], block_size);
  // input_groups > 1 if and only if eff_row_size > simd_lane_ <-> meta.dims[0] > simd_lane_
  size_t input_groups = CeilDiv((uint64_t)in_shape[0], simd_lane_);
  
  SPU_ENFORCE_EQ((size_t)input.numel(), (size_t)meta.batch * meta.dims[0] * meta.dims[1]);

  const auto field = eltype.as<Ring2k>()->field();
  NdArrayRef input_vec = ring_zeros(field, {static_cast<int64_t>(input_groups * number_blocks * simd_lane_)});

  DISPATCH_ALL_FIELDS(field, [&]() {
    using uT = std::make_unsigned<ring2k_t>::type;

    yacl::parallel_for(0, input_groups, [&](uint64_t bgn, uint64_t end) {
      for (uint64_t i = bgn; i < end; ++i) { // for each group of input
        for (size_t b = 0; b < number_blocks; ++b) { // for each block
          size_t block_start = (i * number_blocks + b) * simd_lane_;

          for (size_t c = 0; c < block_size; ++c) { // for each col of the block
            size_t col_start = c * in_shape[0];

            for (size_t r = 0; r < std::min((size_t)in_shape[0], simd_lane_); ++r) { // for each row of the block
              size_t src_row_idx = i * simd_lane_ + r;
              size_t src_col_idx = b * block_size + c;

              if (src_row_idx < meta.batch * meta.dims[0] && src_col_idx < (size_t)meta.dims[1]) {
                input_vec.at<uT>(block_start + col_start + r) =
                    input.at<uT>(src_row_idx * meta.dims[1] + src_col_idx);
                if ((size_t)in_shape[0] <= row_size_) {
                    input_vec.at<uT>(block_start + col_start + r + row_size_) =
                        input.at<uT>(src_row_idx * meta.dims[1] + src_col_idx);
                }
              }
              // else pad 0
            }
          }
        }
      }
    });
  });

  return input_vec;
}

void SIMDBatchMMProt::ParseResult(const Meta& meta, Shape2D in_shape,
                                  absl::Span<const uint64_t> ans_poly,
                                  absl::Span<uint64_t> res_mat) const {
  // res_mat (batch * dims[0]) * dims[2]
  // in_shape = {eff_row_size, block_size};
  size_t block_size = in_shape[1];
  size_t num_col_blocks = CeilDiv((size_t)meta.dims[2], block_size);
  size_t input_groups = CeilDiv((uint64_t)in_shape[0], simd_lane_);

  SPU_ENFORCE_EQ(res_mat.size(), static_cast<uint64_t>(meta.batch * meta.dims[0] * meta.dims[2]));
  if ((size_t)in_shape[0] <= row_size_) { 
    // input_groups == 1
    SPU_ENFORCE_EQ(input_groups, (size_t)1);
    SPU_ENFORCE_EQ(ans_poly.size(), CeilDiv(num_col_blocks, (size_t)2) * simd_lane_);

    for (size_t cb2 = 0; cb2 < CeilDiv(num_col_blocks, (size_t)2); ++cb2) { // for each block pair of 2 cols
      size_t block_start = cb2 * simd_lane_;

      for (size_t c = 0; c < block_size; ++c) { // for each col of the block
        size_t col_start = c * in_shape[0];

        for (size_t r = 0; r < (size_t)in_shape[0]; ++r) { // for each row of the block
          size_t res_row_idx = r;
          size_t res_col_idx1 = (2*cb2) * block_size + c;
          size_t res_col_idx2 = (2*cb2+1) * block_size + c;

          if (res_row_idx < (size_t)meta.batch * meta.dims[0] && res_col_idx1 < (size_t)meta.dims[2]) {
            res_mat[res_row_idx * meta.dims[2] + res_col_idx1] =
                ans_poly[block_start + col_start + r];
          }
          // else skip

          if (res_row_idx < (size_t)meta.batch * meta.dims[0] && res_col_idx2 < (size_t)meta.dims[2]) {
            res_mat[res_row_idx * meta.dims[2] + res_col_idx2] =
                ans_poly[block_start + col_start + r + row_size_];
          }
          // else skip
        }
      }
      
    }
  } else {
    // input_groups >= 1
    SPU_ENFORCE_EQ(ans_poly.size(), input_groups * num_col_blocks * simd_lane_);
    // block_size == 1
    SPU_ENFORCE_EQ(block_size, (size_t)1);
    
    for (size_t i = 0; i < input_groups; ++i) { // for each group of input
      for (size_t cb = 0; cb < num_col_blocks; ++cb) { // for each block
        size_t block_start = (i * num_col_blocks + cb) * simd_lane_;

        // for (size_t r = 0; r < (size_t)in_shape[0]; ++r) { // for each row of the block
        for (size_t r = 0; r < simd_lane_; ++r) { // for each row of the block
          size_t res_row_idx = i * simd_lane_ + r;
          size_t res_col_idx = cb;

          if (res_row_idx < (size_t)meta.batch * meta.dims[0] && res_col_idx < (size_t)meta.dims[2]) {
            res_mat[res_row_idx * meta.dims[2] + res_col_idx] =
                ans_poly[block_start + r];
          }
          // else skip
        }
      }
    }
  }

}

NdArrayRef SIMDBatchMMProt::ParseResult(const Meta& meta, Shape2D in_shape,
                                       const NdArrayRef &ans_poly) const {
  const Type& eltype = ans_poly.eltype();
  SPU_ENFORCE(eltype.isa<Ring2k>(), "must be ring_type, got={}", eltype);

  size_t block_size = in_shape[1];
  size_t num_col_blocks = CeilDiv((size_t)meta.dims[2], block_size);
  size_t input_groups = CeilDiv((uint64_t)in_shape[0], simd_lane_);

  NdArrayRef res_mat = ring_zeros(eltype.as<Ring2k>()->field(),
                                  {static_cast<int64_t>(meta.batch * meta.dims[0] * meta.dims[2])});

  DISPATCH_ALL_FIELDS(eltype.as<Ring2k>()->field(), [&]() {
    using uT = std::make_unsigned<ring2k_t>::type;

    if ((size_t)in_shape[0] <= row_size_) { 
      SPU_ENFORCE_EQ(input_groups, (size_t)1);
      SPU_ENFORCE_EQ((size_t)ans_poly.numel(), CeilDiv(num_col_blocks, (size_t)2) * simd_lane_);

      for (size_t cb2 = 0; cb2 < CeilDiv(num_col_blocks, (size_t)2); ++cb2) { // for each block pair of 2 cols
        size_t block_start = cb2 * simd_lane_;

        for (size_t c = 0; c < block_size; ++c) { // for each col of the block
          size_t col_start = c * in_shape[0];

          for (size_t r = 0; r < (size_t)in_shape[0]; ++r) { // for each row of the block
            size_t res_row_idx = r;
            size_t res_col_idx1 = (2*cb2) * block_size + c;
            size_t res_col_idx2 = (2*cb2+1) * block_size + c;

            if (res_row_idx < (size_t)meta.batch * meta.dims[0] && res_col_idx1 < (size_t)meta.dims[2]) {
              res_mat.at<uT>(res_row_idx * meta.dims[2] + res_col_idx1) =
                  ans_poly.at<uT>(block_start + col_start + r);
            }
            if (res_row_idx < (size_t)meta.batch * meta.dims[0] && res_col_idx2 < (size_t)meta.dims[2]) {
              res_mat.at<uT>(res_row_idx * meta.dims[2] + res_col_idx2) =
                  ans_poly.at<uT>(block_start + col_start + r + row_size_);
            }
            // else skip
          }
        }
      }
    } else {
      SPU_ENFORCE_EQ((size_t)ans_poly.numel(), input_groups * num_col_blocks * simd_lane_);
      SPU_ENFORCE_EQ(block_size, (size_t)1);
      for (size_t i = 0; i < input_groups; ++i) { // for each group of input
        for (size_t cb = 0; cb < num_col_blocks; ++cb) { // for each block
          size_t block_start = (i * num_col_blocks + cb) * simd_lane_;

          for (size_t r = 0; r < simd_lane_; ++r) { // for each row of the block
            size_t res_row_idx = i * simd_lane_ + r;
            size_t res_col_idx = cb;

            if (res_row_idx < (size_t)meta.batch * meta.dims[0] && res_col_idx < (size_t)meta.dims[2]) {
              res_mat.at<uT>(res_row_idx * meta.dims[2] + res_col_idx) =
                  ans_poly.at<uT>(block_start + r);
            }
            // else skip
          }
        }
      }
    }
  });
  return res_mat;
}


void SIMDBatchMMProt::BatchMatMatMul(const Meta& meta, Shape2D in_shape,
                   absl::Span<const RLWECt> lhs_input, absl::Span<const RLWEPt> rhs_weight,
                   const RLWEPublicKey& public_key, const GaloisKeys& gal_keys,
                   const seal::SEALContext& context, 
                   absl::Span<RLWECt> out) const {
  seal::Evaluator evaluator(context);

  size_t block_size = in_shape[1];
  size_t num_row_blocks = CeilDiv((size_t)meta.dims[1], block_size);
  size_t num_col_blocks = CeilDiv((size_t)meta.dims[2], block_size);

  // input_groups > 1 if and only if eff_row_size > simd_lane_ <-> meta.dims[0] > simd_lane_
  size_t input_groups = CeilDiv((uint64_t)in_shape[0], simd_lane_);

  SPU_ENFORCE_EQ(lhs_input.size(), input_groups * num_row_blocks);
  if ((size_t)in_shape[0] <= row_size_) { 
    // input_groups == 1
    SPU_ENFORCE_EQ(input_groups, (size_t)1);
    SPU_ENFORCE_EQ(rhs_weight.size(), num_row_blocks * CeilDiv(num_col_blocks, (size_t)2) * block_size);
    SPU_ENFORCE_EQ(out.size(), CeilDiv(num_col_blocks, (size_t)2));
  

    // 1. Compute baby-step block size
    uint64_t baby_step = absl::bit_ceil(
        static_cast<uint64_t>(std::sqrt(block_size * meta.dims[2] / (double)meta.dims[1])));
    baby_step = std::min(baby_step, block_size);
    // std::cout << "baby_step: " << baby_step << std::endl;

    // 2. Pre-compute all the necessary rotations of input
    //    Rotate ct to get all baby steps
    std::vector<RLWECt> rotated_input(num_row_blocks * baby_step);


    yacl::parallel_for(0, num_row_blocks*baby_step, [&](size_t bgn, size_t end) {
      for (size_t idx = bgn; idx < end; ++idx) {
        size_t rb = idx / baby_step;
        size_t s = idx % baby_step;
        if (s == 0) {
          rotated_input[rb * baby_step + 0] = lhs_input[rb];
        } else {
          CATCH_SEAL_ERROR(evaluator.rotate_rows(lhs_input[rb], s*in_shape[0], gal_keys, 
                                                 rotated_input[rb * baby_step + s]));
        }
      }
    });

    // 3. For each col block pair, compute the BSGS result
    std::vector<RLWECt> tmp_out(CeilDiv(num_col_blocks, (size_t)2) * (block_size / baby_step));

    yacl::parallel_for(0, CeilDiv(num_col_blocks, (size_t)2)*(block_size / baby_step), [&](size_t bgn, size_t end) {
      for (size_t job_id = bgn; job_id < end; ++job_id) { // for each block pair of 2 cols
        size_t cb2 = job_id / (block_size / baby_step);
        size_t gs_id = job_id % (block_size / baby_step);
      // for (size_t cb2 = bgn; cb2 < end; ++cb2) { // for each block pair of 2 cols
        // for (size_t gs_id = 0; gs_id < (block_size / baby_step); ++gs_id) { // for each giant step
          RLWECt res; //zero

          seal::util::encrypt_zero_asymmetric(public_key, context, lhs_input[0].parms_id(), 
                                              lhs_input[0].is_ntt_form(), res);

          for (size_t bs_id = 0; bs_id < baby_step; ++bs_id) { // for each baby step

            for (size_t rb = 0; rb < num_row_blocks; ++rb) { // for each row block / input ct
              size_t rotated_input_idx = rb * baby_step + bs_id;
              size_t rhs_weight_idx = (rb * CeilDiv(num_col_blocks, (size_t)2) + cb2) * block_size + gs_id * baby_step + bs_id;
              RLWECt tmp;
              
              evaluator.multiply_plain(rotated_input[rotated_input_idx], rhs_weight[rhs_weight_idx], tmp);
              evaluator.add_inplace(res, tmp);
            }
          }
          evaluator.rotate_rows_inplace(res, gs_id * baby_step * in_shape[0], gal_keys);
          // evaluator.add_inplace(out[cb2], res);
          tmp_out[job_id] = std::move(res);
      }
    });

    yacl::parallel_for(0, CeilDiv(num_col_blocks, (size_t)2), [&](size_t bgn, size_t end) {
      for (size_t cb2 = bgn; cb2 < end; ++cb2) { // for each block pair of 2 cols
        // output ct must be encrypted zero
        seal::util::encrypt_zero_asymmetric(public_key, context, lhs_input[0].parms_id(), 
                                            lhs_input[0].is_ntt_form(), out[cb2]);
        for (size_t gs_id = 0; gs_id < (block_size / baby_step); ++gs_id) { // for each giant step
          size_t tmp_out_idx = cb2 * (block_size / baby_step) + gs_id;
          evaluator.add_inplace(out[cb2], tmp_out[tmp_out_idx]);
        }
      }
    });

  } else {
    SPU_ENFORCE_EQ(rhs_weight.size(), input_groups * num_row_blocks * num_col_blocks);
    SPU_ENFORCE_EQ(out.size(), input_groups * num_col_blocks);
    // block_size == 1
    SPU_ENFORCE_EQ(block_size, (size_t)1);

    // No rotation needed
    yacl::parallel_for(0, input_groups * num_col_blocks, [&](size_t bgn, size_t end) {
      for (size_t idx = bgn; idx < end; ++idx) { // for each output block
        size_t i = idx / num_col_blocks; // input group id
        size_t cb = idx % num_col_blocks; // col block id

        // output ct must be encrypted zero
        seal::util::encrypt_zero_asymmetric(public_key, context, lhs_input[0].parms_id(), 
                                            lhs_input[0].is_ntt_form(), out[idx]);

        for (size_t rb = 0; rb < num_row_blocks; ++rb) { // for each row block / input ct
          size_t lhs_input_idx = i * num_row_blocks + rb;
          size_t rhs_weight_idx = i * num_row_blocks * num_col_blocks + rb * num_col_blocks + cb;
          RLWECt tmp;
          evaluator.multiply_plain(lhs_input[lhs_input_idx], rhs_weight[rhs_weight_idx], tmp);
          evaluator.add_inplace(out[idx], tmp);
        }
      }
    });

  }
}

void SIMDBatchMMProt::ReshareOutputInplace(absl::Span<RLWECt> ct,
                          absl::Span<const uint64_t> share_mask,
                          const RLWEPublicKey& public_key,
                          const seal::SEALContext& context) {
  SPU_ENFORCE_EQ(CeilDiv(share_mask.size(), (size_t)simd_lane_), ct.size());

  seal::Evaluator evaluator(context);
  // seal::BatchEncoder encoder(context);
  
  constexpr int kMarginBitsForDec = 10;
  seal::parms_id_type final_level_id = context.last_parms_id();
  while (final_level_id != context.first_parms_id()) {
    auto cntxt = context.get_context_data(final_level_id);
    if (cntxt->total_coeff_modulus_bit_count() >=
        kMarginBitsForDec + cntxt->parms().plain_modulus().bit_count()) {
      break;
    }
    final_level_id = cntxt->prev_context_data()->parms_id();
  }

  yacl::parallel_for(0, ct.size(), [&](uint64_t bgn, uint64_t end) {
    for (uint64_t i = bgn; i < end; ++i) {
      RLWEPt rnd;
      RLWECt zero_enc;

      // 1. Noise flooding
      NoiseFloodInplace(ct[i], context);

      // 2. Drop some modulus for a smaller communication
      evaluator.mod_switch_to_inplace(ct[i], final_level_id);

      // 3. Re-randomize via adding enc(0)
      seal::util::encrypt_zero_asymmetric(public_key, context, ct[i].parms_id(), 
                                          ct[i].is_ntt_form(), zero_enc);
      evaluator.add_inplace(ct[i], zero_enc);
      
      // 4. Additive share
      size_t slice_bgn = i * simd_lane_;
      size_t slice_n = simd_lane_;
      EncodeSingle(share_mask.subspan(slice_bgn, slice_n), rnd);
      evaluator.sub_plain_inplace(ct[i], rnd);

      // 5. Truncate for smaller communication
      if (ct[i].coeff_modulus_size() == 1) {
        TruncateBFVForDecryption(ct[i], context);
      }
    }
  });
}

void SIMDBatchMMProt::NoiseFloodInplace(RLWECt &ct,
                                        const seal::SEALContext &context) {
  SPU_ENFORCE(seal::is_metadata_valid_for(ct, context));
  SPU_ENFORCE(ct.size() == 2);
  auto context_data = context.get_context_data(ct.parms_id());
  yacl::CheckNotNull(context_data.get());

  size_t num_coeffs = ct.poly_modulus_degree();
  size_t num_modulus = ct.coeff_modulus_size();

  // e * m for (semi-honest noise) e ~ Gassuain(0, stddev=3.19) and plaintext m
  // \in [-p/2, p/2).
  // |e * m| is bounded by 2*sqrt{N} * 6*stddev * p/2
  size_t noise_bits =
      kNoiseFloodBits + (modulus().bit_count() - 1) +
      std::ceil(
          std::log2(2. * std::sqrt(ct.poly_modulus_degree()) * 6. * 3.19));

  std::vector<uint64_t> wide_noise(num_coeffs * num_modulus);

  // sample r from [-2^{k-1}, 2^{k-1}]
  SampleRanomRNS(absl::MakeSpan(wide_noise), *context_data, noise_bits - 1,
                 ct.is_ntt_form());
  const auto &modulus = context_data->parms().coeff_modulus();

  seal::util::add_poly_coeffmod({wide_noise.data(), num_coeffs},
                                {ct.data(0), num_coeffs}, num_modulus, modulus,
                                {ct.data(0), num_coeffs});
}


} // namespace spu::mpc::cheetah
