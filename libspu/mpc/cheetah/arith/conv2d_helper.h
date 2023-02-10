#pragma once
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/shape_util.h"
#include "libspu/mpc/cheetah/arith/conv2d_prot.h"

namespace spu::mpc::cheetah {

// Obtain a subtensor
// NxHxWxC -> HxWxC
// HxWxIxO -> HxWxI
NdArrayRef SliceDim(const NdArrayRef &base, int64_t dim, int64_t at);

// forward
struct Sliced3DTensor;
struct InputIndexer;
struct KernelIndexer;

class Conv2DHelper {
 public:
  Conv2DHelper(const Conv2DProtocol::Meta &meta, const Shape3D &subshape);

  int64_t slice_size(int d) const {
    SPU_ENFORCE(d >= 0 && d < 3);
    return slices_[d];
  }

  int64_t num_slices() const { return calcNumel(slices_); }

  Shape3D GetSliceShape(const Shape3D &indices) const;

  Sliced3DTensor partition(const NdArrayRef &tensor,
                           const Shape3D &indices) const;

  void GetResultCoefficients(Shape3D indices, std::vector<size_t> *coefficients,
                             Shape3D *oshape = nullptr) const;

 private:
  Conv2DProtocol::Meta meta_;
  Shape3D subshape_;
  Shape3D partition_windows_;

  Shape3D slices_;
};

struct Sliced3DTensor {
 private:
  Sliced3DTensor(const NdArrayRef &base, const Shape3D &offsets,
                 const Shape3D &extents);

 public:
  static Sliced3DTensor Wrap(const NdArrayRef &base, const Shape3D &offsets,
                             const Shape3D &partition_shape);

  Sliced3DTensor(const Sliced3DTensor &oth) = default;

  Sliced3DTensor(Sliced3DTensor &&oth) = default;

  Sliced3DTensor &operator=(const Sliced3DTensor &oth) = delete;

  template <typename T>
  T at(int64_t h, int64_t w, int64_t c) const {
    // NOTE: HxWxC order
    constexpr int kH = 0;
    constexpr int kW = 1;
    constexpr int kC = 2;

    SPU_ENFORCE(h >= 0 && h < mock_extents_[kH]);
    SPU_ENFORCE(w >= 0 && w < mock_extents_[kW]);
    SPU_ENFORCE(c >= 0 && c < mock_extents_[kC]);

    if (c >= extents_[kC]) {
      return static_cast<T>(0);
    }
    if (h < 0 || h >= extents_[kH]) {
      return static_cast<T>(0);
    }
    if (w < 0 || w >= extents_[kW]) {
      return static_cast<T>(0);
    }

    return base_.at<T>({h + offsets_[kH], w + offsets_[kW], c + offsets_[kC]});
  }

  Shape3D shape() const { return mock_extents_; }

  int64_t numel() const { return calcNumel(shape()); }

  FieldType field() const;

  void ZeroPadAs(const Shape3D &extents) {
    for (size_t d = 0; d < 3; ++d) {
      SPU_ENFORCE(extents[d] > 0);
    }
    mock_extents_ = extents;
  }

 private:
  const NdArrayRef &base_;
  Shape3D offsets_;
  Shape3D extents_;

  Shape3D mock_extents_;
};

struct InputIndexer {
 public:
  InputIndexer(Shape3D input_shape, Shape3D kernel_shape);

  int64_t operator()(int64_t h, int64_t w, int64_t c) const;

 private:
  Shape3D shape_;
  int64_t offset_;
};

struct KernelIndexer {
 public:
  KernelIndexer(Shape3D input_shape, Shape3D kernel_shape);

  int64_t operator()(int64_t h, int64_t w, int64_t c) const;

  int64_t index_begin() const { return begin_; }

 private:
  Shape3D shape_;
  int64_t row_nskip_, offset_, begin_;
};

}  // namespace spu::mpc::cheetah
