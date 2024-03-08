// Copyright 2023 Ant Group Co., Ltd.
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

#include <numeric>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libspu/compiler/passes/pass_details.h"
#include "libspu/compiler/passes/passes.h"
#include "libspu/dialect/pphlo/ops.h"

namespace mlir::spu::pphlo {

namespace {

struct SortConversion : public OpRewritePattern<SimpleSortOp> {
private:
  bool sliceAttributesOk(llvm::ArrayRef<int64_t> in,
                         llvm::ArrayRef<int64_t> expected,
                         size_t allow_mismatch_at) const {
    for (size_t idx = 0; idx < in.size(); ++idx) {
      if ((in[idx] != expected[idx]) && (idx != allow_mismatch_at)) {
        return false;
      }
    }
    return true;
  }

  bool isAllOne(llvm::ArrayRef<int64_t> in) const {
    return std::all_of(in.begin(), in.end(), [](int64_t i) { return i == 1; });
  }

public:
  explicit SortConversion(MLIRContext *context)
      : OpRewritePattern<SimpleSortOp>(context) {}

  LogicalResult matchAndRewrite(SimpleSortOp op,
                                PatternRewriter &rewriter) const override {
    // Skip multikey sort
    if (op.getNumKeys() > 1 || op.getSortDirection() != SortDirection::ASC) {
      return failure();
    }

    auto sort_type = op.getType(0).dyn_cast<RankedTensorType>();
    auto rank = sort_type.getRank();
    int64_t sort_dim = op.getDimension();

    // Sort a super small tensor...no need to rewrite
    if (sort_type.getShape()[sort_dim] <= 2) {
      return failure();
    }

    // Collect and verify all uses
    auto uses = op->getUses();
    int64_t start = INT64_MAX;
    int64_t end = -1;
    llvm::SmallVector<SliceOp> slices_to_rewrite;
    for (const auto &use : uses) {
      auto slice = mlir::dyn_cast<SliceOp>(use.getOwner());
      if (!slice) {
        // If one of the use is not a slice, bailout
        return failure();
      }
      // Make sure slice only extract one element on sort dim and no stride
      if (!sliceAttributesOk(slice.getStartIndices(),
                             llvm::SmallVector<int64_t>(rank, 0), sort_dim) ||
          !sliceAttributesOk(slice.getLimitIndices(), sort_type.getShape(),
                             sort_dim) ||
          !isAllOne(slice.getStrides()) ||
          (slice.getLimitIndices()[sort_dim] -
               slice.getStartIndices()[sort_dim] >
           1)) {
        return failure();
      }
      start = std::min(start, slice.getStartIndices()[sort_dim]);
      end = std::max(end, slice.getLimitIndices()[sort_dim]);
      slices_to_rewrite.emplace_back(slice);
    }

    // If slice range is more than 2, bailout
    // Top_k does not guarantee order of elements between [k_lo, -1], so if n >
    // 2, we cannot guarantee slice yields the correct element.
    if (end - start > 2) {
      return failure();
    }

    auto input = op.getOperand(0);
    bool need_transpose = (sort_dim != (rank - 1));

    // If sort axis is not last dim, do a transpose to move sorting dim to last
    // first
    if (need_transpose) {
      llvm::SmallVector<int64_t> permutation(rank);
      // fill [0: sort_dim - 1] as 0..sort_dim-1
      std::iota(permutation.begin(), permutation.begin() + sort_dim, 0);
      // file [sort_dim, end-2] as sort_dim + 1...
      std::iota(permutation.begin() + sort_dim, permutation.end() - 1,
                sort_dim + 1);
      permutation.back() = sort_dim;

      llvm::SmallVector<int64_t> shape(rank);
      for (int64_t idx = 0; idx < rank; ++idx) {
        shape[idx] = sort_type.getShape()[permutation[idx]];
      }

      input = rewriter.create<TransposeOp>(
          op->getLoc(),
          RankedTensorType::get(shape, sort_type.getElementType()), input,
          permutation);

      sort_type = input.getType().dyn_cast<RankedTensorType>();
    }

    // Ask top_k to return k_hi elements
    auto k_hi = *sort_type.getShape().rbegin() - start;
    // Set k_lo to n elements, so [n:-1] has the smallest n elements in top_k
    auto k_lo = k_hi - (end - start);
    llvm::SmallVector<int64_t> topk_shape{sort_type.getShape().begin(),
                                          sort_type.getShape().end()};
    topk_shape.back() = k_hi;
    auto top_k_value_type =
        RankedTensorType::get(topk_shape, sort_type.getElementType());

    // rewrite to top_k
    auto call = rewriter.create<CustomCallOp>(
        op->getLoc(), TypeRange{top_k_value_type}, input, "mhlo.topk");

    auto attr = DictionaryAttr::get(
        op->getContext(), {NamedAttribute(rewriter.getStringAttr("k"),
                                          rewriter.getI64IntegerAttr(k_lo)),
                           NamedAttribute(rewriter.getStringAttr("largest"),
                                          rewriter.getBoolAttr(true)),
                           NamedAttribute(rewriter.getStringAttr("k_hi"),
                                          rewriter.getI64IntegerAttr(k_hi)),
                           NamedAttribute(rewriter.getStringAttr("value_only"),
                                          rewriter.getBoolAttr(true))});
    call->setAttr("mhlo.attributes", attr);

    Value topk = call.getResult(0);

    if (need_transpose) {
      // Transpose result back
      llvm::SmallVector<int64_t> permutation(rank);
      std::iota(permutation.begin(), permutation.begin() + sort_dim, 0);
      std::iota(permutation.begin() + sort_dim + 1, permutation.end(),
                sort_dim);
      permutation[sort_dim] = rank - 1;

      llvm::SmallVector<int64_t> shape(rank);
      for (int64_t idx = 0; idx < rank; ++idx) {
        shape[idx] = topk_shape[permutation[idx]];
      }

      topk = rewriter.create<TransposeOp>(
          op->getLoc(),
          RankedTensorType::get(shape, sort_type.getElementType()), topk,
          permutation);
    }

    // rewrite all slices
    for (const auto &use : uses) {
      auto slice = mlir::dyn_cast<SliceOp>(use.getOwner());
      auto offset = slice.getStartIndices()[sort_dim] - start;
      llvm::SmallVector<int64_t> new_start(slice.getStartIndices().begin(),
                                           slice.getStartIndices().end());
      llvm::SmallVector<int64_t> new_limit(slice.getLimitIndices().begin(),
                                           slice.getLimitIndices().end());
      new_start[sort_dim] = topk_shape.back() - 1 - offset;
      new_limit[sort_dim] = new_start[sort_dim] + 1;
      rewriter.replaceOpWithNewOp<SliceOp>(slice, slice->getResultTypes()[0],
                                           topk, new_start, new_limit,
                                           slice.getStrides());
    }

    rewriter.replaceOp(op, topk);

    return success();
  }
};

struct PartialSortToTopK : public PartialSortToTopKBase<PartialSortToTopK> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOwningPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

private:
  static void populateOwningPatterns(RewritePatternSet *patterns,
                                     MLIRContext *ctx) {
    patterns->insert<SortConversion>(ctx);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createPartialSortToTopK() {
  return std::make_unique<PartialSortToTopK>();
}

} // namespace mlir::spu::pphlo
