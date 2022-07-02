// Copyright 2021 Ant Group Co., Ltd.
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

#include "spu/dialect/pphlo_ops.h"

#include <set>
#include <unordered_map>

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "spu/dialect/pphlo_attrs.h"
#include "spu/dialect/pphlo_ops.h.inc"
#include "spu/dialect/pphlo_struct_attrs.cc.inc"

namespace mlir::pphlo {

namespace {
Type convertPtTypeToPPhloType(Type ptType) {
  return pphlo::PublicType::get(ptType.getContext(), ptType);
}
}  // namespace

template <typename T>
static LogicalResult Verify(T op) {
  return success();
}

// Builds a constant op with the specified attribute `value`.
void ConstOp::build(OpBuilder& builder, OperationState& result,
                    Attribute value) {
  Type type;
  if (auto elemAttr = value.dyn_cast<DenseElementsAttr>()) {
    auto valueType = elemAttr.getType().dyn_cast<RankedTensorType>();
    type = RankedTensorType::get(
        valueType.getShape(),
        convertPtTypeToPPhloType(valueType.getElementType()));
  }

  assert(type && "unsupported attribute type for building pphlo.constant");
  result.types.push_back(type);
  result.addAttribute("value", value);
}

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return value();
}

OpFoldResult ConvertOp::fold(ArrayRef<Attribute> operands) {
  auto operand_ty = getOperand().getType().cast<TensorType>();
  auto result_ty = getResult().getType().cast<TensorType>();
  if (operand_ty == result_ty) {
    return getOperand();
  }

  return {};
}

OpFoldResult ReciprocalOp::fold(ArrayRef<Attribute> operands) {
  auto val = operands[0].dyn_cast_or_null<DenseFPElementsAttr>();

  if (!val) {
    return {};
  }

  if (val.isSplat()) {
    auto splat_val = val.getSplatValue<APFloat>();
    APFloat one(splat_val.getSemantics(), 1);

    return SplatElementsAttr::get(operands[0].getType().dyn_cast<ShapedType>(),
                                  one / splat_val);
  }

  llvm::SmallVector<APFloat, 4> values;
  values.reserve(val.getNumElements());

  auto first_val = *val.getValues<APFloat>().begin();
  APFloat one(first_val.getSemantics(), 1);

  for (auto it : val.getValues<APFloat>()) {
    values.push_back(one / it);
  }

  return DenseFPElementsAttr::get(operands[0].getType().dyn_cast<ShapedType>(),
                                  values);
}

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  auto operand_shape = getOperand().getType().cast<TensorType>().getShape();
  auto result_shape = getResult().getType().cast<TensorType>().getShape();
  if (operand_shape == result_shape) {
    return getOperand();
  }
  return {};
}

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  for (auto it : llvm::enumerate(permutation().getValues<APInt>())) {
    if (it.index() != it.value()) {
      return {};
    }
  }
  return getOperand();
}

template <typename T>
static void printField(AsmPrinter& printer, StringRef name, T field,
                       StringRef& separator) {
  if (field != 0) {
    printer << separator << name << " = " << field;
    separator = ", ";
  }
}
template <typename T>
static void printField(AsmPrinter& printer, StringRef name, ArrayRef<T> field,
                       StringRef& separator) {
  if (!field.empty()) {
    printer << separator << name << " = [";
    llvm::interleaveComma(field, printer);
    printer << "]";
    separator = ", ";
  }
}

template <typename... Ts>
static void printStruct(AsmPrinter& printer, StringRef name,
                        Ts... printFields) {
  printer << "<";
  StringRef separator = "";
  // Fold expression to print each entry in the parameter pack.
  // TODO(mhlo-team): this can be simplified when TF moves to C++17.
  using unused = int[];
  (void)unused{0, (printField(printer, std::get<0>(printFields),
                              std::get<1>(printFields), separator),
                   0)...};
  printer << ">";
}

/// Parse a custom attribute that resembles a struct of the form
/// <
///   foo = something_parsed_by_custom_parser,
///   bar = something_parsed_by_different_custom_parser,
///   baz something_parsed_by_another_custom_parser
/// >
/// The optional argument `parse_equal` array can be used to denote if
/// '=' follows the keyword (see baz in the example above) for a field. If
/// not provided, all fields must be followed by a '='.
static ParseResult parseStruct(
    AsmParser& parser, ArrayRef<StringRef> keywords,
    ArrayRef<llvm::function_ref<ParseResult()>> parseFuncs,
    ArrayRef<bool> parse_equal = {}) {
  assert(keywords.size() == parseFuncs.size());
  assert(parse_equal.empty() || parse_equal.size() == keywords.size());
  SmallVector<bool> seen(keywords.size(), false);
  while (failed(parser.parseOptionalGreater())) {
    bool foundOne = false;
    for (const auto& it : llvm::enumerate(keywords)) {
      size_t index = it.index();
      StringRef keyword = it.value();
      if (succeeded(parser.parseOptionalKeyword(keyword))) {
        if (seen[index]) {
          return parser.emitError(parser.getCurrentLocation())
                 << "duplicated `" << keyword << "` entry";
        }
        if (parse_equal.empty() || parse_equal[index]) {
          if (failed(parser.parseEqual())) {
            return failure();
          }
        }
        if (failed(parseFuncs[index]())) {
          return failure();
        }
        if (failed(parser.parseOptionalComma())) {
          return parser.parseGreater();
        }
        seen[index] = true;
        foundOne = true;
      }
    }
    if (!foundOne) {
      auto parseError = parser.emitError(parser.getCurrentLocation())
                        << "expected one of: ";
      llvm::interleaveComma(keywords, parseError, [&](StringRef kw) {
        parseError << '`' << kw << '`';
      });
      return parseError;
    }
  }
  return success();
}

static ParseResult parseDims(AsmParser& parser, SmallVector<int64_t>& dims) {
  dims.clear();
  if (parser.parseLSquare()) {
    return failure();
  }
  while (failed(parser.parseOptionalRSquare())) {
    dims.emplace_back();
    if (parser.parseInteger(dims.back())) {
      return failure();
    }
    parser.parseOptionalComma();
  }
  return success();
}

void GatherDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(printer, "gather", std::make_pair("offset_dims", getOffsetDims()),
              std::make_pair("collapsed_slice_dims", getCollapsedSliceDims()),
              std::make_pair("start_index_map", getStartIndexMap()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}

Attribute GatherDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }

  SmallVector<int64_t> offset_dims;
  SmallVector<int64_t> collapsed_slice_dims;
  SmallVector<int64_t> start_index_map;
  int64_t index_vector_dim = 0;

  if (failed(parseStruct(
          parser,
          {"offset_dims", "collapsed_slice_dims", "start_index_map",
           "index_vector_dim"},
          {[&]() { return parseDims(parser, offset_dims); },
           [&]() { return parseDims(parser, collapsed_slice_dims); },
           [&]() { return parseDims(parser, start_index_map); },
           [&]() { return parser.parseInteger(index_vector_dim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing gather dimension numbers attribute";
    return {};
  }

  return GatherDimensionNumbersAttr::get(parser.getContext(), offset_dims,
                                         collapsed_slice_dims, start_index_map,
                                         index_vector_dim);
}

namespace {
enum NonSpatialDim : int64_t {
  IOBatch = -1,    // Input or output batch dimension
  IOFeature = -2,  // Input or output feature dimension
  KIFeature = -3,  // Kernel input feature dimension
  KOFeature = -4,  // Kernel output feature dimensions.
};

struct DenseMapInfoNonSpatialDim {
  static inline NonSpatialDim getEmptyKey() {
    return NonSpatialDim(DenseMapInfo<int64_t>::getEmptyKey());
  }

  static inline NonSpatialDim getTombstoneKey() {
    return NonSpatialDim(DenseMapInfo<int64_t>::getTombstoneKey());
  }

  static unsigned getHashValue(const NonSpatialDim& Key) {
    return DenseMapInfo<int64_t>::getHashValue(Key);
  }

  static bool isEqual(const NonSpatialDim& LHS, const NonSpatialDim& RHS) {
    return LHS == RHS;
  }
};

char NonSpatialDimToString(NonSpatialDim dim) {
  switch (dim) {
    case IOBatch:
      return 'b';
    case IOFeature:
      return 'f';
    case KIFeature:
      return 'i';
    case KOFeature:
      return 'o';
  }
  llvm_unreachable("Unexpected spatial dim");
}
}  // namespace

// Custom printer and parser for struct attributes.
void printConvolutionDimensions(AsmPrinter& p, ConvDimensionNumbersAttr dnums) {
  constexpr int64_t kUnknownDim = std::numeric_limits<int64_t>::min();
  auto print_dim =
      [&](ArrayRef<int64_t> spatial_dims,
          ArrayRef<std::pair<int64_t, NonSpatialDim>> non_spatial_dims) {
        int64_t num_dims = 0;
        if (!spatial_dims.empty()) {
          num_dims =
              *std::max_element(spatial_dims.begin(), spatial_dims.end()) + 1;
        }
        for (const auto& dim : non_spatial_dims) {
          num_dims = std::max(num_dims, dim.first + 1);
        }

        llvm::SmallVector<int64_t> dims(num_dims, kUnknownDim);
        // Fill each element of dims with a (< 0) NonSpatialDim enum or a (>=0)
        // spatial dimension index.
        for (const std::pair<int64_t, NonSpatialDim>& non_spatial_dim :
             non_spatial_dims) {
          dims[non_spatial_dim.first] = non_spatial_dim.second;
        }
        for (const auto& spatial_dim : llvm::enumerate(spatial_dims)) {
          dims[spatial_dim.value()] = static_cast<int64_t>(spatial_dim.index());
        }

        // Each dimension numbers will be printed as a comma separated list
        // surrounded by square brackets, e.g., [b, 0, 1, 2, f]
        p << '[';
        llvm::interleaveComma(dims, p, [&](int64_t dim) {
          if (dim == kUnknownDim) {
            p << "?";
          } else if (dim >= 0) {
            p << dim;
          } else {
            p << NonSpatialDimToString(static_cast<NonSpatialDim>(dim));
          }
        });
        p << ']';
      };

  print_dim(dnums.getInputSpatialDimensions(),
            {{dnums.getInputBatchDimension(), IOBatch},
             {dnums.getInputFeatureDimension(), IOFeature}});
  p << "x";
  print_dim(dnums.getKernelSpatialDimensions(),
            {{dnums.getKernelInputFeatureDimension(), KIFeature},
             {dnums.getKernelOutputFeatureDimension(), KOFeature}});
  p << "->";
  print_dim(dnums.getOutputSpatialDimensions(),
            {{dnums.getOutputBatchDimension(), IOBatch},
             {dnums.getOutputFeatureDimension(), IOFeature}});
}

void printConvolutionDimensions(AsmPrinter& p, Operation* /*unused*/,
                                ConvDimensionNumbersAttr dnums) {
  printConvolutionDimensions(p, dnums);
}

void ConvDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printer << "<";
  printConvolutionDimensions(printer, *this);
  printer << ">";
}

ParseResult parseConvolutionDimensions(AsmParser& parser,
                                       ConvDimensionNumbersAttr& dnums) {
  // Parsing a single set of dim numbers gives the spatial dimensions as a
  // single ArrayRef<int64_t> and a list of non-spatial dimensions as
  // IntegerAttrs (indexed by the NonSpatialDim enum).
  using parse_dim_result_t =
      std::pair<llvm::SmallVector<int64_t>,
                llvm::SmallDenseMap<NonSpatialDim, int64_t, 4,
                                    DenseMapInfoNonSpatialDim>>;

  // Note that the allowed_non_spatial_dims is a set (as opposed to unordered
  // set) because its used to print a list of allowed non spatial dims in the
  // error messages, so making it a set keeps the error messages deterministic.
  auto parse_dims =
      [&](std::set<NonSpatialDim, std::greater<>> allowed_non_spatial_dims,
          parse_dim_result_t& parsed_dims) -> ParseResult {
    auto& spatial_dims = std::get<0>(parsed_dims);
    auto& non_spatial_dims = std::get<1>(parsed_dims);
    spatial_dims.clear();
    non_spatial_dims.clear();

    // Parse the starting [
    if (parser.parseLSquare()) {
      return failure();
    }

    llvm::SmallDenseMap<int64_t, int64_t> spatial_dims_map;
    constexpr int64_t kInvalidDimension = -1;
    // Keep track of the maximum spatial dimension parsed as we expect to see
    // all the dimensions from 0 to maximum dimension parsed.
    int64_t max_parsed_spatial_dim = kInvalidDimension;

    int64_t index = 0;
    do {
      int64_t spatial_dim{};
      auto dim_location = parser.getCurrentLocation();
      OptionalParseResult parseResult =
          parser.parseOptionalInteger(spatial_dim);
      if (parseResult.hasValue()) {
        if (parseResult.getValue().failed()) {
          return failure();
        }
        // We were successful in parsing an integer. Check if it is a valid
        // dimension (non-negative and no duplicate) and add its index to the
        // spatial dims map.
        if (spatial_dim < 0) {
          return parser.emitError(dim_location)
                 << "Unexpected dimension " << spatial_dim;
        }
        if (!spatial_dims_map
                 .insert(std::pair<int64_t, int64_t>(spatial_dim, index))
                 .second) {
          return parser.emitError(dim_location)
                 << "Duplicate entries for spatial dimension " << spatial_dim;
        }
        max_parsed_spatial_dim = std::max(spatial_dim, max_parsed_spatial_dim);
      } else if (!parser.parseOptionalQuestion()) {
        // Do nothing other than increment `index` at the bottom of the loop;
        // '?' means "unknown dimension", and it's not represented in the
        // return value of this function.
      } else {
        // We did not parse an integer or question mark. We expect a keyword
        // token.
        StringRef keyword;
        if (parser.parseKeyword(&keyword)) {
          return failure();
        }
        if (keyword.size() != 1 || allowed_non_spatial_dims.empty()) {
          return parser.emitError(dim_location, "Unexpected keyword ")
                 << keyword;
        }
        // Check if the keyword matches one of the allowed non-spatial dims.
        // If so, add it to the non_spatial dims and remove it from the
        // allowed set so that it won't be allowed again.
        bool is_allowed = false;
        for (NonSpatialDim allowed : allowed_non_spatial_dims) {
          if (keyword[0] == NonSpatialDimToString(allowed)) {
            non_spatial_dims.insert({allowed, index});
            allowed_non_spatial_dims.erase(allowed);
            is_allowed = true;
            break;
          }
        }

        if (!is_allowed) {
          mlir::InFlightDiagnostic diag =
              parser.emitError(dim_location, "Unexpected dimension ");
          diag << keyword << ", expecting ";
          llvm::interleaveComma(
              allowed_non_spatial_dims, diag,
              [&](NonSpatialDim dim) { diag << NonSpatialDimToString(dim); });
          return diag;
        }
      }
      index++;
    } while (parser.parseOptionalComma().succeeded());

    // Make sure all expected non-spatial dimensions are parsed.
    if (!allowed_non_spatial_dims.empty()) {
      mlir::InFlightDiagnostic diag =
          parser.emitError(parser.getCurrentLocation(), "Expected dimensions ");
      llvm::interleaveComma(
          allowed_non_spatial_dims, diag,
          [&](NonSpatialDim dim) { diag << NonSpatialDimToString(dim); });
      diag << " not specified";
      return diag;
    }

    // parse ending ]
    if (parser.parseRSquare()) {
      return failure();
    }

    // Number of expected spatial dimensions is one more than the maximum parsed
    // spatial dimension. For example, if we parse [0, 3, 2, b, i, 1], then the
    // maximum parsed spatial dimension is 3 and the number of expected spatial
    // dimensions is 4.
    int64_t num_spatial_dimensions = max_parsed_spatial_dim + 1;
    spatial_dims.resize(num_spatial_dimensions);
    // Store spatial dimensions in a vector which maps spatial dim (vector
    // index) -> index in the tensor dimensions. For example, for parsed
    // dimension numbers [0, 3, 2, b, i, 1] the spatial dimension vector would
    // be [0, 5, 2, 1].
    //
    // Get all the unspecified spatial dimensions to throw a more descriptive
    // error later.
    llvm::SmallVector<int64_t> unspecified_spatial_dims;
    constexpr int kPrintUnspecifiedDimsMax = 10;
    for (int dim = 0; dim < num_spatial_dimensions; ++dim) {
      auto it = spatial_dims_map.find(dim);
      if (it == spatial_dims_map.end()) {
        // Have an upper bound on the number of unspecified dimensions to print
        // in the error message.
        if (unspecified_spatial_dims.size() < kPrintUnspecifiedDimsMax)
          unspecified_spatial_dims.push_back(dim);
        continue;
      }
      spatial_dims[dim] = it->second;
    }

    // Verify that we got all spatial dimensions between 0 and maximum parsed
    // spatial dimension.
    if (!unspecified_spatial_dims.empty()) {
      mlir::InFlightDiagnostic diag = parser.emitError(
          parser.getCurrentLocation(), "Expected spatial dimensions ");
      llvm::interleaveComma(unspecified_spatial_dims, diag);
      diag << " not specified";
      return diag;
    }

    return success();
  };

  parse_dim_result_t parsed_dims;
  if (parse_dims({IOBatch, IOFeature}, parsed_dims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> input_spatial_dimensions = parsed_dims.first;
  int64_t input_batch_dimension = parsed_dims.second[IOBatch];
  int64_t input_feature_dimension = parsed_dims.second[IOFeature];
  if (parser.parseKeyword("x")) {
    return failure();
  }
  if (parse_dims({KIFeature, KOFeature}, parsed_dims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> kernel_spatial_dimensions = parsed_dims.first;
  int64_t kernel_input_feature_dimension = parsed_dims.second[KIFeature];
  int64_t kernel_output_feature_dimension = parsed_dims.second[KOFeature];
  if (parser.parseArrow()) {
    return failure();
  }
  if (parse_dims({IOBatch, IOFeature}, parsed_dims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> output_spatial_dimensions = parsed_dims.first;
  int64_t output_batch_dimension = parsed_dims.second[IOBatch];
  int64_t output_feature_dimension = parsed_dims.second[IOFeature];
  dnums = ConvDimensionNumbersAttr::get(
      parser.getBuilder().getContext(), input_batch_dimension,
      input_feature_dimension, input_spatial_dimensions,
      kernel_input_feature_dimension, kernel_output_feature_dimension,
      kernel_spatial_dimensions, output_batch_dimension,
      output_feature_dimension, output_spatial_dimensions);

  return success();
}

Attribute ConvDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }
  ConvDimensionNumbersAttr dnums;

  if (failed(parseConvolutionDimensions(parser, dnums))) {
    return {};
  }
  if (failed(parser.parseGreater())) {
    return {};
  }
  return dnums;
}

namespace {
// Custom formatting for convolution window attributes.
void printWindowAttribute(OpAsmPrinter& p, DenseElementsAttr attribute) {
  if (attribute.getElementType().isInteger(/*width=*/1)) {
    // boolean attribute.
    llvm::interleaveComma(attribute.getValues<bool>(), p,
                          [&](bool b) { p << (b ? 1 : 0); });
    return;
  }
  if (attribute.getType().getRank() == 2) {
    // Padding is Nx2 attribute.
    auto it = attribute.value_begin<int64_t>();
    std::vector<std::pair<int64_t, int64_t>> values(attribute.getNumElements() /
                                                    2);
    for (auto& item : values) {
      int64_t first = *it;
      ++it;
      int64_t second = *it;
      ++it;
      item = {first, second};
    }
    llvm::interleaveComma(
        values, p, [&](const std::pair<int64_t, int64_t> pair) {
          p << '[' << pair.first << ", " << pair.second << ']';
        });
  } else {
    llvm::interleaveComma(attribute.getValues<int64_t>(), p);
  }
}
}  // namespace

void printWindowAttributes(OpAsmPrinter& p, Operation* op,
                           llvm::Optional<DenseIntElementsAttr> window_strides,
                           llvm::Optional<DenseIntElementsAttr> padding,
                           llvm::Optional<DenseIntElementsAttr> lhs_dilation,
                           llvm::Optional<DenseIntElementsAttr> rhs_dilation) {
  using pair_t = std::pair<DenseElementsAttr, StringRef>;
  std::array<pair_t, 5> printed_attributes = {{
      {window_strides ? *window_strides : nullptr, "stride"},
      {padding ? *padding : nullptr, "pad"},
      {lhs_dilation ? *lhs_dilation : nullptr, "lhs_dilate"},
      {rhs_dilation ? *rhs_dilation : nullptr, "rhs_dilate"},
  }};

  // Do not print attributes that do no exist.
  auto non_null_attributes = llvm::make_filter_range(
      printed_attributes,
      [](const pair_t& a) { return static_cast<bool>(a.first); });

  llvm::interleaveComma(non_null_attributes, p, [&](const pair_t& a) {
    p << a.second << " = [";
    printWindowAttribute(p, a.first);
    p << "]";
  });
}

ParseResult parseWindowAttributes(OpAsmParser& parser,
                                  DenseIntElementsAttr& window_strides,
                                  DenseIntElementsAttr& padding,
                                  DenseIntElementsAttr& lhs_dilation,
                                  DenseIntElementsAttr& rhs_dilation) {
  StringRef attribute_name;

  // Helper to parse an array of the form [ e0, e1, .. ]
  auto parse_array = [&](std::function<ParseResult(void)> parse_element,
                         llvm::Optional<size_t> expected_size =
                             llvm::None) -> ParseResult {
    if (parser.parseLSquare()) {
      return failure();
    }
    size_t size = 0;
    do {
      if (parse_element()) {
        return failure();
      }
      size++;
    } while (parser.parseOptionalComma().succeeded());
    if (parser.parseRSquare()) {
      return failure();
    }
    if (expected_size && size != *expected_size) {
      return parser.emitError(parser.getCurrentLocation(),
                              "Expected array with")
             << *expected_size << " elements, got " << size
             << " elements instead";
    }
    return success();
  };

  llvm::StringSet<> allowed_attribute_names{
      {"stride", "pad", "lhs_dilate", "rhs_dilate", "reverse"}};

  while (parser.parseOptionalKeyword(&attribute_name).succeeded()) {
    // Verify that the attribute name is valid and erase it.
    if (!allowed_attribute_names.erase(attribute_name)) {
      return parser.emitError(parser.getCurrentLocation(),
                              "Unexpected keyword ")
             << attribute_name;
    }

    if (parser.parseEqual()) {
      return failure();
    }

    // parse the attribute value. We need to support either 1D and Nx2 array of
    // integers to parse.
    llvm::SmallVector<int64_t> values;
    auto int64_parser = [&]() {
      return parser.parseInteger(values.emplace_back(0));
    };

    if (attribute_name == "pad") {
      // Parse a 2D array of integers.
      auto inner_parser = [&]() {
        return parse_array(int64_parser, /*expected_size=*/2);
      };
      if (parse_array(inner_parser)) {
        return failure();
      }
      const int64_t size = static_cast<int64_t>(values.size());
      // values should be filled with the Nx2 padding values.
      auto ty = RankedTensorType::get({size / 2, 2},
                                      parser.getBuilder().getIntegerType(64));
      padding = DenseIntElementsAttr::get(ty, values);
    } else {
      // Parse 1D array of integers.
      if (parse_array(int64_parser)) {
        return failure();
      }
      auto attr = parser.getBuilder().getI64TensorAttr(values);

      if (attribute_name == "stride") {
        window_strides = attr;
      } else if (attribute_name == "lhs_dilate") {
        lhs_dilation = attr;
      } else if (attribute_name == "rhs_dilate") {
        rhs_dilation = attr;
      } else {
        llvm_unreachable("Unexpected attribute name");
      }
    }
    // continue parsing if there is a comma at the end.
    if (parser.parseOptionalComma().failed()) {
      break;
    }
  }
  return success();
}

}  // namespace mlir::pphlo

#define GET_OP_CLASSES
#include "spu/dialect/pphlo_ops.cc.inc"