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

// This is a template for dispatch op
template <typename OpT, typename... MoreOpT>
static void dispatchOp(OpExecutor *executor, SPUContext *sctx,
                       SymbolScope *sscope, mlir::Operation &op,
                       const ExecutionOptions &opts,
                       std::function<std::string(mlir::Operation &)>
                           &&name_printer = defaultOpNamePrinter) {
  if (auto casted = llvm::dyn_cast<OpT>(op)) {
    // Execute op
    SPU_TRACE_ACTION(GET_TRACER(sctx), sctx->lctx(), (TR_HLO | TR_LAR), ~TR_HLO,
                     name_printer(op));
    execute(executor, sctx, sscope, casted, opts);
  } else {
    if constexpr (!sizeof...(MoreOpT)) {
      SPU_THROW("Unhandled mlir op {} at {}", mlir::spu::mlirObjectToString(op),
                mlir::spu::mlirObjectToString(op.getLoc()));
    } else {
      dispatchOp<MoreOpT...>(executor, sctx, sscope, op, opts);
    }
  }
}
