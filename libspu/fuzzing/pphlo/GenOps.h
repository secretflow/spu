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

#ifndef mDefOpGenUnary
#define mDefOpGenUnary mDefOpGen
#endif
#ifndef mDefOpGenBinary
#define mDefOpGenBinary mDefOpGen
#endif
#ifndef mDefOpGenBinaryLogical
#define mDefOpGenBinaryLogical mDefOpGen
#endif
#ifndef mDefOpGenComparison
#define mDefOpGenComparison mDefOpGen
#endif
// clang-format off
mDefOpGen(ConstantIntOp)
mDefOpGen(ConstantFloatOp)
mDefOpGenUnary(SqrtOp)
mDefOpGenUnary(RsqrtOp)
mDefOpGenUnary(TanhOp)
mDefOpGenUnary(NegOp)
mDefOpGenUnary(ExpOp)
mDefOpGenUnary(Expm1Op)
mDefOpGenUnary(LogOp)
mDefOpGenUnary(Log1pOp)
mDefOpGenUnary(CeilOp)
mDefOpGenUnary(FloorOp)
mDefOpGenUnary(RoundOp)
mDefOpGenUnary(AbsOp)
mDefOpGenUnary(ReciprocalOp)
mDefOpGenUnary(LogisticOp)
mDefOpGenUnary(NotOp)
mDefOpGenUnary(SignOp)

mDefOpGenBinary(SubtractOp)
mDefOpGenBinary(MaxOp)
mDefOpGenBinary(MinOp)
mDefOpGenBinary(DivOp)
mDefOpGenBinary(AddOp)
mDefOpGenBinary(MulOp)
mDefOpGenBinary(PowOp)
mDefOpGenBinary(RemOp)
mDefOpGenBinary(ShiftLeftOp)
mDefOpGenBinary(ShiftRightLogicalOp)
mDefOpGenBinary(ShiftRightArithmeticOp)

mDefOpGenBinaryLogical(AndOp)
mDefOpGenBinaryLogical(OrOp)
mDefOpGenBinaryLogical(XorOp)

mDefOpGenComparison(EqualOp)
mDefOpGenComparison(NotEqualOp)
mDefOpGenComparison(GreaterEqualOp)
mDefOpGenComparison(GreaterOp)
mDefOpGenComparison(LessEqualOp)
mDefOpGenComparison(LessOp)

// clang-format on

#undef mDefOpGen
#undef mDefOpGenUnary
#undef mDefOpGenBinary
#undef mDefOpGenBinaryLogical
#undef mDefOpGenComparison