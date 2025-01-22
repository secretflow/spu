#pragma once

#include "libspu/core/type.h"

#define GET_BTYPE(input, type) input.eltype().as<type>()->getBacktype()
#define GET_AFIELD(input, type) input.eltype().as<type>()->field()
#define GET_BTYPE_FROM_BW(bitwidth, type) makeType<type>(calcBShareBacktype(bitwidth), bitwidth);

