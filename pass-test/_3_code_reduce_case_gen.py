import json

test_case_full = """HLO_IR|HloModule jit_test_kmeans_random, entry_computation_layout={(s32[4,2]{1,0})->f32[4,2]{1,0}}, frontend_attributes={xla.sdy.meshes={}}
HLO_IR|
HLO_IR|_unstack.13 {
HLO_IR|  Arg_0.14 = s32[5,4]{1,0} parameter(0)
HLO_IR|  slice.15 = s32[1,4]{1,0} slice(Arg_0.14), slice={[0:1], [0:4]}
HLO_IR|  reshape.16 = s32[4]{0} reshape(slice.15)
HLO_IR|  slice.17 = s32[1,4]{1,0} slice(Arg_0.14), slice={[1:2], [0:4]}
HLO_IR|  reshape.18 = s32[4]{0} reshape(slice.17)
HLO_IR|  slice.19 = s32[1,4]{1,0} slice(Arg_0.14), slice={[2:3], [0:4]}
HLO_IR|  reshape.20 = s32[4]{0} reshape(slice.19)
HLO_IR|  slice.21 = s32[1,4]{1,0} slice(Arg_0.14), slice={[3:4], [0:4]}
HLO_IR|  reshape.22 = s32[4]{0} reshape(slice.21)
HLO_IR|  slice.23 = s32[1,4]{1,0} slice(Arg_0.14), slice={[4:5], [0:4]}
HLO_IR|  reshape.24 = s32[4]{0} reshape(slice.23)
HLO_IR|  ROOT tuple.25 = (s32[4]{0}, s32[4]{0}, s32[4]{0}, s32[4]{0}, s32[4]{0}) tuple(reshape.16, reshape.18, reshape.20, reshape.22, reshape.24)
HLO_IR|}
HLO_IR|
HLO_IR|region_0.73 {
HLO_IR|  Arg_0.74 = s32[] parameter(0)
HLO_IR|  Arg_1.75 = s32[] parameter(1)
HLO_IR|  ROOT add.76 = s32[] add(Arg_0.74, Arg_1.75)
HLO_IR|}
HLO_IR|
HLO_IR|region_1.78 {
HLO_IR|  Arg_0.79 = s32[] parameter(0)
HLO_IR|  Arg_2.81 = s32[] parameter(2)
HLO_IR|  compare.83 = pred[] compare(Arg_0.79, Arg_2.81), direction=LT
HLO_IR|  select.88 = s32[] select(compare.83, Arg_0.79, Arg_2.81)
HLO_IR|  compare.84 = pred[] compare(Arg_0.79, Arg_2.81), direction=EQ
HLO_IR|  Arg_1.80 = s32[] parameter(1)
HLO_IR|  Arg_3.82 = s32[] parameter(3)
HLO_IR|  compare.85 = pred[] compare(Arg_1.80, Arg_3.82), direction=LT
HLO_IR|  and.86 = pred[] and(compare.84, compare.85)
HLO_IR|  or.87 = pred[] or(compare.83, and.86)
HLO_IR|  select.89 = s32[] select(or.87, Arg_1.80, Arg_3.82)
HLO_IR|  ROOT tuple.90 = (s32[], s32[]) tuple(select.88, select.89)
HLO_IR|}
HLO_IR|
HLO_IR|argmin.91 {
HLO_IR|  Arg_0.92 = s32[5,4,4]{2,1,0} parameter(0)
HLO_IR|  iota.95 = s32[4]{0} iota(), iota_dimension=0
HLO_IR|  broadcast.96 = s32[5,4,4]{2,1,0} broadcast(iota.95), dimensions={1}
HLO_IR|  constant.94 = s32[] constant(2147483647)
HLO_IR|  constant.93 = s32[] constant(0)
HLO_IR|  reduce.97 = (s32[5,4]{1,0}, s32[5,4]{1,0}) reduce(Arg_0.92, broadcast.96, constant.94, constant.93), dimensions={1}, to_apply=region_1.78
HLO_IR|  get-tuple-element.98 = s32[5,4]{1,0} get-tuple-element(reduce.97), index=0
HLO_IR|  ROOT get-tuple-element.99 = s32[5,4]{1,0} get-tuple-element(reduce.97), index=1
HLO_IR|}
HLO_IR|
HLO_IR|region_2.126 {
HLO_IR|  Arg_0.127 = s32[] parameter(0)
HLO_IR|  Arg_1.128 = s32[] parameter(1)
HLO_IR|  ROOT add.129 = s32[] add(Arg_0.127, Arg_1.128)
HLO_IR|}
HLO_IR|
HLO_IR|region_3.131 {
HLO_IR|  Arg_0.132 = s32[] parameter(0)
HLO_IR|  Arg_1.133 = s32[] parameter(1)
HLO_IR|  ROOT add.134 = s32[] add(Arg_0.132, Arg_1.133)
HLO_IR|}
HLO_IR|
HLO_IR|region_4.157 {
HLO_IR|  Arg_0.158 = f32[] parameter(0)
HLO_IR|  Arg_1.159 = f32[] parameter(1)
HLO_IR|  ROOT add.160 = f32[] add(Arg_0.158, Arg_1.159)
HLO_IR|}
HLO_IR|
HLO_IR|region_5.162 {
HLO_IR|  Arg_0.163 = f32[] parameter(0)
HLO_IR|  Arg_2.165 = f32[] parameter(2)
HLO_IR|  compare.167 = pred[] compare(Arg_0.163, Arg_2.165), direction=LT
HLO_IR|  compare.168 = pred[] compare(Arg_0.163, Arg_0.163), direction=NE
HLO_IR|  or.169 = pred[] or(compare.167, compare.168)
HLO_IR|  select.174 = f32[] select(or.169, Arg_0.163, Arg_2.165)
HLO_IR|  compare.170 = pred[] compare(Arg_0.163, Arg_2.165), direction=EQ
HLO_IR|  Arg_1.164 = s32[] parameter(1)
HLO_IR|  Arg_3.166 = s32[] parameter(3)
HLO_IR|  compare.171 = pred[] compare(Arg_1.164, Arg_3.166), direction=LT
HLO_IR|  and.172 = pred[] and(compare.170, compare.171)
HLO_IR|  or.173 = pred[] or(or.169, and.172)
HLO_IR|  select.175 = s32[] select(or.173, Arg_1.164, Arg_3.166)
HLO_IR|  ROOT tuple.176 = (f32[], s32[]) tuple(select.174, select.175)
HLO_IR|}
HLO_IR|
HLO_IR|argmin_0.177 {
HLO_IR|  Arg_0.178 = f32[5,4,4]{2,1,0} parameter(0)
HLO_IR|  iota.181 = s32[4]{0} iota(), iota_dimension=0
HLO_IR|  broadcast.182 = s32[5,4,4]{2,1,0} broadcast(iota.181), dimensions={1}
HLO_IR|  constant.180 = f32[] constant(inf)
HLO_IR|  constant.179 = s32[] constant(0)
HLO_IR|  reduce.183 = (f32[5,4]{1,0}, s32[5,4]{1,0}) reduce(Arg_0.178, broadcast.182, constant.180, constant.179), dimensions={1}, to_apply=region_5.162
HLO_IR|  get-tuple-element.184 = f32[5,4]{1,0} get-tuple-element(reduce.183), index=0
HLO_IR|  ROOT get-tuple-element.185 = s32[5,4]{1,0} get-tuple-element(reduce.183), index=1
HLO_IR|}
HLO_IR|
HLO_IR|region_6.212 {
HLO_IR|  Arg_0.213 = s32[] parameter(0)
HLO_IR|  Arg_1.214 = s32[] parameter(1)
HLO_IR|  ROOT add.215 = s32[] add(Arg_0.213, Arg_1.214)
HLO_IR|}
HLO_IR|
HLO_IR|region_7.217 {
HLO_IR|  Arg_0.218 = s32[] parameter(0)
HLO_IR|  Arg_1.219 = s32[] parameter(1)
HLO_IR|  ROOT add.220 = s32[] add(Arg_0.218, Arg_1.219)
HLO_IR|}
HLO_IR|
HLO_IR|region_8.243 {
HLO_IR|  Arg_0.244 = f32[] parameter(0)
HLO_IR|  Arg_1.245 = f32[] parameter(1)
HLO_IR|  ROOT add.246 = f32[] add(Arg_0.244, Arg_1.245)
HLO_IR|}
HLO_IR|
HLO_IR|region_5.248 {
HLO_IR|  Arg_0.249 = f32[] parameter(0)
HLO_IR|  Arg_2.251 = f32[] parameter(2)
HLO_IR|  compare.253 = pred[] compare(Arg_0.249, Arg_2.251), direction=LT
HLO_IR|  compare.254 = pred[] compare(Arg_0.249, Arg_0.249), direction=NE
HLO_IR|  or.255 = pred[] or(compare.253, compare.254)
HLO_IR|  select.260 = f32[] select(or.255, Arg_0.249, Arg_2.251)
HLO_IR|  compare.256 = pred[] compare(Arg_0.249, Arg_2.251), direction=EQ
HLO_IR|  Arg_1.250 = s32[] parameter(1)
HLO_IR|  Arg_3.252 = s32[] parameter(3)
HLO_IR|  compare.257 = pred[] compare(Arg_1.250, Arg_3.252), direction=LT
HLO_IR|  and.258 = pred[] and(compare.256, compare.257)
HLO_IR|  or.259 = pred[] or(or.255, and.258)
HLO_IR|  select.261 = s32[] select(or.259, Arg_1.250, Arg_3.252)
HLO_IR|  ROOT tuple.262 = (f32[], s32[]) tuple(select.260, select.261)
HLO_IR|}
HLO_IR|
HLO_IR|argmin_0.263 {
HLO_IR|  Arg_0.264 = f32[5,4,4]{2,1,0} parameter(0)
HLO_IR|  iota.267 = s32[4]{0} iota(), iota_dimension=0
HLO_IR|  broadcast.268 = s32[5,4,4]{2,1,0} broadcast(iota.267), dimensions={1}
HLO_IR|  constant.266 = f32[] constant(inf)
HLO_IR|  constant.265 = s32[] constant(0)
HLO_IR|  reduce.269 = (f32[5,4]{1,0}, s32[5,4]{1,0}) reduce(Arg_0.264, broadcast.268, constant.266, constant.265), dimensions={1}, to_apply=region_5.248
HLO_IR|  get-tuple-element.270 = f32[5,4]{1,0} get-tuple-element(reduce.269), index=0
HLO_IR|  ROOT get-tuple-element.271 = s32[5,4]{1,0} get-tuple-element(reduce.269), index=1
HLO_IR|}
HLO_IR|
HLO_IR|region_9.298 {
HLO_IR|  Arg_0.299 = s32[] parameter(0)
HLO_IR|  Arg_1.300 = s32[] parameter(1)
HLO_IR|  ROOT add.301 = s32[] add(Arg_0.299, Arg_1.300)
HLO_IR|}
HLO_IR|
HLO_IR|region_10.303 {
HLO_IR|  Arg_0.304 = s32[] parameter(0)
HLO_IR|  Arg_1.305 = s32[] parameter(1)
HLO_IR|  ROOT add.306 = s32[] add(Arg_0.304, Arg_1.305)
HLO_IR|}
HLO_IR|
HLO_IR|region_11.329 {
HLO_IR|  Arg_0.330 = f32[] parameter(0)
HLO_IR|  Arg_1.331 = f32[] parameter(1)
HLO_IR|  ROOT add.332 = f32[] add(Arg_0.330, Arg_1.331)
HLO_IR|}
HLO_IR|
HLO_IR|region_5.334 {
HLO_IR|  Arg_0.335 = f32[] parameter(0)
HLO_IR|  Arg_2.337 = f32[] parameter(2)
HLO_IR|  compare.339 = pred[] compare(Arg_0.335, Arg_2.337), direction=LT
HLO_IR|  compare.340 = pred[] compare(Arg_0.335, Arg_0.335), direction=NE
HLO_IR|  or.341 = pred[] or(compare.339, compare.340)
HLO_IR|  select.346 = f32[] select(or.341, Arg_0.335, Arg_2.337)
HLO_IR|  compare.342 = pred[] compare(Arg_0.335, Arg_2.337), direction=EQ
HLO_IR|  Arg_1.336 = s32[] parameter(1)
HLO_IR|  Arg_3.338 = s32[] parameter(3)
HLO_IR|  compare.343 = pred[] compare(Arg_1.336, Arg_3.338), direction=LT
HLO_IR|  and.344 = pred[] and(compare.342, compare.343)
HLO_IR|  or.345 = pred[] or(or.341, and.344)
HLO_IR|  select.347 = s32[] select(or.345, Arg_1.336, Arg_3.338)
HLO_IR|  ROOT tuple.348 = (f32[], s32[]) tuple(select.346, select.347)
HLO_IR|}
HLO_IR|
HLO_IR|argmin_0.349 {
HLO_IR|  Arg_0.350 = f32[5,4,4]{2,1,0} parameter(0)
HLO_IR|  iota.353 = s32[4]{0} iota(), iota_dimension=0
HLO_IR|  broadcast.354 = s32[5,4,4]{2,1,0} broadcast(iota.353), dimensions={1}
HLO_IR|  constant.352 = f32[] constant(inf)
HLO_IR|  constant.351 = s32[] constant(0)
HLO_IR|  reduce.355 = (f32[5,4]{1,0}, s32[5,4]{1,0}) reduce(Arg_0.350, broadcast.354, constant.352, constant.351), dimensions={1}, to_apply=region_5.334
HLO_IR|  get-tuple-element.356 = f32[5,4]{1,0} get-tuple-element(reduce.355), index=0
HLO_IR|  ROOT get-tuple-element.357 = s32[5,4]{1,0} get-tuple-element(reduce.355), index=1
HLO_IR|}
HLO_IR|
HLO_IR|region_12.384 {
HLO_IR|  Arg_0.385 = s32[] parameter(0)
HLO_IR|  Arg_1.386 = s32[] parameter(1)
HLO_IR|  ROOT add.387 = s32[] add(Arg_0.385, Arg_1.386)
HLO_IR|}
HLO_IR|
HLO_IR|region_13.389 {
HLO_IR|  Arg_0.390 = s32[] parameter(0)
HLO_IR|  Arg_1.391 = s32[] parameter(1)
HLO_IR|  ROOT add.392 = s32[] add(Arg_0.390, Arg_1.391)
HLO_IR|}
HLO_IR|
HLO_IR|region_14.415 {
HLO_IR|  Arg_0.416 = f32[] parameter(0)
HLO_IR|  Arg_1.417 = f32[] parameter(1)
HLO_IR|  ROOT add.418 = f32[] add(Arg_0.416, Arg_1.417)
HLO_IR|}
HLO_IR|
HLO_IR|region_5.420 {
HLO_IR|  Arg_0.421 = f32[] parameter(0)
HLO_IR|  Arg_2.423 = f32[] parameter(2)
HLO_IR|  compare.425 = pred[] compare(Arg_0.421, Arg_2.423), direction=LT
HLO_IR|  compare.426 = pred[] compare(Arg_0.421, Arg_0.421), direction=NE
HLO_IR|  or.427 = pred[] or(compare.425, compare.426)
HLO_IR|  select.432 = f32[] select(or.427, Arg_0.421, Arg_2.423)
HLO_IR|  compare.428 = pred[] compare(Arg_0.421, Arg_2.423), direction=EQ
HLO_IR|  Arg_1.422 = s32[] parameter(1)
HLO_IR|  Arg_3.424 = s32[] parameter(3)
HLO_IR|  compare.429 = pred[] compare(Arg_1.422, Arg_3.424), direction=LT
HLO_IR|  and.430 = pred[] and(compare.428, compare.429)
HLO_IR|  or.431 = pred[] or(or.427, and.430)
HLO_IR|  select.433 = s32[] select(or.431, Arg_1.422, Arg_3.424)
HLO_IR|  ROOT tuple.434 = (f32[], s32[]) tuple(select.432, select.433)
HLO_IR|}
HLO_IR|
HLO_IR|argmin_0.435 {
HLO_IR|  Arg_0.436 = f32[5,4,4]{2,1,0} parameter(0)
HLO_IR|  iota.439 = s32[4]{0} iota(), iota_dimension=0
HLO_IR|  broadcast.440 = s32[5,4,4]{2,1,0} broadcast(iota.439), dimensions={1}
HLO_IR|  constant.438 = f32[] constant(inf)
HLO_IR|  constant.437 = s32[] constant(0)
HLO_IR|  reduce.441 = (f32[5,4]{1,0}, s32[5,4]{1,0}) reduce(Arg_0.436, broadcast.440, constant.438, constant.437), dimensions={1}, to_apply=region_5.420
HLO_IR|  get-tuple-element.442 = f32[5,4]{1,0} get-tuple-element(reduce.441), index=0
HLO_IR|  ROOT get-tuple-element.443 = s32[5,4]{1,0} get-tuple-element(reduce.441), index=1
HLO_IR|}
HLO_IR|
HLO_IR|region_15.470 {
HLO_IR|  Arg_0.471 = s32[] parameter(0)
HLO_IR|  Arg_1.472 = s32[] parameter(1)
HLO_IR|  ROOT add.473 = s32[] add(Arg_0.471, Arg_1.472)
HLO_IR|}
HLO_IR|
HLO_IR|region_16.475 {
HLO_IR|  Arg_0.476 = s32[] parameter(0)
HLO_IR|  Arg_1.477 = s32[] parameter(1)
HLO_IR|  ROOT add.478 = s32[] add(Arg_0.476, Arg_1.477)
HLO_IR|}
HLO_IR|
HLO_IR|region_17.501 {
HLO_IR|  Arg_0.502 = f32[] parameter(0)
HLO_IR|  Arg_1.503 = f32[] parameter(1)
HLO_IR|  ROOT add.504 = f32[] add(Arg_0.502, Arg_1.503)
HLO_IR|}
HLO_IR|
HLO_IR|region_18.506 {
HLO_IR|  Arg_0.507 = f32[] parameter(0)
HLO_IR|  Arg_1.508 = f32[] parameter(1)
HLO_IR|  ROOT minimum.509 = f32[] minimum(Arg_0.507, Arg_1.508)
HLO_IR|}
HLO_IR|
HLO_IR|region_19.511 {
HLO_IR|  Arg_0.512 = f32[] parameter(0)
HLO_IR|  Arg_1.513 = f32[] parameter(1)
HLO_IR|  ROOT add.514 = f32[] add(Arg_0.512, Arg_1.513)
HLO_IR|}
HLO_IR|
HLO_IR|region_20.516 {
HLO_IR|  Arg_0.517 = f32[] parameter(0)
HLO_IR|  Arg_2.519 = f32[] parameter(2)
HLO_IR|  compare.521 = pred[] compare(Arg_0.517, Arg_2.519), direction=LT
HLO_IR|  compare.522 = pred[] compare(Arg_0.517, Arg_0.517), direction=NE
HLO_IR|  or.523 = pred[] or(compare.521, compare.522)
HLO_IR|  select.528 = f32[] select(or.523, Arg_0.517, Arg_2.519)
HLO_IR|  compare.524 = pred[] compare(Arg_0.517, Arg_2.519), direction=EQ
HLO_IR|  Arg_1.518 = s32[] parameter(1)
HLO_IR|  Arg_3.520 = s32[] parameter(3)
HLO_IR|  compare.525 = pred[] compare(Arg_1.518, Arg_3.520), direction=LT
HLO_IR|  and.526 = pred[] and(compare.524, compare.525)
HLO_IR|  or.527 = pred[] or(or.523, and.526)
HLO_IR|  select.529 = s32[] select(or.527, Arg_1.518, Arg_3.520)
HLO_IR|  ROOT tuple.530 = (f32[], s32[]) tuple(select.528, select.529)
HLO_IR|}
HLO_IR|
HLO_IR|argmin_1.531 {
HLO_IR|  Arg_0.532 = f32[5]{0} parameter(0)
HLO_IR|  iota.535 = s32[5]{0} iota(), iota_dimension=0
HLO_IR|  constant.534 = f32[] constant(inf)
HLO_IR|  constant.533 = s32[] constant(0)
HLO_IR|  reduce.536 = (f32[], s32[]) reduce(Arg_0.532, iota.535, constant.534, constant.533), dimensions={0}, to_apply=region_20.516
HLO_IR|  get-tuple-element.537 = f32[] get-tuple-element(reduce.536), index=0
HLO_IR|  ROOT get-tuple-element.538 = s32[] get-tuple-element(reduce.536), index=1
HLO_IR|}
HLO_IR|
HLO_IR|region_21.545 {
HLO_IR|  Arg_0.546 = f32[] parameter(0)
HLO_IR|  Arg_1.547 = f32[] parameter(1)
HLO_IR|  ROOT compare.548 = pred[] compare(Arg_0.546, Arg_1.547), direction=LT, type=TOTALORDER
HLO_IR|}
HLO_IR|
HLO_IR|sort.549 {
HLO_IR|  Arg_0.550 = f32[4,2]{1,0} parameter(0)
HLO_IR|  ROOT sort.551 = f32[4,2]{1,0} sort(Arg_0.550), dimensions={0}, is_stable=true, to_apply=region_21.545
HLO_IR|}
HLO_IR|
HLO_IR|ENTRY main.553 {
HLO_IR|  Arg_0.1 = s32[4,2]{1,0} parameter(0)
HLO_IR|  reshape.461 = s32[1,1,4,2]{3,2,1,0} reshape(Arg_0.1)
HLO_IR|  broadcast.462 = s32[1,1,4,2]{3,2,1,0} broadcast(reshape.461), dimensions={0,1,2,3}
HLO_IR|  reshape.463 = s32[4,2]{1,0} reshape(broadcast.462)
HLO_IR|  broadcast.464 = s32[5,4,4,2]{3,2,1,0} broadcast(reshape.463), dimensions={2,3}
HLO_IR|  reshape.403 = s32[1,4,2]{2,1,0} reshape(Arg_0.1)
HLO_IR|  convert.405 = f32[1,4,2]{2,1,0} convert(reshape.403)
HLO_IR|  reshape.406 = f32[1,1,4,2]{3,2,1,0} reshape(convert.405)
HLO_IR|  broadcast.407 = f32[1,1,4,2]{3,2,1,0} broadcast(reshape.406), dimensions={0,1,2,3}
HLO_IR|  reshape.408 = f32[4,2]{1,0} reshape(broadcast.407)
HLO_IR|  broadcast.409 = f32[5,4,4,2]{3,2,1,0} broadcast(reshape.408), dimensions={2,3}
HLO_IR|  reshape.375 = s32[1,1,4,2]{3,2,1,0} reshape(Arg_0.1)
HLO_IR|  broadcast.376 = s32[1,1,4,2]{3,2,1,0} broadcast(reshape.375), dimensions={0,1,2,3}
HLO_IR|  reshape.377 = s32[4,2]{1,0} reshape(broadcast.376)
HLO_IR|  broadcast.378 = s32[5,4,4,2]{3,2,1,0} broadcast(reshape.377), dimensions={2,3}
HLO_IR|  reshape.317 = s32[1,4,2]{2,1,0} reshape(Arg_0.1)
HLO_IR|  convert.319 = f32[1,4,2]{2,1,0} convert(reshape.317)
HLO_IR|  reshape.320 = f32[1,1,4,2]{3,2,1,0} reshape(convert.319)
HLO_IR|  broadcast.321 = f32[1,1,4,2]{3,2,1,0} broadcast(reshape.320), dimensions={0,1,2,3}
HLO_IR|  reshape.322 = f32[4,2]{1,0} reshape(broadcast.321)
HLO_IR|  broadcast.323 = f32[5,4,4,2]{3,2,1,0} broadcast(reshape.322), dimensions={2,3}
HLO_IR|  reshape.289 = s32[1,1,4,2]{3,2,1,0} reshape(Arg_0.1)
HLO_IR|  broadcast.290 = s32[1,1,4,2]{3,2,1,0} broadcast(reshape.289), dimensions={0,1,2,3}
HLO_IR|  reshape.291 = s32[4,2]{1,0} reshape(broadcast.290)
HLO_IR|  broadcast.292 = s32[5,4,4,2]{3,2,1,0} broadcast(reshape.291), dimensions={2,3}
HLO_IR|  reshape.231 = s32[1,4,2]{2,1,0} reshape(Arg_0.1)
HLO_IR|  convert.233 = f32[1,4,2]{2,1,0} convert(reshape.231)
HLO_IR|  reshape.234 = f32[1,1,4,2]{3,2,1,0} reshape(convert.233)
HLO_IR|  broadcast.235 = f32[1,1,4,2]{3,2,1,0} broadcast(reshape.234), dimensions={0,1,2,3}
HLO_IR|  reshape.236 = f32[4,2]{1,0} reshape(broadcast.235)
HLO_IR|  broadcast.237 = f32[5,4,4,2]{3,2,1,0} broadcast(reshape.236), dimensions={2,3}
HLO_IR|  reshape.203 = s32[1,1,4,2]{3,2,1,0} reshape(Arg_0.1)
HLO_IR|  broadcast.204 = s32[1,1,4,2]{3,2,1,0} broadcast(reshape.203), dimensions={0,1,2,3}
HLO_IR|  reshape.205 = s32[4,2]{1,0} reshape(broadcast.204)
HLO_IR|  broadcast.206 = s32[5,4,4,2]{3,2,1,0} broadcast(reshape.205), dimensions={2,3}
HLO_IR|  reshape.145 = s32[1,4,2]{2,1,0} reshape(Arg_0.1)
HLO_IR|  convert.147 = f32[1,4,2]{2,1,0} convert(reshape.145)
HLO_IR|  reshape.148 = f32[1,1,4,2]{3,2,1,0} reshape(convert.147)
HLO_IR|  broadcast.149 = f32[1,1,4,2]{3,2,1,0} broadcast(reshape.148), dimensions={0,1,2,3}
HLO_IR|  reshape.150 = f32[4,2]{1,0} reshape(broadcast.149)
HLO_IR|  broadcast.151 = f32[5,4,4,2]{3,2,1,0} broadcast(reshape.150), dimensions={2,3}
HLO_IR|  reshape.117 = s32[1,1,4,2]{3,2,1,0} reshape(Arg_0.1)
HLO_IR|  broadcast.118 = s32[1,1,4,2]{3,2,1,0} broadcast(reshape.117), dimensions={0,1,2,3}
HLO_IR|  reshape.119 = s32[4,2]{1,0} reshape(broadcast.118)
HLO_IR|  broadcast.120 = s32[5,4,4,2]{3,2,1,0} broadcast(reshape.119), dimensions={2,3}
HLO_IR|  reshape.64 = s32[1,1,4,2]{3,2,1,0} reshape(Arg_0.1)
HLO_IR|  broadcast.65 = s32[1,1,4,2]{3,2,1,0} broadcast(reshape.64), dimensions={0,1,2,3}
HLO_IR|  reshape.66 = s32[4,2]{1,0} reshape(broadcast.65)
HLO_IR|  broadcast.67 = s32[5,4,4,2]{3,2,1,0} broadcast(reshape.66), dimensions={2,3}
HLO_IR|  constant.12 = s32[5,4]{1,0} constant({ { 0, 1, 3, 0 }, { 3, 1, 0, 1 }, { 1, 3, 2, 0 }, { 1, 3, 0, 3 }, { 2, 1, 1, 1 } })
HLO_IR|  call.26 = (s32[4]{0}, s32[4]{0}, s32[4]{0}, s32[4]{0}, s32[4]{0}) call(constant.12), to_apply=_unstack.13
HLO_IR|  get-tuple-element.27 = s32[4]{0} get-tuple-element(call.26), index=0
HLO_IR|  constant.6 = s32[] constant(0)
HLO_IR|  broadcast.7 = s32[4]{0} broadcast(constant.6), dimensions={}
HLO_IR|  compare.32 = pred[4]{0} compare(get-tuple-element.27, broadcast.7), direction=LT
HLO_IR|  constant.4 = s32[] constant(4)
HLO_IR|  broadcast.5 = s32[4]{0} broadcast(constant.4), dimensions={}
HLO_IR|  add.33 = s32[4]{0} add(get-tuple-element.27, broadcast.5)
HLO_IR|  select.34 = s32[4]{0} select(compare.32, add.33, get-tuple-element.27)
HLO_IR|  reshape.35 = s32[4,1]{1,0} reshape(select.34)
HLO_IR|  gather.36 = s32[4,2]{1,0} gather(Arg_0.1, reshape.35), offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,2}
HLO_IR|  reshape.57 = s32[1,4,2]{2,1,0} reshape(gather.36)
HLO_IR|  get-tuple-element.28 = s32[4]{0} get-tuple-element(call.26), index=1
HLO_IR|  compare.37 = pred[4]{0} compare(get-tuple-element.28, broadcast.7), direction=LT
HLO_IR|  add.38 = s32[4]{0} add(get-tuple-element.28, broadcast.5)
HLO_IR|  select.39 = s32[4]{0} select(compare.37, add.38, get-tuple-element.28)
HLO_IR|  reshape.40 = s32[4,1]{1,0} reshape(select.39)
HLO_IR|  gather.41 = s32[4,2]{1,0} gather(Arg_0.1, reshape.40), offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,2}
HLO_IR|  reshape.58 = s32[1,4,2]{2,1,0} reshape(gather.41)
HLO_IR|  get-tuple-element.29 = s32[4]{0} get-tuple-element(call.26), index=2
HLO_IR|  compare.42 = pred[4]{0} compare(get-tuple-element.29, broadcast.7), direction=LT
HLO_IR|  add.43 = s32[4]{0} add(get-tuple-element.29, broadcast.5)
HLO_IR|  select.44 = s32[4]{0} select(compare.42, add.43, get-tuple-element.29)
HLO_IR|  reshape.45 = s32[4,1]{1,0} reshape(select.44)
HLO_IR|  gather.46 = s32[4,2]{1,0} gather(Arg_0.1, reshape.45), offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,2}
HLO_IR|  reshape.59 = s32[1,4,2]{2,1,0} reshape(gather.46)
HLO_IR|  get-tuple-element.30 = s32[4]{0} get-tuple-element(call.26), index=3
HLO_IR|  compare.47 = pred[4]{0} compare(get-tuple-element.30, broadcast.7), direction=LT
HLO_IR|  add.48 = s32[4]{0} add(get-tuple-element.30, broadcast.5)
HLO_IR|  select.49 = s32[4]{0} select(compare.47, add.48, get-tuple-element.30)
HLO_IR|  reshape.50 = s32[4,1]{1,0} reshape(select.49)
HLO_IR|  gather.51 = s32[4,2]{1,0} gather(Arg_0.1, reshape.50), offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,2}
HLO_IR|  reshape.60 = s32[1,4,2]{2,1,0} reshape(gather.51)
HLO_IR|  get-tuple-element.31 = s32[4]{0} get-tuple-element(call.26), index=4
HLO_IR|  compare.52 = pred[4]{0} compare(get-tuple-element.31, broadcast.7), direction=LT
HLO_IR|  add.53 = s32[4]{0} add(get-tuple-element.31, broadcast.5)
HLO_IR|  select.54 = s32[4]{0} select(compare.52, add.53, get-tuple-element.31)
HLO_IR|  reshape.55 = s32[4,1]{1,0} reshape(select.54)
HLO_IR|  gather.56 = s32[4,2]{1,0} gather(Arg_0.1, reshape.55), offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,2}
HLO_IR|  reshape.61 = s32[1,4,2]{2,1,0} reshape(gather.56)
HLO_IR|  concatenate.62 = s32[5,4,2]{2,1,0} concatenate(reshape.57, reshape.58, reshape.59, reshape.60, reshape.61), dimensions={0}
HLO_IR|  reshape.63 = s32[5,4,1,2]{3,2,1,0} reshape(concatenate.62)
HLO_IR|  broadcast.68 = s32[5,4,1,2]{3,2,1,0} broadcast(reshape.63), dimensions={0,1,2,3}
HLO_IR|  reshape.69 = s32[5,4,2]{2,1,0} reshape(broadcast.68)
HLO_IR|  broadcast.70 = s32[5,4,4,2]{3,2,1,0} broadcast(reshape.69), dimensions={0,1,3}
HLO_IR|  subtract.71 = s32[5,4,4,2]{3,2,1,0} subtract(broadcast.67, broadcast.70)
HLO_IR|  multiply.72 = s32[5,4,4,2]{3,2,1,0} multiply(subtract.71, subtract.71)
HLO_IR|  constant.11 = s32[] constant(0)
HLO_IR|  reduce.77 = s32[5,4,4]{2,1,0} reduce(multiply.72, constant.11), dimensions={3}, to_apply=region_0.73
HLO_IR|  call.100 = s32[5,4]{1,0} call(reduce.77), to_apply=argmin.91
HLO_IR|  reshape.101 = s32[5,1,1,1,4]{4,3,2,1,0} reshape(call.100)
HLO_IR|  broadcast.102 = s32[5,1,1,1,4]{4,3,2,1,0} broadcast(reshape.101), dimensions={0,1,2,3,4}
HLO_IR|  reshape.103 = s32[5,1,1,4]{3,2,1,0} reshape(broadcast.102)
HLO_IR|  broadcast.104 = s32[5,4,1,1,4]{4,3,2,1,0} broadcast(reshape.103), dimensions={0,2,3,4}
HLO_IR|  reshape.105 = s32[5,4,4]{2,1,0} reshape(broadcast.104)
HLO_IR|  transpose.107 = s32[5,4,4]{1,2,0} transpose(reshape.105), dimensions={0,2,1}
HLO_IR|  iota.106 = s32[4]{0} iota(), iota_dimension=0
HLO_IR|  reshape.108 = s32[1,1,4]{2,1,0} reshape(iota.106)
HLO_IR|  broadcast.109 = s32[1,1,4]{2,1,0} broadcast(reshape.108), dimensions={0,1,2}
HLO_IR|  reshape.110 = s32[4]{0} reshape(broadcast.109)
HLO_IR|  broadcast.111 = s32[5,4,4]{2,1,0} broadcast(reshape.110), dimensions={2}
HLO_IR|  subtract.112 = s32[5,4,4]{1,2,0} subtract(transpose.107, broadcast.111)
HLO_IR|  transpose.113 = s32[5,4,4]{2,1,0} transpose(subtract.112), dimensions={0,2,1}
HLO_IR|  constant.2 = s32[] constant(0)
HLO_IR|  broadcast.3 = s32[5,4,4]{2,1,0} broadcast(constant.2), dimensions={}
HLO_IR|  compare.114 = pred[5,4,4]{2,1,0} compare(transpose.113, broadcast.3), direction=EQ
HLO_IR|  reshape.115 = pred[5,4,4,1]{3,2,1,0} reshape(compare.114)
HLO_IR|  convert.116 = s32[5,4,4,1]{3,2,1,0} convert(reshape.115)
HLO_IR|  broadcast.121 = s32[5,4,4,1]{3,2,1,0} broadcast(convert.116), dimensions={0,1,2,3}
HLO_IR|  reshape.122 = s32[5,4,4]{2,1,0} reshape(broadcast.121)
HLO_IR|  broadcast.123 = s32[5,4,4,2]{3,2,1,0} broadcast(reshape.122), dimensions={0,1,2}
HLO_IR|  multiply.124 = s32[5,4,4,2]{3,2,1,0} multiply(broadcast.120, broadcast.123)
HLO_IR|  reduce.135 = s32[5,4,2]{2,1,0} reduce(multiply.124, constant.11), dimensions={2}, to_apply=region_3.131
HLO_IR|  transpose.136 = s32[5,2,4]{1,2,0} transpose(reduce.135), dimensions={0,2,1}
HLO_IR|  convert.137 = f32[5,2,4]{1,2,0} convert(transpose.136)
HLO_IR|  convert.125 = s32[5,4,4]{2,1,0} convert(compare.114)
HLO_IR|  reduce.130 = s32[5,4]{1,0} reduce(convert.125, constant.11), dimensions={2}, to_apply=region_2.126
HLO_IR|  convert.138 = f32[5,4]{1,0} convert(reduce.130)
HLO_IR|  reshape.139 = f32[5,1,4]{2,1,0} reshape(convert.138)
HLO_IR|  broadcast.140 = f32[5,1,4]{2,1,0} broadcast(reshape.139), dimensions={0,1,2}
HLO_IR|  reshape.141 = f32[5,4]{1,0} reshape(broadcast.140)
HLO_IR|  broadcast.142 = f32[5,2,4]{2,1,0} broadcast(reshape.141), dimensions={0,2}
HLO_IR|  divide.143 = f32[5,2,4]{1,2,0} divide(convert.137, broadcast.142)
HLO_IR|  transpose.144 = f32[5,4,2]{2,1,0} transpose(divide.143), dimensions={0,2,1}
HLO_IR|  reshape.146 = f32[5,4,1,2]{3,2,1,0} reshape(transpose.144)
HLO_IR|  broadcast.152 = f32[5,4,1,2]{3,2,1,0} broadcast(reshape.146), dimensions={0,1,2,3}
HLO_IR|  reshape.153 = f32[5,4,2]{2,1,0} reshape(broadcast.152)
HLO_IR|  broadcast.154 = f32[5,4,4,2]{3,2,1,0} broadcast(reshape.153), dimensions={0,1,3}
HLO_IR|  subtract.155 = f32[5,4,4,2]{3,2,1,0} subtract(broadcast.151, broadcast.154)
HLO_IR|  multiply.156 = f32[5,4,4,2]{3,2,1,0} multiply(subtract.155, subtract.155)
HLO_IR|  constant.10 = f32[] constant(0)
HLO_IR|  reduce.161 = f32[5,4,4]{2,1,0} reduce(multiply.156, constant.10), dimensions={3}, to_apply=region_4.157
HLO_IR|  call.186 = s32[5,4]{1,0} call(reduce.161), to_apply=argmin_0.177
HLO_IR|  reshape.187 = s32[5,1,1,1,4]{4,3,2,1,0} reshape(call.186)
HLO_IR|  broadcast.188 = s32[5,1,1,1,4]{4,3,2,1,0} broadcast(reshape.187), dimensions={0,1,2,3,4}
HLO_IR|  reshape.189 = s32[5,1,1,4]{3,2,1,0} reshape(broadcast.188)
HLO_IR|  broadcast.190 = s32[5,4,1,1,4]{4,3,2,1,0} broadcast(reshape.189), dimensions={0,2,3,4}
HLO_IR|  reshape.191 = s32[5,4,4]{2,1,0} reshape(broadcast.190)
HLO_IR|  transpose.193 = s32[5,4,4]{1,2,0} transpose(reshape.191), dimensions={0,2,1}
HLO_IR|  iota.192 = s32[4]{0} iota(), iota_dimension=0
HLO_IR|  reshape.194 = s32[1,1,4]{2,1,0} reshape(iota.192)
HLO_IR|  broadcast.195 = s32[1,1,4]{2,1,0} broadcast(reshape.194), dimensions={0,1,2}
HLO_IR|  reshape.196 = s32[4]{0} reshape(broadcast.195)
HLO_IR|  broadcast.197 = s32[5,4,4]{2,1,0} broadcast(reshape.196), dimensions={2}
HLO_IR|  subtract.198 = s32[5,4,4]{1,2,0} subtract(transpose.193, broadcast.197)
HLO_IR|  transpose.199 = s32[5,4,4]{2,1,0} transpose(subtract.198), dimensions={0,2,1}
HLO_IR|  compare.200 = pred[5,4,4]{2,1,0} compare(transpose.199, broadcast.3), direction=EQ
HLO_IR|  reshape.201 = pred[5,4,4,1]{3,2,1,0} reshape(compare.200)
HLO_IR|  convert.202 = s32[5,4,4,1]{3,2,1,0} convert(reshape.201)
HLO_IR|  broadcast.207 = s32[5,4,4,1]{3,2,1,0} broadcast(convert.202), dimensions={0,1,2,3}
HLO_IR|  reshape.208 = s32[5,4,4]{2,1,0} reshape(broadcast.207)
HLO_IR|  broadcast.209 = s32[5,4,4,2]{3,2,1,0} broadcast(reshape.208), dimensions={0,1,2}
HLO_IR|  multiply.210 = s32[5,4,4,2]{3,2,1,0} multiply(broadcast.206, broadcast.209)
HLO_IR|  reduce.221 = s32[5,4,2]{2,1,0} reduce(multiply.210, constant.11), dimensions={2}, to_apply=region_7.217
HLO_IR|  transpose.222 = s32[5,2,4]{1,2,0} transpose(reduce.221), dimensions={0,2,1}
HLO_IR|  convert.223 = f32[5,2,4]{1,2,0} convert(transpose.222)
HLO_IR|  convert.211 = s32[5,4,4]{2,1,0} convert(compare.200)
HLO_IR|  reduce.216 = s32[5,4]{1,0} reduce(convert.211, constant.11), dimensions={2}, to_apply=region_6.212
HLO_IR|  convert.224 = f32[5,4]{1,0} convert(reduce.216)
HLO_IR|  reshape.225 = f32[5,1,4]{2,1,0} reshape(convert.224)
HLO_IR|  broadcast.226 = f32[5,1,4]{2,1,0} broadcast(reshape.225), dimensions={0,1,2}
HLO_IR|  reshape.227 = f32[5,4]{1,0} reshape(broadcast.226)
HLO_IR|  broadcast.228 = f32[5,2,4]{2,1,0} broadcast(reshape.227), dimensions={0,2}
HLO_IR|  divide.229 = f32[5,2,4]{1,2,0} divide(convert.223, broadcast.228)
HLO_IR|  transpose.230 = f32[5,4,2]{2,1,0} transpose(divide.229), dimensions={0,2,1}
HLO_IR|  reshape.232 = f32[5,4,1,2]{3,2,1,0} reshape(transpose.230)
HLO_IR|  broadcast.238 = f32[5,4,1,2]{3,2,1,0} broadcast(reshape.232), dimensions={0,1,2,3}
HLO_IR|  reshape.239 = f32[5,4,2]{2,1,0} reshape(broadcast.238)
HLO_IR|  broadcast.240 = f32[5,4,4,2]{3,2,1,0} broadcast(reshape.239), dimensions={0,1,3}
HLO_IR|  subtract.241 = f32[5,4,4,2]{3,2,1,0} subtract(broadcast.237, broadcast.240)
HLO_IR|  multiply.242 = f32[5,4,4,2]{3,2,1,0} multiply(subtract.241, subtract.241)
HLO_IR|  reduce.247 = f32[5,4,4]{2,1,0} reduce(multiply.242, constant.10), dimensions={3}, to_apply=region_8.243
HLO_IR|  call.272 = s32[5,4]{1,0} call(reduce.247), to_apply=argmin_0.263
HLO_IR|  reshape.273 = s32[5,1,1,1,4]{4,3,2,1,0} reshape(call.272)
HLO_IR|  broadcast.274 = s32[5,1,1,1,4]{4,3,2,1,0} broadcast(reshape.273), dimensions={0,1,2,3,4}
HLO_IR|  reshape.275 = s32[5,1,1,4]{3,2,1,0} reshape(broadcast.274)
HLO_IR|  broadcast.276 = s32[5,4,1,1,4]{4,3,2,1,0} broadcast(reshape.275), dimensions={0,2,3,4}
HLO_IR|  reshape.277 = s32[5,4,4]{2,1,0} reshape(broadcast.276)
HLO_IR|  transpose.279 = s32[5,4,4]{1,2,0} transpose(reshape.277), dimensions={0,2,1}
HLO_IR|  iota.278 = s32[4]{0} iota(), iota_dimension=0
HLO_IR|  reshape.280 = s32[1,1,4]{2,1,0} reshape(iota.278)
HLO_IR|  broadcast.281 = s32[1,1,4]{2,1,0} broadcast(reshape.280), dimensions={0,1,2}
HLO_IR|  reshape.282 = s32[4]{0} reshape(broadcast.281)
HLO_IR|  broadcast.283 = s32[5,4,4]{2,1,0} broadcast(reshape.282), dimensions={2}
HLO_IR|  subtract.284 = s32[5,4,4]{1,2,0} subtract(transpose.279, broadcast.283)
HLO_IR|  transpose.285 = s32[5,4,4]{2,1,0} transpose(subtract.284), dimensions={0,2,1}
HLO_IR|  compare.286 = pred[5,4,4]{2,1,0} compare(transpose.285, broadcast.3), direction=EQ
HLO_IR|  reshape.287 = pred[5,4,4,1]{3,2,1,0} reshape(compare.286)
HLO_IR|  convert.288 = s32[5,4,4,1]{3,2,1,0} convert(reshape.287)
HLO_IR|  broadcast.293 = s32[5,4,4,1]{3,2,1,0} broadcast(convert.288), dimensions={0,1,2,3}
HLO_IR|  reshape.294 = s32[5,4,4]{2,1,0} reshape(broadcast.293)
HLO_IR|  broadcast.295 = s32[5,4,4,2]{3,2,1,0} broadcast(reshape.294), dimensions={0,1,2}
HLO_IR|  multiply.296 = s32[5,4,4,2]{3,2,1,0} multiply(broadcast.292, broadcast.295)
HLO_IR|  reduce.307 = s32[5,4,2]{2,1,0} reduce(multiply.296, constant.11), dimensions={2}, to_apply=region_10.303
HLO_IR|  transpose.308 = s32[5,2,4]{1,2,0} transpose(reduce.307), dimensions={0,2,1}
HLO_IR|  convert.309 = f32[5,2,4]{1,2,0} convert(transpose.308)
HLO_IR|  convert.297 = s32[5,4,4]{2,1,0} convert(compare.286)
HLO_IR|  reduce.302 = s32[5,4]{1,0} reduce(convert.297, constant.11), dimensions={2}, to_apply=region_9.298
HLO_IR|  convert.310 = f32[5,4]{1,0} convert(reduce.302)
HLO_IR|  reshape.311 = f32[5,1,4]{2,1,0} reshape(convert.310)
HLO_IR|  broadcast.312 = f32[5,1,4]{2,1,0} broadcast(reshape.311), dimensions={0,1,2}
HLO_IR|  reshape.313 = f32[5,4]{1,0} reshape(broadcast.312)
HLO_IR|  broadcast.314 = f32[5,2,4]{2,1,0} broadcast(reshape.313), dimensions={0,2}
HLO_IR|  divide.315 = f32[5,2,4]{1,2,0} divide(convert.309, broadcast.314)
HLO_IR|  transpose.316 = f32[5,4,2]{2,1,0} transpose(divide.315), dimensions={0,2,1}
HLO_IR|  reshape.318 = f32[5,4,1,2]{3,2,1,0} reshape(transpose.316)
HLO_IR|  broadcast.324 = f32[5,4,1,2]{3,2,1,0} broadcast(reshape.318), dimensions={0,1,2,3}
HLO_IR|  reshape.325 = f32[5,4,2]{2,1,0} reshape(broadcast.324)
HLO_IR|  broadcast.326 = f32[5,4,4,2]{3,2,1,0} broadcast(reshape.325), dimensions={0,1,3}
HLO_IR|  subtract.327 = f32[5,4,4,2]{3,2,1,0} subtract(broadcast.323, broadcast.326)
HLO_IR|  multiply.328 = f32[5,4,4,2]{3,2,1,0} multiply(subtract.327, subtract.327)
HLO_IR|  reduce.333 = f32[5,4,4]{2,1,0} reduce(multiply.328, constant.10), dimensions={3}, to_apply=region_11.329
HLO_IR|  call.358 = s32[5,4]{1,0} call(reduce.333), to_apply=argmin_0.349
HLO_IR|  reshape.359 = s32[5,1,1,1,4]{4,3,2,1,0} reshape(call.358)
HLO_IR|  broadcast.360 = s32[5,1,1,1,4]{4,3,2,1,0} broadcast(reshape.359), dimensions={0,1,2,3,4}
HLO_IR|  reshape.361 = s32[5,1,1,4]{3,2,1,0} reshape(broadcast.360)
HLO_IR|  broadcast.362 = s32[5,4,1,1,4]{4,3,2,1,0} broadcast(reshape.361), dimensions={0,2,3,4}
HLO_IR|  reshape.363 = s32[5,4,4]{2,1,0} reshape(broadcast.362)
HLO_IR|  transpose.365 = s32[5,4,4]{1,2,0} transpose(reshape.363), dimensions={0,2,1}
HLO_IR|  iota.364 = s32[4]{0} iota(), iota_dimension=0
HLO_IR|  reshape.366 = s32[1,1,4]{2,1,0} reshape(iota.364)
HLO_IR|  broadcast.367 = s32[1,1,4]{2,1,0} broadcast(reshape.366), dimensions={0,1,2}
HLO_IR|  reshape.368 = s32[4]{0} reshape(broadcast.367)
HLO_IR|  broadcast.369 = s32[5,4,4]{2,1,0} broadcast(reshape.368), dimensions={2}
HLO_IR|  subtract.370 = s32[5,4,4]{1,2,0} subtract(transpose.365, broadcast.369)
HLO_IR|  transpose.371 = s32[5,4,4]{2,1,0} transpose(subtract.370), dimensions={0,2,1}
HLO_IR|  compare.372 = pred[5,4,4]{2,1,0} compare(transpose.371, broadcast.3), direction=EQ
HLO_IR|  reshape.373 = pred[5,4,4,1]{3,2,1,0} reshape(compare.372)
HLO_IR|  convert.374 = s32[5,4,4,1]{3,2,1,0} convert(reshape.373)
HLO_IR|  broadcast.379 = s32[5,4,4,1]{3,2,1,0} broadcast(convert.374), dimensions={0,1,2,3}
HLO_IR|  reshape.380 = s32[5,4,4]{2,1,0} reshape(broadcast.379)
HLO_IR|  broadcast.381 = s32[5,4,4,2]{3,2,1,0} broadcast(reshape.380), dimensions={0,1,2}
HLO_IR|  multiply.382 = s32[5,4,4,2]{3,2,1,0} multiply(broadcast.378, broadcast.381)
HLO_IR|  reduce.393 = s32[5,4,2]{2,1,0} reduce(multiply.382, constant.11), dimensions={2}, to_apply=region_13.389
HLO_IR|  transpose.394 = s32[5,2,4]{1,2,0} transpose(reduce.393), dimensions={0,2,1}
HLO_IR|  convert.395 = f32[5,2,4]{1,2,0} convert(transpose.394)
HLO_IR|  convert.383 = s32[5,4,4]{2,1,0} convert(compare.372)
HLO_IR|  reduce.388 = s32[5,4]{1,0} reduce(convert.383, constant.11), dimensions={2}, to_apply=region_12.384
HLO_IR|  convert.396 = f32[5,4]{1,0} convert(reduce.388)
HLO_IR|  reshape.397 = f32[5,1,4]{2,1,0} reshape(convert.396)
HLO_IR|  broadcast.398 = f32[5,1,4]{2,1,0} broadcast(reshape.397), dimensions={0,1,2}
HLO_IR|  reshape.399 = f32[5,4]{1,0} reshape(broadcast.398)
HLO_IR|  broadcast.400 = f32[5,2,4]{2,1,0} broadcast(reshape.399), dimensions={0,2}
HLO_IR|  divide.401 = f32[5,2,4]{1,2,0} divide(convert.395, broadcast.400)
HLO_IR|  transpose.402 = f32[5,4,2]{2,1,0} transpose(divide.401), dimensions={0,2,1}
HLO_IR|  reshape.404 = f32[5,4,1,2]{3,2,1,0} reshape(transpose.402)
HLO_IR|  broadcast.410 = f32[5,4,1,2]{3,2,1,0} broadcast(reshape.404), dimensions={0,1,2,3}
HLO_IR|  reshape.411 = f32[5,4,2]{2,1,0} reshape(broadcast.410)
HLO_IR|  broadcast.412 = f32[5,4,4,2]{3,2,1,0} broadcast(reshape.411), dimensions={0,1,3}
HLO_IR|  subtract.413 = f32[5,4,4,2]{3,2,1,0} subtract(broadcast.409, broadcast.412)
HLO_IR|  multiply.414 = f32[5,4,4,2]{3,2,1,0} multiply(subtract.413, subtract.413)
HLO_IR|  reduce.419 = f32[5,4,4]{2,1,0} reduce(multiply.414, constant.10), dimensions={3}, to_apply=region_14.415
HLO_IR|  call.444 = s32[5,4]{1,0} call(reduce.419), to_apply=argmin_0.435
HLO_IR|  reshape.445 = s32[5,1,1,1,4]{4,3,2,1,0} reshape(call.444)
HLO_IR|  broadcast.446 = s32[5,1,1,1,4]{4,3,2,1,0} broadcast(reshape.445), dimensions={0,1,2,3,4}
HLO_IR|  reshape.447 = s32[5,1,1,4]{3,2,1,0} reshape(broadcast.446)
HLO_IR|  broadcast.448 = s32[5,4,1,1,4]{4,3,2,1,0} broadcast(reshape.447), dimensions={0,2,3,4}
HLO_IR|  reshape.449 = s32[5,4,4]{2,1,0} reshape(broadcast.448)
HLO_IR|  transpose.451 = s32[5,4,4]{1,2,0} transpose(reshape.449), dimensions={0,2,1}
HLO_IR|  iota.450 = s32[4]{0} iota(), iota_dimension=0
HLO_IR|  reshape.452 = s32[1,1,4]{2,1,0} reshape(iota.450)
HLO_IR|  broadcast.453 = s32[1,1,4]{2,1,0} broadcast(reshape.452), dimensions={0,1,2}
HLO_IR|  reshape.454 = s32[4]{0} reshape(broadcast.453)
HLO_IR|  broadcast.455 = s32[5,4,4]{2,1,0} broadcast(reshape.454), dimensions={2}
HLO_IR|  subtract.456 = s32[5,4,4]{1,2,0} subtract(transpose.451, broadcast.455)
HLO_IR|  transpose.457 = s32[5,4,4]{2,1,0} transpose(subtract.456), dimensions={0,2,1}
HLO_IR|  compare.458 = pred[5,4,4]{2,1,0} compare(transpose.457, broadcast.3), direction=EQ
HLO_IR|  reshape.459 = pred[5,4,4,1]{3,2,1,0} reshape(compare.458)
HLO_IR|  convert.460 = s32[5,4,4,1]{3,2,1,0} convert(reshape.459)
HLO_IR|  broadcast.465 = s32[5,4,4,1]{3,2,1,0} broadcast(convert.460), dimensions={0,1,2,3}
HLO_IR|  reshape.466 = s32[5,4,4]{2,1,0} reshape(broadcast.465)
HLO_IR|  broadcast.467 = s32[5,4,4,2]{3,2,1,0} broadcast(reshape.466), dimensions={0,1,2}
HLO_IR|  multiply.468 = s32[5,4,4,2]{3,2,1,0} multiply(broadcast.464, broadcast.467)
HLO_IR|  reduce.479 = s32[5,4,2]{2,1,0} reduce(multiply.468, constant.11), dimensions={2}, to_apply=region_16.475
HLO_IR|  transpose.480 = s32[5,2,4]{1,2,0} transpose(reduce.479), dimensions={0,2,1}
HLO_IR|  convert.481 = f32[5,2,4]{1,2,0} convert(transpose.480)
HLO_IR|  convert.469 = s32[5,4,4]{2,1,0} convert(compare.458)
HLO_IR|  reduce.474 = s32[5,4]{1,0} reduce(convert.469, constant.11), dimensions={2}, to_apply=region_15.470
HLO_IR|  convert.482 = f32[5,4]{1,0} convert(reduce.474)
HLO_IR|  reshape.483 = f32[5,1,4]{2,1,0} reshape(convert.482)
HLO_IR|  broadcast.484 = f32[5,1,4]{2,1,0} broadcast(reshape.483), dimensions={0,1,2}
HLO_IR|  reshape.485 = f32[5,4]{1,0} reshape(broadcast.484)
HLO_IR|  broadcast.486 = f32[5,2,4]{2,1,0} broadcast(reshape.485), dimensions={0,2}
HLO_IR|  divide.487 = f32[5,2,4]{1,2,0} divide(convert.481, broadcast.486)
HLO_IR|  transpose.488 = f32[5,4,2]{2,1,0} transpose(divide.487), dimensions={0,2,1}
HLO_IR|  reshape.489 = s32[1,4,2]{2,1,0} reshape(Arg_0.1)
HLO_IR|  convert.491 = f32[1,4,2]{2,1,0} convert(reshape.489)
HLO_IR|  reshape.492 = f32[1,1,4,2]{3,2,1,0} reshape(convert.491)
HLO_IR|  broadcast.493 = f32[1,1,4,2]{3,2,1,0} broadcast(reshape.492), dimensions={0,1,2,3}
HLO_IR|  reshape.494 = f32[4,2]{1,0} reshape(broadcast.493)
HLO_IR|  broadcast.495 = f32[5,4,4,2]{3,2,1,0} broadcast(reshape.494), dimensions={2,3}
HLO_IR|  reshape.490 = f32[5,4,1,2]{3,2,1,0} reshape(transpose.488)
HLO_IR|  broadcast.496 = f32[5,4,1,2]{3,2,1,0} broadcast(reshape.490), dimensions={0,1,2,3}
HLO_IR|  reshape.497 = f32[5,4,2]{2,1,0} reshape(broadcast.496)
HLO_IR|  broadcast.498 = f32[5,4,4,2]{3,2,1,0} broadcast(reshape.497), dimensions={0,1,3}
HLO_IR|  subtract.499 = f32[5,4,4,2]{3,2,1,0} subtract(broadcast.495, broadcast.498)
HLO_IR|  multiply.500 = f32[5,4,4,2]{3,2,1,0} multiply(subtract.499, subtract.499)
HLO_IR|  reduce.505 = f32[5,4,4]{2,1,0} reduce(multiply.500, constant.10), dimensions={3}, to_apply=region_17.501
HLO_IR|  constant.9 = f32[] constant(inf)
HLO_IR|  reduce.510 = f32[5,4]{1,0} reduce(reduce.505, constant.9), dimensions={1}, to_apply=region_18.506
HLO_IR|  reduce.515 = f32[5]{0} reduce(reduce.510, constant.10), dimensions={1}, to_apply=region_19.511
HLO_IR|  call.539 = s32[] call(reduce.515), to_apply=argmin_1.531
HLO_IR|  compare.540 = pred[] compare(call.539, constant.11), direction=LT
HLO_IR|  constant.8 = s32[] constant(5)
HLO_IR|  add.541 = s32[] add(call.539, constant.8)
HLO_IR|  select.542 = s32[] select(compare.540, add.541, call.539)
HLO_IR|  dynamic-slice.543 = f32[1,4,2]{2,1,0} dynamic-slice(transpose.488, select.542, constant.11, constant.11), dynamic_slice_sizes={1,4,2}
HLO_IR|  reshape.544 = f32[4,2]{1,0} reshape(dynamic-slice.543)
HLO_IR|  ROOT call.552 = f32[4,2]{1,0} call(reshape.544), to_apply=sort.549
HLO_IR|}"""

input_test_case = [line.split("HLO_IR|")[1] for line in test_case_full.split("\n")]
# input_test_case = [line for line in test_case_full.split("\n")]

case_name = "kmeans_ran_i5-sort-cheetah-dd"

pass_option_mut = {'ori': [], 'mut': ['able_stablesortexpander_0']}
# pass_option_mut = {'ori': [], 'mut': ['able_dynamicdimensionsimplifier_19']}
# pass_option_mut = {'ori': ['setfalse_enableoptimizeselectpredtruefalse'], 'mut': ['disable_algebraicsimplifier']}
# pass_option_mut = {"ori":[],"mut":["able_stablesortexpander_0"]}
# pass_option_mut={"ori":["setfalse_enableoptimizedsidinnderid"],"mut":["disable_algebraicsimplifier"]}
# pass_option_mut={"ori":["setfalse_enableoptimizeselectpredtruefalse", "setfalse_enableoptimizedstranspose"],"mut":["disable_algebraicsimplifier"]}
# pass_option_mut={"ori":[],"mut":["disable_gathersimplifier"]}
# pass_option_mut={"ori":[],"mut":["disable_whileloopsimplifier"]}

# protocalchosen = "SEMI2K"
# protocalchosen = "ABY3"
protocalchosen = "CHEETAH"

matrix = "send_bytes"
# matrix = "send_actions"

## save the test case
case_dir = f"reduce_case/{case_name}.json"
case_dict = {}
case_dict["test_case"] = input_test_case
case_dict["pass_option_mut"] = pass_option_mut
case_dict["protocalchosen"] = protocalchosen
case_dict["matrix"] = matrix
with open(case_dir, "w") as f:
    json.dump(case_dict, f, indent=4)