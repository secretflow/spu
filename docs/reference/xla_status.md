## XLA Implementation Status

List of XLA(mhlo-mlir) Ops that SPU supports:

    The list of mhlo ops is obtained from this file:
        https://github.com/openxla/xla/blob/main/xla/mlir_hlo/mhlo/IR/hlo_ops.td

    General limitation with SPU:
        * Dynamic shape is not supported
        * SPU uses fixed-point numbers to simulate floating point, so nonfinite numbers are not supported

### XLA nullary ops

|  Op Name       | supported(yes/partial/no) | notes       |
| :------------: | :-----------------------: | ----------- |
| `constant`     | yes                       | Always yields a public value
| `iota`         | yes                       | Always yields a public value
| `dynamic_iota` | no                        |
| `create_token` | no                        |


### XLA unary element-wise ops

|  Op Name       | supported(yes/partial/no) | notes       |
| :------------: | :-----------------------: | ----------- |
| `abs`          | yes                       |
| `cbrt`         | no                        |
| `ceil`         | yes                       |
| `convert`      | yes                       |
| `count_leading_zeros`| no                  |
| `cosine`       | yes                       |
| `exponential`  | yes                       |
| `exponential_minus_one`| yes               |
| `floor`        | yes                       |
| `imag`         | yes                       |
| `is_finite`    | no                        |
| `log`          | yes                       |
| `log_plus_one` | yes                       |
| `logistic`     | yes                       |
| `not`          | yes                       |
| `negate`       | yes                       |
| `popcnt`       | not                       |
| `real`         | yes                       |
| `round_nearest_afz`| not                   |
| `rsqrt`        | yes                       |
| `sign`         | partial                   |
| `sine`         | yes                       |
| `sqrt`         | yes                       |
| `tanh`         | yes                       |


### XLA binary element-wise ops

|  Op Name       | supported(yes/partial/no) | notes       |
| :------------: | :-----------------------: | ----------- |
| `add`          | yes                       |
| `atan2`        | no                        |
| `complex`      | yes                       |
| `compare`      | yes                       |
| `divide`       | yes                       |
| `maximum`      | yes                       |
| `minimum`      | yes                       |
| `multiply`     | yes                       |
| `power`        | yes                       |
| `remainder`    | yes                       |
| `shift_left`   | partial                   | rhs must be a public
| `shift_right_arithmetic` | partial         | rhs must be a public
| `shift_right_logical` | partial            | rhs must be a public
| `subtract`     | yes                       |


### XLA binary logical element-wise ops

|  Op Name       | supported(yes/partial/no) | notes       |
| :------------: | :-----------------------: | ----------- |
| `and`          | yes                       |
| `or`           | yes                       |
| `xor`          | yes                       |


### XLA communication ops

|  Op Name       | supported(yes/partial/no) | notes       |
| :------------: | :-----------------------: | ----------- |
| `infeed`       | no                        |
| `outfeed`      | no                        |
| `send`         | no                        |
| `recv`         | no                        |


### XLA parallelism related ops

|  Op Name       | supported(yes/partial/no) | notes       |
| :------------: | :-----------------------: | ----------- |
| `replica_id`   | no                        |


### XLA control flow ops

|  Op Name       | supported(yes/partial/no) | notes       |
| :------------: | :-----------------------: | ----------- |
| `after_all`    | no                        |
| `if`           | yes                       |
| `case`         | no                        |
| `while`        | partial                   | condition region must return a public scalar
| `all_gather`   | no                        |
| `all_reduce`   | no                        |
| `reduce_scatter` | no                      |
| `all_to_all`   | no                        |
| `reduce`       | yes                       | inherits limitations from reduce function


### XLA tuple ops

|  Op Name       | supported(yes/partial/no) | notes       |
| :------------: | :-----------------------: | ----------- |
| `get_tuple_element` | yes                  |
| `tuple`        | yes                       |


### XLA other ops

|  Op Name       | supported(yes/partial/no) | notes       |
| :------------: | :-------------------------: | ----------- |
| `slice`        | yes                       |
| `dynamic-slice`| yes                       |
| `dynamic-update-slice`| yes                |
| `batch_norm_grad`| yes                     | Rely on XLA's batchnorm_expander pass
| `batch_norm_inference`| yes                | Rely on XLA's batchnorm_expander pass
| `batch_norm_training` | yes                | Rely on XLA's batchnorm_expander pass
| `bitcast_convert` | partial                | Only supports convert to type of same size
| `broadcast`    | yes                       |
| `broadcast_in_dim` | yes                   |
| `dynamic_broadcast_in_dim` | no            |
| `cholesky`     | yes                       | Rely on CholeskyExpander pass
| `clamp`        | yes                       |
| `concatenate`  | yes                       |
| `collective_permute` | no                  |
| `convolution`  | yes                       |
| `copy`         | no                        |
| `cross-replica-sum` | no                   |
| `custom_call`  | no                        |
| `dot`          | yes                       |
| `dot_general`  | yes                       |
| `einsum`       | yes                       |
| `unary_einsum` | yes                       |
| `fft`          | no                        |
| `gather`       | yes                       |
| `get_dimension_size` | no                  |
| `map`          | yes                       | Rely on XLA's MapInliner pass
| `reshape`      | yes                       |
| `dynamic_reshape` | no                     |
| `scatter`      | no                        |
| `select`       | yes                       |
| `select_and_scatter` | yes                 |
| `set_dimension_size` | no                  |
| `sort`         | yes                       |
| `reverse`      | yes                       |
| `pad`          | yes                       |
| `trace`        | no                        |
| `transpose`    | yes                       |
| `triangular_solve` | yes                   | Rely on XLA's TriangularSolverExpander pass
| `reduce_window`| yes                       |
| `return`       | yes                       |
| `torch_index_select` | no                  |
| `optimization_barrier` | no                |


### XLA RNG ops

|  Op Name       | supported(yes/partial/no) | notes       |
| :------------: | :-----------------------: | ----------- |
| `rng_uniform`  | yes                       | Bound [a, b) must all be public scalar, result is also a public tensor
| `rng_normal`   | no                        |
| `rng_bit_generator` | no                   |


### XLA quantize op

|  Op Name       | supported(yes/partial/no) | notes       |
| :------------: | :-----------------------: | ----------- |
| `dequantize`   | no                        |


### XLA miscellaneous ops

|  Op Name       | supported(yes/partial/no) | notes       |
| :------------: | :-----------------------: | ----------- |
| `fusion`       | no                        |
| `bitcast`      | no                        | Internal op to XLA/GPU
| `reduce_precision` | no                    |
| `real_dynamic_slice` | no                  |
| `dynamic_pad`  | no                        |
| `dynamic_gather` | no                      |
| `dynamic_conv` | no                        |
| `print`        | no                        |
| `compute_reshape_shape` | no               |
| `cstr_reshapable` | no                     |

