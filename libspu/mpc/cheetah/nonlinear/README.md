# Performance Stats

|                               | bit width      | Send & Recv (bits) |
| ----------------------------- | -------------- | ------------------ |
| Millionare (radix = 4)        | 32             | 348.377            |
| Millionare (radix = 4)        | 40             | 428.377            |
| Millionare (radix = 4)        | 64             | 690.503            |
| TruncatePr (unknown sign bit) | k=32, fxp = 12 | 369.503            |
| TruncatePr (known sign bit)   | k=32, fxp = 12 | 32.252             |
| TruncatePr (unknown sign bit) | k=64, fxp = 12 | 723.754            |
| TruncatePr (known sign bit)   | k=64, fxp = 12 | 36.252             |
|   Truncate (unknown sign bit) | k=32, fxp = 12 | 518.503            |
|   Truncate (known sign bit)   | k=32, fxp = 12 | 197.503            |
|   Truncate (unknown sign bit) | k=64, fxp = 12 | 904.754            |
|   Truncate (known sign bit)   | k=64, fxp = 12 | 233.503            |


* Note: The perfermance stats here are average from at least 2^17 input length.

* _Millionare_ Protocol implements the tree-based protocol from [CrypTFlow2](https://eprint.iacr.org/2020/1002)).
  We set the comparison radix as `4`
* _TruncatePr_ Protocol implements the 1-bit approximated truncate protocol from [Cheetah](https://eprint.iacr.org/2022/207.pdf)).
  We also set the comparison radix as `4` by default.
* _Truncate_ Protocol is the exact version of _TruncatePr_.
