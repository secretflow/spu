# Pitfalls - Fxp Arithmetic

Due to approximations during fxp arithmetic, we have confirmed the precision issues or possible limitations with the following ops.
We will update this part promptly.

### Experiment Settings

```
PROTOCOL = ABY3
FIELD = 64
FXP_BITS = 18

DATAPOINT_PER_EXP = 10000
RTOL = 1e-2
ATOL = 0.0001
```

Passing criteria is
> absolute(a - b) <= (atol + rtol * absolute(b))

#### Reciprocal & Div

SPU uses Goldschmidt's method to calculate Div and Reciprocal.
Referrence: [Secure Computation With Fixed-Point Numbers](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.1305&rep=rep1&type=pdf)

##### Reciprocal Experiments

|    id    |Pass Rate|     Description      |
|----------|--------:|----------------------|
|Testcase 1|   1.0000|`uniform(0, 1e3)`       |
|Testcase 2|   0.2675|`uniform(1e3, 1e6)`     |
|Testcase 3|   0.0042|`uniform(1e6, 1e9)`     |
|Testcase 4|   1.0000|`uniform(-1e3, 0)`      |
|Testcase 4|   0.9997|`uniform(-2**18, 2**18)`|

##### Div Experiments

|    id    |Pass Rate|                   Description                    |
|----------|--------:|--------------------------------------------------|
|Testcase 1|   0.4354|`both x and y belongs to uniform(-2**18, 2**18)`    |
|Testcase 2|   1.0000|`x: uniform(-2**18, 2**18), y: uniform(-2**9, 2**9)`|
|Testcase 3|   0.6335|`x: uniform(-2**9, 2**9), y: uniform(-2**18, 2**18)`|

We can draw the conclusion from experiments that precision of Reciprocal and Div is quite high.
If you would like to improve the precision, you may need to enlarge FXP_BITS.

Another possible path is to investigate the inital approximation of 1/b or dynamic selection of magic numbers.

#### Log(natural logarithm) & Log1p

Current implementation is based on Pade approximation.
In SPU, we can ensure accuracy if input is in [0, 2\*\*18]. If x is above 2\*\*18, the result is wrong due to our normalization implementation.

Referrence: [Benchmarking Privacy Preserving Scientific Operations](https://www.esat.kuleuven.be/cosic/publications/article-3013.pdf)

##### Log Experiments

|    id    |Pass Rate|   Description   |
|----------|--------:|-----------------|
|Testcase 1|        1|`uniform(0, 2**18)`|

#### Exp(natural exponential)

The current implementation is based on Pade approximation. Exp depends on Log, so it only works in specific ranges, please check the experiment result below.

Referrence: [Benchmarking Privacy Preserving Scientific Operations](https://www.esat.kuleuven.be/cosic/publications/article-3013.pdf)

##### Exp Experiments

|    id    |Pass Rate|  Description   |
|----------|--------:|----------------|
|Testcase 1|    0.924|`uniform(-10, 20)`|

#### Logistic

Please set sigmoid_mode to *REAL* for high precision. Otherwise, logistic is implemented by Taylor approximations.

##### Logistic Experiments

|    id    |Pass Rate|  Description   |
|----------|--------:|----------------|
|Testcase 1|    0.967|`uniform(-10, 10)`|

#### Power

At this moment, the base number must be positive.

#### Tanh(hyperbolic tangent)

We use order (5,5) pade approximation, refer to https://www.wolframalpha.com/input?i=Pade+approximation+tanh%28x%29+order+5%2C5.
