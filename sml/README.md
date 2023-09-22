# SML: Secure Machine Learning

**SML** is a python module implementing machine learning algorithm with [JAX](https://github.com/google/jax),
which can do **secure** training and inferring under the magic of [SPU](https://github.com/secretflow/spu).

Our vision is to establish a general-purpose privacy-preserving machine learning(PPML) library,
being a secure version of [scikit-learn](https://github.com/scikit-learn/scikit-learn).

Normally, the APIs of our algorithms are designed to be as consistent as possible with scikit-learn.
However, due to safety considerations and certain limitations of the SPU, some APIs will undergo changes.
Detailed explanations will be provided for any differences in the doc.

## Why not scikit-learn

First, scikit-learn is built top on Numpy and SciPy, running on centralized mode.
So you must collect all data into one node, which can't protect the privacy of data.

The implementations in scikit-learn are usually very efficient and valid, then why not we just "translate" it to MPC?

The quick answer for this question is **accuracy** and **efficiency**.

In PPML, we observe that most framework encodes floating-point to fixed-point number,
which parameterized by `field`(bitwidth of underlying integer) and `fxp_fraction_bits`(fractional part bitwidth),
greatly restricting the effective range and precision of floating-point numbers.
on other hand, The major determinant of computational overhead is determined by the MPC protocol,
so the origin cpu-friendly ops may have pool performance.

### Our Solution

So we establish a new library SML trying to bridge these gaps:

1. accuracy: optimize and test the algorithm based on fixed-point number,
e.g. prefer high-precision ops(`rsqrt` rather than `1/sqrt`),
essential re-transform to accommodate the valid range of non-linear ops
(see [fxp pitfalls](../docs/development/fxp.ipynb)).
2. efficiency: use MPC-friendly op to replace CPU-friendly op,
e.g. use numeric approximation trick to avoid sophistic computation, prefer arithmetic ops to comparison ops.

Of course, we also supply an easy-to-test toolbox for advanced developer
who wants to develop their own MPC program:

1. `Simulator`: provide a fixed-point computation environment and run at high speed.
But it's unable to provide a real SPU performance environment,
the test results cannot reflect the actual performance of the algorithm.
2. `Emulator`: emulate on the real MPC protocol using multiple processes/Docker(coming soon),
and can provide effective performance results.

So the **accuracy** can be proved if the algorithm pass the test of `simulator`,
and you should test the **efficiency** using `emulator`.

> WARNING: currently, SML is undergoing rapid developments,
> so it is not recommended for direct use in production environments.

## Installation

First, you should clone the spu repo to your local disk:

```bash
git clone https://github.com/secretflow/spu.git
```

Some [Prerequisites](../CONTRIBUTING.md#build) are required according to your system.
After all these installed, you can run any test like:

```bash
# run kmeans simulation
# simulation: run program in single process
# used for correctness test
bazel run -c opt //sml/cluster/tests:kmeans_test

# run kmeans emulation
# emulation: run program with multiple processes(LAN setting)
# or multiple dockers(WAN setting, will come soon)
# used for efficiency test.
bazel run -c opt //sml/cluster/emulations:kmeans_emul
```

## Algorithm Support lists

See [support lists](./support_lists.md) for all our algorithms and features we support.

## Development

See [development](./development.md) if you would like to contribute to SML.

## FAQ

We collect some [FAQ](./faq.md), you can check it first before submitting an issue.
