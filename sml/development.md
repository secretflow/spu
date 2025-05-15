# Development

We welcome developers of all skill levels to contribute their expertise.
There are many ways to contribute to SML including reporting a bug, improving the documentation and contributing new algorithm.
Of course, if you have any suggestion or feature request, feel free to open an [issue](https://github.com/secretflow/spu/issues).

## Submitting a bug report

If you want to submit an issue, please do your best to follow these guidelines which will make it easier and quicker to provide you with good feedback:

- Contains a **short reproducible** code snippet, so anyone can reproduce the bug easily
- If an exception is raised, please provide **the full traceback**.
- including your operating system type, version of JAX, SPU(or commit id)

## Contributing code

![sml develop paradiam](./sml_develop.svg)

> 1. To avoid duplicating work, it is highly advised that you search through the issue tracker and the PR list.
> If in doubt about duplicated work, or if you want to work on a non-trivial feature,
> it's **recommended** to first open an issue in the issue tracker to get some feedbacks from core developers.
> 2. Some essential [documents](https://www.secretflow.org.cn/docs/spu/latest/en-US) about SPU are highly recommended.
[This](../docs/tutorials/develop_your_first_mpc_application.ipynb) is a good first tutorial for new developers,
[pitfall](../docs/development/fxp.ipynb) will be a cheatsheet when you come across numerical problem.

The preferred way to contribute to SML is to fork the main repository, then submit a "pull request" (PR).

1. Create a GitHub account if you do not have one.
2. Fork the [project repository](https://github.com/secretflow/spu),
your can refer to [this](https://docs.github.com/en/get-started/quickstart/fork-a-repo) for more details.
3. Following the instructions on [CONTRIBUTING](../CONTRIBUTING.md), installing the prerequisites and running tests successfully.
4. Develop the feature on **your feature branch** on your computer,
using [Git](https://docs.github.com/en/get-started/quickstart/set-up-git) to do the version control.
5. Following [Before Pull Request](<./development.md#Before Pull Request>) to place or test your codes,
[these](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
to create a pull request from your fork.
6. Committers do code review and then merge.

## Before Pull Request

When finishing your coding work, you are supposed to do some extra work before pulling request.

1. **Make sure your code is up-to-date**: It is often helpful to keep your local feature branch **synchronized** with
the latest changes of the main SPU repository.
2. **Place your codes properly**: Generally speaking, for every algorithm, at least 3 files are needed
(e.g. for kmeans, check [PR](https://github.com/secretflow/spu/pull/277/files) as an example).
   - `kmeans.py`: implementation of algorithm or new features, it should be a **"jit-able"** program which run correctly in plaintext
   (same or near to output from scikit-learn).
   - `kmeans_test.py`: a unittest python file, in which you test your program with **simulator**, then you should report the behavior
   (like correctness or error rate) under MPC setting.
   - `kmeans_emul.py`: similar to the above file, except you will test program with **emulator**,
   then you can get sense of efficiency under different MPC protocols.
3. **Other things**: there are still some small fixes to do.
   - **Add copyright**: see [this](<../CONTRIBUTING.md#Contributor License Agreement>) for details.
   - **Add necessary doc**: your implementation may only have part features, or some changes have been made for limitation of both JAX and SPU,
    you **MUST** describe these things explicitly!
   - **Add/change bazel file**: currently, we adopt [bazel](https://github.com/bazelbuild/bazel) to manage our module,
   so you might need add some [python rules](https://bazel.build/reference/be/python) in `BUILD.bazel`.
   - **Format all your files**: using [buildifier](https://github.com/bazelbuild/buildtools/tree/master/buildifier) to format bazel file,
    [black](https://github.com/psf/black) to format python file, [isort](https://github.com/PyCQA/isort) to sort the python imports.

## Advanced Topics

### Reveal some data during your programs

First and foremost, it is important to emphasize that revealing data in plaintext is a risky operation.
You must be fully aware of the potential data leakage risks associated with this action.
If you still need to reveal some data from the SPU, you can use the `sml_reveal` function defined in `sml/utils/utils.py`,
which allows you to reveal one or more arrays as plaintext.

#### How to use the revealed values

Note that the data revealed in your program will still be in the form of arrays, and you must ensure that your program remains fully jitable.
Therefore, you **cannot** use standard Python control flow statements (such as if-else or while loops); instead, you should use JAX's control flow constructs.

**CASE 1**: Conditional judgment on revealed data

```python

def reveal_func_single(x):
   # We assume the input `x` is an 1-d array
    y = jnp.log(x)
    # reveal single value
    xx = sml_reveal(x)

    # x is 1-d array, so we fetch the first element
    pred = xx[0] > 0

    def true_branch(xx):
        return jnp.log(xx), True

    def false_branch(xx):
        return jnp.log(-xx), False

    # use jax.lax.cond to replace if-else
    yy, preds = jax.lax.cond(pred, true_branch, false_branch, xx)

    return y, yy, preds
```

**CASE 2**: Use revealed data to determine while loop exit

```python
def reveal_while_loop(x):
    y = sml_reveal(jnp.max(x))

    def cond_fun(carry):
        _, y = carry
        # jnp.max return 0-dim array, so we can fetch y directly
        return y > 3

    def body_fun(carry):
        x, _ = carry
        new_x = x - 1
        new_y = sml_reveal(jnp.max(new_x))
        return new_x, new_y

    x, _ = jax.lax.while_loop(cond_fun, body_fun, (x, y))

    return x

```

For concrete usage of the two examples above, please refer to `sml/utils/tests/reveal_test.py`.

Finally, in `sml/linear_model/logistic.py`, we have also implemented a simple early stopping mechanism based on parameter changes (as a practical
application of `while_loop`). This allows for more efficient logistic model training by revealing only a few bits,
which can be considered to have very limited information leakage.
