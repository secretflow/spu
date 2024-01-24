Visibility Inference
####################

Overview
********

This document is for SPU compiler developers.

In SPU compiler stack, visibility is part of type qualifier. During legalize stablehlo to pphlo,
every SSA value and BlockArg need to assign with a proper visibility. This procedure is referred as **Visibility Inference**

Common Visibility
_________________

When an operation inputs have different visibilities, the way of compute common visibility is defined as:

.. code-block:: haskell

    let public = 0
    let secret = 1
    common_visibility = max(v1, v2,..., vn)


Nullary Op
__________

Nullary ops in stablehlo are `ConstantOp <https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant>`_
and `IotaOp <https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota>`_

All nullary ops follows the following inference rule:

.. code-block:: haskell

    op() -> public


Unary Op
________

Most unary ops in **stablehlo** cannot change visibility of input argument and follows the following inference rule:

.. code-block:: haskell

    op(public) -> public
    op(secret) -> secret


Binary Op
_________

Most binary ops in **stablehlo** yield a result with common visibility of **lhs** and **rhs**
and follows the following inference rule:

.. code-block:: haskell

    op(v1,  v2) -> common_visibility(v1, v2)


Control Flow Op
_______________

#. `If <https://github.com/openxla/stablehlo/blob/main/docs/spec.md#if>`_ and `Case <https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case>`_
    In short, each result of an if statement is common visibility of results from different branches and predicate.

    If different branches yield results of different visibilities, cast to common visibility will be inserted before relative return.

    .. code-block:: haskell

        // True branch and false branch yield same visibility for a result
        if(vpred) ({
            return (v0, v1)
        }, {
            return (v0, v1)
        }) -> common_visibility(vpred, v0), common_visibility(vpred, v1)

        // True branch and false branch yield different visibilities for a result
        if(vpred) ({
            return (v0)
        }, {
            return (v1)
        }) -> common_visibility(vpred, v0, v1)

        // predicate has the most common visibility
        if(vpred) ({
            return (v0)
        }, {
            return (v1)
        }) -> vpred

#. `While <https://github.com/openxla/stablehlo/blob/main/docs/spec.md#while>`_
    For while body, consider result visibility might be different from input visibility, multi-rounds of visibility inference
    is applied on body region. The final result will be all input visibility matches result visibility.

    **Attention**: Although no protocol supports **while** with a non-public cond region at this point,
    compiler in general does not error out here.

    .. code-block:: haskell

        while(v0, v1, v2) ({
            return vpred
        }, {
            return (v0, v1, v2)
        }) -> common_visibility(vpred, v0), common_visibility(vpred, v1), common_visibility(vpred, v2)



Reduce Related Op
_________________

#. `Reduce <https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce>`_ and `ReduceWindow <https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_window>`_

    .. code-block:: haskell

        v0 = common_visibility(Vop0, Vinit0)
        v1 = common_visibility(Vop1, Vinit1)
        reduce(v0, v1, v0, v1) ({
            return (v0, v1)
        }) -> (v0, v1)

#. `SelectAndScatter <https://github.com/openxla/stablehlo/blob/main/docs/spec.md#select_and_scatter>`_
    In general the rule is
        * visibility(opernad) == visibility(init)
        * visibility(result) == common_visibility(operand, source, init)

    .. code-block:: haskell

        vOpAndInit = common_visibility(Vop, Vinit)
        reduce(vOpAndInit, vSource, vOpAndInit) ({
            return Vselect
        },
        {
            return Vscatter
        )) -> common_visibility(vSource, vOpAndInit)
