Type System
===========

Overview
--------

This document is for VM developers.

Everything in SPU could be treated as an object, each object has a type.

There are only two types of objects, *value* or *operator*, which means if a symbol is not a *value*, it's an *operator*.

- **value**: an object that is managed by SPU runtime, represent a public/secret data.
- **operator**: an object that takes one or more values and outputs a return value, i.e. `multiply` is an operator.

Value type
~~~~~~~~~~

A value type is a tuple (**V**, **D**, **S**), where:

- **V** is *visibility*, could be one of *{public, private, secret}*
- **D** is *data type*, could be one of *{int, fxp}*
- **S** is *shape*, which make a value a tensor.

We can define a hyper type function, which takes three parameters and return a concrete value type.

.. code-block:: haskell

  type(V, D, S) -> ValueType

To simplify things a little bit, we can ignore *shape* for now and assume that runtime will handle it correctly.

.. code-block:: haskell

  type(V, D) -> ValueType

With this type function, we can define a list of types in the SPU type system.

.. code-block:: haskell

  sint = type(secret, int)
  sfxp = type(secret, fxp)
  pint = type(public, int)
  pfxp = type(public, fxp)
  vint = type(private, int)
  vfxp = type(private, fxp)

Operator type
~~~~~~~~~~~~~

*Operators* takes a list of values as parameters and return exactly one value as result, operator's type is determined by the types of input parameters and return values.

In SPU IR, an operator could take a polymorphic typed parameter and the return type could be deduced from the parameters. For example:

.. code-block:: haskell

  add (sint, pint) -> sint
  add (sint, sint) -> sint
  add (pint, pint) -> pint
  ...
  add (sfxp, pint) -> sfxp
  ...

The `add` operator takes a pair of `type(V, D)` as parameter, which has 2x3x2x3 = 36 different kinds of combinations. To support this type of operators, we introduce the following *type functor*.

1. **dtype promotion**, which promotes two dtypes to a more relaxed type, in SPU system, *int* is always promoted to *fxp*.

.. code-block:: haskell

  promote :: list[D] -> D
  -- for example
  add (int, fxp) -> fxp

2. **visibility narrow**, which narrows the visibility when two or more operands have different visibility properties, this is the key to maintain the "secure semantic" of SPU VM, since the resulting visibility of ops will always be more strict. i.e. if one of operands is *secret*, the result is a *secret*.

.. code-block:: haskell

  narrow :: list[V] -> V

  -- for example
  add (secret, public) -> secret
  add (public, public) -> public

Now we can represent the polymorphic mul op as:

.. code-block:: haskell

  mul :: type(V0, D0) -> type(V1, D1) -> type(narrow(V0, V1), promote(D0, D1))

- the op takes two parameters, first type is :code:`type(V0, D0)`, second type is :code:`type(V1, D1)`.
- the op returns :code:`type(narrow(V0, V1), promote(D0, D1))` as a result.
- when applying the op to two arbitrary arguments, the result could be deduced from the above type expression.


Use of type
~~~~~~~~~~~

There are many uses for types.

- First, the most important one, type is self descriptive, with an accurate defined type system, we can describe *SPU IR* more accurate.
- Second, runtime type information is used to do runtime dispatch, which is important for polymorphic operators.
- Third, the type system could be used by static type checker, and could be used to double check runtime implementation.


Ops dispatch
------------

As described above, type helps for dispatch, here we use `MUL` instruction as an example.

.. code-block:: python

  %3 = MUL %1 %2


The above `MUL` instruction does element-wise multiplication, `%1` and `%2` are parameters and `%3` is the return value.

The dispatch problem
~~~~~~~~~~~~~~~~~~~~

In this example, `%1` and `%2` are SPU values, each of them belongs one of four types `{sint, pint, sfxp, pfxp}`, the type of `MUL` is:

.. math::
  
  \begin{Bmatrix} sint \\ pint \\ sfxp \\ pfxp \end{Bmatrix}
  \times
  \begin{Bmatrix} sint \\ pint \\ sfxp \\ pfxp \end{Bmatrix}

**The problem is dispatch to correct kernel according to the arguments' type information**.

A simple idea is to pattern match all these type combinations and dispatch to different kernels accordingly, with this way we got 4x4=16 different kernels.

.. mermaid::

  graph LR
    mul[mul] --> dispatch((dispatch))
    dispatch:::dispatch --> mul_si_si[mul<sint,sint>]
    dispatch:::dispatch --> mul_si_sf[mul<sint,sfxp>]
    dispatch:::dispatch --> mul_si_pi[mul<sint,pint>]
    dispatch:::dispatch --> mul_si_pf[mul<sint,pfxp>]
    dispatch:::dispatch --> mul_sf_si[mul<sfxp,sint>]
    dispatch:::dispatch --> mul____[...]
    dispatch:::dispatch --> mul_pf_pf[mul<pfxp,pfxp>]
    classDef dispatch fill:#f96;
    classDef compose fill:#03fcb1;


Layered dispatch
~~~~~~~~~~~~~~~~

A better way to is to dispatch layer by layer, for example, first dispatch by dtype, then dispatch by vtype.

.. mermaid::

  graph LR
    mul[mul] --> mul_ddispatch((dtype dispatch))
    mul_ddispatch:::dispatch --> imul[imul]
    imul --> rmul
    mul_ddispatch --> fmul[fmul]
    fmul --> fmul_d{+}
    fmul_d:::compose --> rmul[rmul]
    rmul --> rmul_vdispatch((vtype dispatch))
    rmul_vdispatch:::dispatch  --> mulss[mulss]
    rmul_vdispatch --> mulsp[mulsp]
    rmul_vdispatch --> mulpp[mulpp]
    fmul_d --> rtrunc[rtrunc]
    rtrunc --> rtrunc_vdispatch((vtype dispatch))
    rtrunc_vdispatch:::dispatch  --> truncss[truncss]
    rtrunc_vdispatch --> truncsp[truncsp]
    rtrunc_vdispatch --> truncpp[truncpp]

    classDef dispatch fill:#f96;
    classDef compose fill:#03fcb1;

In the above diagram:

- **mul** is general *multiplication* method.
- **imul** is integer multiplication method.
- **fmul** is fixedpoint multiplication method.
- **rmul** is untyped multiplication method over ring 2k.
- **mulss** multiplies two secret, the domain and behavior are secure protocol dependent.

The above idea can be expressed in code like:

.. code-block:: cpp
  :linenos:

  Value i2f(Value); // convert int to fxp

  Value mul(Value x, Value y) {
    Type xt = x.type();
    Type yt = y.type();

    // first level, dispatch by dtype.
    if (is_int(xt) && is_int(yt)) return imul(x, y);
    if (is_int(xt) && is_fxp(yt)) return fmul(i2f(x), y);
    if (is_fxp(xt) && is_int(yt)) return fmul(x, i2f(y));
    if (is_fxp(xt) && is_fxp(yt)) return fmul(x, y);
  }

  Value imul(Value x, Value y) {
    Type xt = x.type();
    Type yt = y.type();

    // second level, dispatch by vtype.
    if (is_secret(xt) && is_secret(yt)) return _mul_ss(x, y);
    if (is_secret(xt) && is_public(yt)) return _mul_sp(x, y);
    if (is_public(xt) && is_secret(yt)) return _mul_sp(y, x); // commutative
    if (is_public(xt) && is_public(yt)) return _mul_pp(x, y);
  }

  Value fmul(Value x, Value y) {
    Value z = imul(x, y);
    return truncate(z);
  }

Fast dispatch
~~~~~~~~~~~~~

In the above example, we observe that `i2f` and `truncation` could be optimized, the intuition is when a value is converted from `int` to `fxp` and later convert back, these two conversion introduce non-trivial computation overhead in MPC setting.

We use the so called *fast dispatch* to optimize it, when doing cross `int` and `fxp` multiplication, we could directly do `imul` without type lift and truncation.

.. code-block:: cpp
  :linenos:

  Value i2f(Value); // convert int to fxp

  Value mul(Value x, Value y) {
    Type xt = x.type();
    Type yt = y.type();

    // fast dispatch
    if (one_int_another_fxp(xt, yt)) return imul(x, y);

    if (is_int(xt) && is_int(yt)) return imul(x, y);
    if (is_int(xt) && is_fxp(yt)) return fmul(i2f(x), y);  // lift to f, then truncation back.
    if (is_fxp(xt) && is_int(yt)) return fmul(x, i2f(y));  // lift to f, then truncation back.
    if (is_fxp(xt) && is_fxp(yt)) return fmul(x, y);
  }

Note: 

- in the above implementation we didn't maintain the type correctness.
- this pattern match based *fast dispatch* is exactly the same as compile-time *peephole optimization*.
- dispatch inside a protocol is also complicated and beyond the scope of this article.


Implementation
~~~~~~~~~~~~~~

With *type functor*, we have the following op definitions in `mul` dispatch chain.

.. code-block:: haskell

  mul   :: type(#V0,$D0) -> type(#V1,$D1) -> type(narrow(#V0, #V1), promote($D0, $D1))
  fmul  :: type(#V0,FXP) -> type(#V1,FXP) -> type(narrow(#V0, #V1), FXP)
  rmul  :: type(#V0,$$) -> type(#V1,$$) -> type(narrow(#V0, #V1), $$)
  mulss :: type(SECRET,$$) -> type(SECRET,$$) -> type(SECRET,$$)

In dispatch phrase, SPU runtime uses type information to select next dispatch op. In this example, `(x:sfxp, y:sfxp)` is applied op `mul`, via pattern matching we got `(V0=SECRET,D0=FXP), (V1=SECRET,D1=FXP)`, and the dispatch stack looks like:

.. code-block:: python

  mul(x:sfxp, y:sfxp)            # dtype dispatch, use D0=FXP, D1=FXP to select fmul
    fmul(x:sfxp, y:sfxp)         # fixed point arithmetic implementation,
                                 #   first do ring multiplication, then truncate the result.
      z = rmul(x:s<T>, y:s<T>)   # rmul does ring arithmetic over protocol dependent
                                 #   encoding, and use (V0=SECRET,V1=SECRET) to select mulss
        mulss(x:U, y:U)          # mulss dispatch to concrete protocol implementation,
                                 #   in protocol defined field.
      rtruncate(z:s<T>)          # rtrunc does ring truncation over protocol dependent
                                 #   field, and use (V0=SECERT) to select truncates
        truncs(z:U)              # dispatch to concrete protocol implementation.


Note:

- We use C++-like template type notation to represent polymorphic type constraints.

Partial type
^^^^^^^^^^^^

In the type dispatch step, type information is used to select next op, and when partial of type information is used, it's *erased*. For example, when `dtype` is used to select `fmul` in the above example, dtype is useless in the future and could be erased, the lower level op does not distinguish dtype (via a generic type parameter). In a real implementation, we don't erase the type explicitly, just leave it there without further use.

The return value takes the `reverse progress` of dispatch. The return type is filled from bottom to up. For example, in the above progress, when :code:`z=rmul(x,y)` is called, `rmul` knows `z`'s visibility type is `SECRET` but does not know its dtype yet, so here `z` has a partial type `type(SECRET, $UNKNOWN)`. The type will be filled step by step during stack popup, and eventually completed as a full type when the whole dispatch progress is done.
