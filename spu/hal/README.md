# Object model
Generally, we adopt concept-based polymorphism, it's widely used in MLIR, and it provides:
1. runtime polymorphism, just like other object oriented system.
2. value semantic, aka, RAII semantic.

see [here](https://sean-parent.stlab.cc/papers-and-presentations/#value-semantics-and-concept-based-polymorphism) for more details.

# Type system
Type could be modeled as a tuple `<vistype, datatype, shape>`
* **vistype**, the visibility type, could be one of {Public, Secret}.
* **datatype**, the data encoding type, could be on of FXP, INT.
* **shape**, the shape.

Note
* Share type (AShare/BShare/HShare) is not exposed to vm IR, so it's not exposed to IR type.
* plaintext type (u8/i8/u32/i32/f32/f64/...) is 'readonly', we can only import/export them via `buffer view` objects.

# Dispatching
Due to the complexity of MPC value types, the dispatch chain is complicated.

For example,
```python
multiply :: (value<SFXP>, value<PINT>)         # multiply secret fxp to public integer.
```

should be firstly dispatched via encoding type, result in:
```python
multiply :: (SFXP, PINT)         # multiply secret fxp to public integer.
  multiply :: (SINT, PINT) -> SINT
  truncate :: (SINT)
```

then dispatch via visibility type, result in:
```python
multiply :: (SFXP, PINT)         # multiply secret fxp to public integer.
  multiply :: (SINT, PINT) -> SINT
    hess.mul :: (ASHR, PSHR) -> ASHR
  truncate :: (SINT)
```

## Dispatching schema
To model the dispatching rule, we use the following schema,
1. make one dispatch do one thing, i.e. either lowering datatype or visibility type.
2. make the dispatch layered, so they could be moved to compile-time progressively.

Schema: `[<dtype>_]op_name[_<Visibility::type>]`

- if `<dtype>` is missing, (op start with `_`), then dtype will be unchecked.
- if `[<dtype>_]` not provided, then both fxp & int will be well handled.
- if `[_<Visibility::type>]` not provided, then both secret/public will be well handled.

For example:
- `i_add_sp`: int add between secret and public.
-  `_add_sp`: unchecked add between secret & public, works on ring (integer)
-   `add_sp`: add between secret&public, both int, fxp will be handled.
- `f_mul_pp`: fxp mul between public and public.
- `f_mul   `: fxp mul, both secret&public will be handled.

If <dtype> is well specified, then output dtype will be certain, otherwise
output dtype will set to CtType::UNKNOWN.
- `i_add_sp`: out dtype is int.
-   `add_sp`: out dtype is specified, according to inputs type.
-  `_add_sp`: out dtype is UNKNOWN.

## A complicated example
`secret reciprocal` is a typical non-polynomial op, in this example, we use Newton-Raphson to find the approximated result.

The approximation progress depends on other exp/square/add/mul/div, the tracing result is listed below.

Note: `secret reciprocal` may dependends on `divide to public`, while general divide depends on reciprocal, to break the dependenty chain, the typed dispatch schema is import.

Please see [spu/modules/fxp.cc][] for details.

```python
make_secret(BufferView)                    # Value a = make_secret(&ctx, 3.14f);
  constant(BufferView)
  _cast_p2s(value<PFXP>)
f_reciprocal(value<SFXP>)                  # Value c = f_reciprocal(&ctx, a);
  constant(BufferView)                     #  constant(3f)
  constant(BufferView)                     #  constant(0.5f)
  f_sub(value<PFXP>, value<SFXP>)          #  %1 = 0.5 - a
    f_negate(value<SFXP>)                  #    f_negate(a)          # fxp negate
      _negate_s(value<SFXP>)               #      _negate_s(a)       # ring negate for secret
    f_add(value<PFXP>, value<SFXP>)        #    f_add(0.5, -a)       # fxp addition
      _add_sp(value<SFXP>, value<PFXP>)    #      _add_sp(-a, 0.5)   # ring add secret to public, commutative.
                                           # exp iteration begins, exp(x) = (1 + x / n) ^ n
  f_exp(value<SFXP>)                       #  %2 = f_exp(%1)
    constant(BufferView)                   #  constant(256)
    f_div(value<SFXP>, value<PFXP>)        #    f_div(%1, 256)           # fxp divition
      reciprocal_p(value<PFXP>)            #      t = reciprocal(256)    # public reciprocal
      f_mul(value<SFXP>, value<PFXP>)      #      f_mul(%1, t)
        _mul_sp(value<SFXP>, value<PFXP>)  #        _mul_sp(%1, t)       # ring multiply secret to public
    constant(BufferView)
    f_add(value<SFXP>, value<PFXP>)
      _add_sp(value<SFXP>, value<PFXP>)
    f_square(value<SFXP>)
      f_mul(value<SFXP>, value<SFXP>)
        _mul_ss(value<SFXP>, value<SFXP>)
    ...
    f_square(value<SFXP>)
      f_mul(value<SFXP>, value<SFXP>)
        _mul_ss(value<SFXP>, value<SFXP>)
                                           # Newton-Rapson iteration begins, 1/x = 3 * exp(0.5 - x) + 0.003
  f_mul(value<PFXP>, value<SFXP>)
    _mul_sp(value<SFXP>, value<PFXP>)
  constant(BufferView)
  f_add(value<SFXP>, value<PFXP>)
    _add_sp(value<SFXP>, value<PFXP>)
  f_square(value<SFXP>)
    f_mul(value<SFXP>, value<SFXP>)
      _mul_ss(value<SFXP>, value<SFXP>)
  f_mul(value<SFXP>, value<SFXP>)
    _mul_ss(value<SFXP>, value<SFXP>)
  f_sub(value<SFXP>, value<SFXP>)
    f_negate(value<SFXP>)
      _negate_s(value<SFXP>)
    f_add(value<SFXP>, value<SFXP>)
      _add_ss(value<SFXP>, value<SFXP>)
  f_add(value<SFXP>, value<SFXP>)
    _add_ss(value<SFXP>, value<SFXP>)
  ...
_s2p(value<SFXP>)
dump_public(value<PFXP>)
```

## Progressive lowering
The lowering could be done in runtime (dynamic) or compile-time (static), the final goal is to make a thin-runtime (secret protocol only) and fat-compile-time (all data computation, numeric approximations..), but currently it's hard to move all into compile-time, so we layering the runtime dispatch rule and try to move them one by one to compile-time.
