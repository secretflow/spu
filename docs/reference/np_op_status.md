
JAX NumPy Operators Status
==========================

# Overview


SPU recommends users to use JAX as frontend language to write logics. We found most of users would utilize **jax.numpy** modules in their programs. We have conducted tests with some *selected* Operators for reference.

Just keep in mind, if you couldn't find a **jax.numpy** operator in this list, it doesn't mean it's not supported. We just haven't test it, e.g. jax.numpy.sort. And we don't test other **JAX** modules at this moment.

Please contact us if
- You need to confirm the status of another **jax.numpy** operator not listed here.
- You find a **jax.numpy** is not working even it is marked as **PASS**. e.g. The precision is bad.



# Tested Operators List

## abs

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.abs.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## add

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.add.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## angle

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.angle.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


stablehlo.atan2
## angle

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.angle.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


stablehlo.atan2
## arccos

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arccos.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


stablehlo.atan2
## arccosh

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arccosh.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


nan
## arcsin

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arcsin.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


stablehlo.atan2
## arcsinh

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arcsinh.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## arctan

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arctan.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


stablehlo.atan2
## arctan2

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arctan2.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


stablehlo.atan2
## arctanh

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.arctanh.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


[-1, 1] nan
## argmax

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.argmax.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## argmin

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.argmin.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## array_equal

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array_equal.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## array_equiv

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array_equiv.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## atleast_1d

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.atleast_1d.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## atleast_2d

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.atleast_2d.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## atleast_3d

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.atleast_3d.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## bitwise_and

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.bitwise_and.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- int16
- int32
- uint16
- uint32

## bitwise_not

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.bitwise_not.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- int16
- int32
- uint16
- uint32

## bitwise_or

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.bitwise_or.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- int16
- int32
- uint16
- uint32

## bitwise_xor

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.bitwise_xor.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- int16
- int32
- uint16
- uint32

## cbrt

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.cbrt.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


stablehlo.cbrt
## ceil

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ceil.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## conj

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.conj.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## conjugate

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.conjugate.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## copysign

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.copysign.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


shift
## cos

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.cos.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


stablehlo.cosine
## cosh

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.cosh.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## deg2rad

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.deg2rad.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32

## divide

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.divide.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## divmod

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.divmod.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## ediff1d

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ediff1d.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- int32

## equal

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.equal.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## exp

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.exp.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## exp2

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.exp2.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## expm1

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.expm1.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## fabs

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fabs.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32

## fix

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fix.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## float_power

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.float_power.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32

## floor

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.floor.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## floor_divide

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.floor_divide.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## fmax

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fmax.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## fmin

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fmin.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## fmod

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fmod.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## gcd

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.gcd.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


secret while
## greater

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.greater.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## greater_equal

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.greater_equal.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## heaviside

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.heaviside.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## hypot

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.hypot.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## i0

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.i0.html
### Status

Result is incorrect.
Please check *Note* for details.
### Note


accuracy
## imag

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.imag.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## invert

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.invert.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- int16
- int32
- uint16
- uint32

## isclose

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.isclose.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## iscomplex

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.iscomplex.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## isfinite

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.isfinite.html
### Status

Not supported by design. We couldn't fix this in near future.
Please check *Note* for details.
## isinf

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.isinf.html
### Status

Not supported by design. We couldn't fix this in near future.
Please check *Note* for details.
## isnan

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.isnan.html
### Status

Not supported by design. We couldn't fix this in near future.
Please check *Note* for details.
## isneginf

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.isneginf.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32

## isposinf

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.isposinf.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32

## isreal

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.isreal.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## isrealobj

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.isrealobj.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## kron

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.kron.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## lcm

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.lcm.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


secret while
## ldexp

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ldexp.html
### Status

Result is incorrect.
Please check *Note* for details.
### Note


IEEE-754
## left_shift

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.left_shift.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- int16
- int32
- uint16
- uint32

## less

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.less.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## less_equal

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.less_equal.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## log

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.log.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## log10

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.log10.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## log1p

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.log1p.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## log1p

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.log1p.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## log2

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.log2.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## logaddexp

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.logaddexp.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32

## logaddexp2

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.logaddexp2.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32

## logical_and

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.logical_and.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## logical_not

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.logical_not.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## logical_or

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.logical_or.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## logical_xor

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.logical_xor.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## maximum

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.maximum.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## mean

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.mean.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## minimum

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.minimum.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## mod

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.mod.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## modf

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.modf.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## multiply

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.multiply.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## nanargmax

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nanargmax.html
### Status

Not supported by design. We couldn't fix this in near future.
Please check *Note* for details.
## nanargmin

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nanargmin.html
### Status

Not supported by design. We couldn't fix this in near future.
Please check *Note* for details.
## nanmean

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nanmean.html
### Status

Not supported by design. We couldn't fix this in near future.
Please check *Note* for details.
## nanprod

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nanprod.html
### Status

Not supported by design. We couldn't fix this in near future.
Please check *Note* for details.
## nansum

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nansum.html
### Status

Not supported by design. We couldn't fix this in near future.
Please check *Note* for details.
## negative

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.negative.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## nextafter

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nextafter.html
### Status

Result is incorrect.
Please check *Note* for details.
### Note


IEEE-754
## not_equal

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.not_equal.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## outer

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.outer.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## polyval

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.polyval.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## positive

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.positive.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## power

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.power.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## prod

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.prod.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## rad2deg

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.rad2deg.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32

## ravel

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ravel.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## real

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.real.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## reciprocal

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.reciprocal.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32

## remainder

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.remainder.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## right_shift

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.right_shift.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- int16
- int32
- uint16
- uint32

## rint

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.rint.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


stablehlo.round_nearest_even
## sign

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sign.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## signbit

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.signbit.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## sin

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sin.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


stablehlo.sine
## sinc

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sinc.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


stablehlo.sine
## sinh

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sinh.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


stablehlo.cosine
## sqrt

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sqrt.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## square

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.square.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## subtract

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.subtract.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## sum

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sum.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- int16
- int32
- uint16
- uint32

## tan

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tan.html
### Status

Not supported by compiler or runtime. But we could implement on demand in future.
Please check *Note* for details.
### Note


stablehlo.sine
## tanh

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tanh.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## transpose

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.transpose.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- bool_
- float32
- int16
- int32
- uint16
- uint32

## true_divide

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.true_divide.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## trunc

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.trunc.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
- int16
- int32
- uint16
- uint32

## unwrap

JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.unwrap.html
### Status

**PASS**
Please check *Supported Dtypes* as well.
### Supported Dtypes

- float32
