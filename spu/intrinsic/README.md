# SPU Intrinsic

## How to create a new intrinsic function

To add a new intrinsic, please follow steps here

### Generate necessary boilerplate

Simply using run `add_new_intrinsic.py`

```python
python ./add_new_intrinsic.py new_intrinsic
```

Replace `new_intrinsic` with the name of intrinsic to be added

### Filling python part

Open `new_intrinsic_impl.py` file under this folder, and provide necessary implementation to all unimplemented function.

In `_example_lowering` function, the prepopulated lowering may not fit the new intrinsic, make modifications when necessary.

### Add C++ dispatch code

Open `libspu/device/pphlo/pphlo_intrinsic_executor.cc`, find the if branch that matches your intrinsic name, fill proper dispatch code.

### (Optional) Add compile time visibility inference rule

By default, compiler uses a common inference rule to deduce output visibility.
If your new intrinsic requires some special treatment, please open `libspu/dialect/pphlo/transforms/visibility_inference.cc`,
and update `inferIntrinsic` method.
