Frequently Asked Questions (FAQ)
================================

Installation
------------

When I install from PyPI, it complains "Could not find a version that satisfies the requirement".
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We have only uploaded SPU binaries with limited version.
Please check https://pypi.org/project/spu/#files to confirm whether your environment meets the requirement of tags.
Please refer to https://github.com/pypa/manylinux to check the tags.


Usage
-----

How can I check logs of SPU?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You have to enable these flags in :ref:`/reference/runtime_config.md` when you start SPU cluster:

- enable_action_trace
- enable_pphlo_trace


General
-------

Does SPU support PyTorch?
~~~~~~~~~~~~~~~~~~~~~~~~~~

And this moment, we only ship SPU with JAX support. But theoretically, all the frontend languages which could be transferred into XLA could be
consumed by SPU compiler. In the near future, SPU team is not going to support other frontend languages however.

I have met huge inconsistency between SPU result and Plaintext(JAX) result.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Arithmetic operation of SPU is based on Fxp. Please check :ref:`reference/fxp:pitfalls - fxp arithmetic`. In most cases, you have 
to choose:

1. Use large field or modify fraction bits.
2. Modify arithmetic ops approximation approach.

But there's no such thing as a free lunch. More accurate result sometimes requires a huger cost.

Could I convert any Jax code to XLA and run by SPU?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Well, first you must make sure your Jax code is **jitable**. You have to apply some tricks to achieve this actually.
Then even your code is jitable, sometime we still will disappoint you since we haven't implemented all XLA ops. Please
check :ref:`/reference/xla_status.md`. We are working hard to finish them, you have my word!

