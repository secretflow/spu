Python API Reference
====================

Python API is used to control & access SPU, for example, to do data infeed/outfeed, to compile an XLA program to PPHlo, or to fire a PPHlo on an SPU runtime.


Runtime Setup
-------------

.. autoclass:: spu.Runtime
    :members:


Runtime IO
----------

.. autoclass:: spu.Io
    :members:

Compiler
--------

.. autofunction:: spu.compile

Simulation
----------

.. autoclass:: spu.Simulator
   :members:
   :undoc-members:

.. autofunction:: spu.sim_jax
