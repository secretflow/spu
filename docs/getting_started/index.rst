.. _getting_started:

Getting started
===============

First steps
-----------

New to SPU? This is the Beginner's Guide.

If you don't know anything about SPU, please check :ref:`getting_started/introduction:what is spu?` first.

After that, follow :ref:`getting_started/install:Installation Guidelines` to install SPU and :doc:`/tutorials/quick_start` to go through a simple example.

Finally, Check other examples in :ref:`/tutorials/index.rst`.


Getting help
------------

If you meet any issue, please check :ref:`getting_started/faq:frequently asked questions (faq)` and then contact us with :ref:`/getting_started/issue.rst`.


Advanced
--------

For all SPU users, please check :ref:`/reference/index.rst` for details.

For advanced users and contributors, we encourage you to investigate :ref:`/development/index.rst`.

Topics
------

Compiler
~~~~~~~~

**Design**: :doc:`/development/basic_concepts` | :doc:`/development/compiler` | :doc:`/development/ir_dump`

**Reference**: :ref:`/reference/py_api.rst#compiler` | :doc:`/reference/np_op_status` | :doc:`/reference/xla_status`


Runtime
~~~~~~~

**Design**: :doc:`/development/basic_concepts` | :doc:`/development/runtime` | :doc:`/development/type_system`

**Reference**: :ref:`/reference/py_api.rst#spu.Runtime` | :ref:`/reference/py_api.rst#spu.Io` | :doc:`/reference/pphlo_doc` | :doc:`/reference/mpc_status`

**Advanced**: :doc:`/development/policy_sgd_insight` | :doc:`/development/fxp`

**Config**: :doc:`/reference/runtime_config`

**Extending**: :doc:`/development/add_protocols`


.. toctree::
   :hidden:

   introduction
   install
   issue
   faq
