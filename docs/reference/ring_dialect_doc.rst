Ring Dialect reference
===================

Ring a low-level assembly language of SPU. It assumes the IR has already lowered to fixed-point numbers and all values can be represented with an integer based ring.

Ring IR is built on `MLIR <https://mlir.llvm.org/>`_ infrastructure, the concrete ops definition could be found :spu_code_host:`here <spu/blob/main/libspu/dialect/ring/IR/ops.td>`.

Op List
~~~~~~~

.. include:: ring_op_doc.md
   :parser: myst_parser.sphinx_
