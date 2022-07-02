C++ Example: Logistic Regression
================================

To use SPU C++ API, we have to first :ref:`getting_started/install:build from source`, this document shows how to write a privacy preserving logistic regression program with SPU C++ API.


Logistic Regression
-------------------

.. literalinclude:: ../../../examples/cpp/simple_lr.cc
  :language: cpp

.. todo:: could you help to add comments for *simple_lr.cc* @shantang


Run it
------

Start two terminals.

In the first terminal.

.. code-block:: bash

  bazel run //examples/cpp:simple_lr -- -rank 0 -dataset examples/cpp/data/perfect_logit_a.csv -has_label=true

In the second terminal.

.. code-block:: bash

  bazel run //examples/cpp:simple_lr -- -rank 1 -dataset examples/cpp/data/perfect_logit_b.csv

