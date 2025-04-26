C++ Example: Logistic Regression
================================

To use SPU C++ API, we have to first :ref:`build from source<getting_started/install:Building From Source>`, this document shows how to write a privacy preserving logistic regression program with SPU C++ API.


Logistic Regression
-------------------

The following code is located in `examples/cpp/simple_lr.cc`.

.. literalinclude:: ../../examples/cpp/simple_lr.cc
  :language: cpp


Run it
------

Start two terminals from the root of the project.

In the first terminal.

.. code-block:: bash

  bazelisk run //examples/cpp:simple_lr -- -rank 0 -dataset examples/data/perfect_logit_a.csv -has_label=true

In the second terminal.

.. code-block:: bash

  bazelisk run //examples/cpp:simple_lr -- -rank 1 -dataset examples/data/perfect_logit_b.csv
