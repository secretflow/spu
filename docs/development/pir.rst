PIR QuickStart
===============

Quick start with SPU Private Information Retrival (PIR).

Supported Protocols
-------------------

+---------------+--------------+---------------+
| PIR protocols | Type         | Server Number |
+===============+==============+===============+
| SealPIR       | IndexPIR     | Single Server |
+---------------+--------------+---------------+
| Labeled PS    |KeywordPIR    | Single Server |
+---------------+--------------+---------------+


Run keyword PIR c++ example
---------------------------

First build pir examples.

.. code-block:: bash

  bazel build //examples/cpp/pir -c opt

setup phase
>>>>>>>>>>>

Start server's terminal.


.. code-block:: bash

  ./keyword_pir_setup -in_path psi_server_data.csv -oprfkey_path secret_key.bin \\
      -key_columns id -label_columns label -data_per_query 256 -label_max_len 40  \\
      -out_path pir_setup_dir -params_path psi_params.bin

query phase
>>>>>>>>>>>

Start two terminals.

In the server's terminal.

.. code-block:: bash

  keyword_pir_server -rank 0 -setup_path pir_setup_dir  \\
         -oprfkey_path secret_key.bin -data_per_query 256 -label_max_len 40  \\
         -params_path psi_params.bin -label_columns label 

In the client's terminal.

.. code-block:: bash

  ./keyword_pir_client -rank 1 -in_path psi_client_data.csv.csv  \\
        -key_columns id -data_per_query 256 -out_path pir_out.csv  



PIR query results write to pir_out.csv.
