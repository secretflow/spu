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

  bazel build //examples/cpp/pir/... -c opt

setup phase
>>>>>>>>>>>

Generate test usage oprf_key.bin

.. code-block:: bash

    dd if=/dev/urandom of=oprf_key.bin bs=32 count=1

Start server's terminal.

.. code-block:: bash

  ./bazel-bin/examples/cpp/pir/keyword_pir_setup -in_path examples/data/pir_server_data.csv \\
      -key_columns id -label_columns label -count_per_query 256 -max_label_length 40  \\
      -oprf_key_path oprf_key.bin -setup_path pir_setup_dir 

query phase
>>>>>>>>>>>

Start two terminals.

In the server's terminal.

.. code-block:: bash

  ./bazel-bin/examples/cpp/pir/keyword_pir_server -rank 0 -setup_path pir_setup_dir  \\
         -oprf_key_path oprf_key.bin
         

In the client's terminal.

.. code-block:: bash

  ./bazel-bin/examples/cpp/pir/keyword_pir_client -rank 1 \\
        -in_path examples/data/pir_client_data.csv.csv  \\
        -key_columns id -out_path pir_out.csv  

PIR query results write to pir_out.csv.
Run examples on two host, Please add '-parties ip1:port1,ip2:port2'.

Run keyword PIR python example
---------------------------

First build spu python whl package or install from network.

.. code-block:: bash

  bash build_wheel_entrypoint.sh

install dist/spu-*.whl 

setup phase
>>>>>>>>>>>

Start server's terminal.


.. code-block:: bash

  python examples/python/pir/pir_setup.py --in_path examples/data/pir_server_data.csv \\
      --oprf_key_path oprf_key.bin  --key_columns id --label_columns label \\
      --count_per_query 256 --max_label_length 40  \\
      --setup_path pir_setup_dir 

query phase
>>>>>>>>>>>

Start two terminals.

In the server's terminal.

.. code-block:: bash

  python examples/python/pir/pir_server.py --rank 0 --setup_path pir_setup_dir  \\
         --oprf_key_path oprf_key.bin 

In the client's terminal.

.. code-block:: bash

  python examples/python/pir/pir_client.py -rank 1  \\
        -in_path examples/data/pir_client_data.csv.csv \\
        -key_columns id -out_path pir_out.csv  

PIR query results write to pir_out.csv.
Run examples on two host, Please add '--party_ips ip1:port1,ip2:port2'.
