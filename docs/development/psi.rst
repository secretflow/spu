PSI
===

Quick start with SPU Private Set Intersection (PSI).

Supported Protocols
----------------------

.. The :spu_code_host:`ECDH-PSI </spu/blob/master/spu/psi/core/ecdh_psi.h>` is favorable if the bandwidth is the bottleneck.
.. If the computing is the bottleneck, you should try the BaRK-OPRF based
.. PSI :spu_code_host:`KKRT-PSI API </spu/blob/master/spu/psi/core/kkrt_psi.h>`.

+----------------+---------------+----------------+
| PSI protocols  | Threat Model  | Party Number   |
+================+===============+================+
| ECDH-PSI       | Semi-Honest   | 2P, >2P        |
+----------------+---------------+----------------+
| ECDH-OPRF-PSI  | Semi-Honest   | 2P             |
+----------------+---------------+----------------+
| `KKRT`_        | Semi-Honest   | 2P             |
+----------------+---------------+----------------+
| `Mini-PSI`_    | Malicious     | 2P             |
+----------------+---------------+----------------+

MPC and PSI protocols are designed for specific Security model (or Threat Models). 

Security model are widely considered to capture the capabilities of adversaries. 
Adversaries of semi-honest model and malicious model are Semi-honest Adversary and
Malicious Adversary. 

- `Semi-honest Adversary <https://wiki.mpcalliance.org/semi_honest_adversary.html>`_
- `Malicious Adversary <https://wiki.mpcalliance.org/malicious_adversary.html>`_

Semi-Honest PSI Must not be used in Malicious environment, may be attacked and leak information.

Our implementation of ECDH-PSI protocol supports multiple ECC curves:

- `Curve25519 <https://en.wikipedia.org/wiki/Curve25519>`_
- `Secp256k1 <https://en.bitcoin.it/wiki/Secp256k1>`_
- `FourQ <https://en.wikipedia.org/wiki/FourQ>`_
- `SM2 <https://www.cryptopp.com/wiki/SM2>`_

.. _KKRT: https://eprint.iacr.org/2016/799.pdf
.. _Mini-PSI: https://eprint.iacr.org/2021/1159.pdf

PSI Example
------------

In-memory psi example
>>>>>>>>>>>>>>>>>>>>>

A simple in-memory psi example. 

.. literalinclude:: ../../examples/cpp/simple_in_memory_psi.cc
  :language: cpp

Streaming psi example
>>>>>>>>>>>>>>>>>>>>>

A streaming example where users could perform PSI for billion items.
Read data from in_path, and Write PSI result to out_path.
Select psi protocol from ecdh/kkrt. 

:spu_code_host:`C++ streaming psi example </spu/blob/master/examples/cpp/simple_psi.cc>` .

:spu_code_host:`Python streaming psi example </spu/blob/master/examples/python/simple_psi.py>` .

How To Run
----------

Run In-memory C++ PSI example
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Start two terminals.

In the first terminal.

.. code-block:: bash

  bazel run //examples/cpp:simple_in_memory_psi -- -rank 0 -data_size 1000

In the second terminal.

.. code-block:: bash

  bazel run //examples/cpp:simple_in_memory_psi -- -rank 1 -data_size 1000

Run Streaming C++ PSI example
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Start two terminals.

In the first terminal.

.. code-block:: bash

  bazel run //examples/cpp:simple_psi -- -rank 0 -protocol ecdh -in_path ./examples/cpp/data/psi_1.csv -field_names id -out_path /tmp/p1.out 

In the second terminal.

.. code-block:: bash

  bazel run //examples/cpp:simple_psi -- -rank 1 -protocol ecdh -in_path ./examples/cpp/data/psi_2.csv -field_names id -out_path /tmp/p2.out 

Run Streaming Python PSI example
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Start two terminals.

In the first terminal.

.. code-block:: bash

  python3 ./examples/python/simple_psi.py --rank 0 --protocol ecdh --in_path ./examples/cpp/data/psi_1.csv --field_names id --out_path /tmp/p1.out 

In the second terminal.

.. code-block:: bash

  python3 ./examples/python/simple_psi.py --rank 1 --protocol ecdh --in_path ./examples/cpp/data/psi_2.csv --field_names id --out_path /tmp/p2.out 

Benchmark
----------

benchmark result without data load time 

ecdh-psi Benchmark
>>>>>>>>>>>>>>>>>>

:spu_code_host:`DH-PSI benchmark code </spu/blob/master/spu/psi/core/ecdh_psi_bench.cc>`

cpu limited by docker(--cpu)

+---------------------------+-----+--------+--------+---------+---------+---------+
| Intel(R) Xeon(R) Platinum | cpu | 2^20   | 2^21   | 2^22    | 2^23    | 2^24    |
+===========================+=====+========+========+=========+=========+=========+
|                           | 4c  | 40.181 | 81.227 | 163.509 | 330.466 | 666.807 |
|  8269CY CPU @ 2.50GHz     +-----+--------+--------+---------+---------+---------+
|                           | 8c  | 20.682 | 42.054 | 85.272  | 173.836 | 354.842 |
|  with curve25519-donna    +-----+--------+--------+---------+---------+---------+
|                           | 16c | 11.639 | 23.670 | 48.965  | 100.903 | 208.156 |
+---------------------------+-----+--------+--------+---------+---------+---------+

`ipp-crypto Multi-buffer Functions <https://www.intel.com/content/www/us/en/develop/documentation/ipp-crypto-reference/top/multi-buffer-cryptography-functions/montgomery-curve25519-elliptic-curve-functions.html>`_


+---------------------------+-----+--------+--------+---------+---------+---------+
| Intel(R) Xeon(R) Platinum | cpu | 2^20   | 2^21   | 2^22    | 2^23    | 2^24    |
+===========================+=====+========+========+=========+=========+=========+
|                           | 4c  | 7.37   | 15.32  | 31.932  | 66.802  | 139.994 |
|  8369B CPU @ 2.70GHz      +-----+--------+--------+---------+---------+---------+
|                           | 8c  | 4.3    | 9.095  | 18.919  | 40.828  | 87.649  |
|  curve25519(ipp-crypto)   +-----+--------+--------+---------+---------+---------+
|                           | 16c | 2.921  | 6.081  | 13.186  | 29.614  | 65.186  |
+---------------------------+-----+--------+--------+---------+---------+---------+

kkrt-psi Benchmark
>>>>>>>>>>>>>>>>>>>

All of our experiments use a single thread for each party. 

If the bandwidth is enough, the upstream could try to perform multi-threading optimizations

bandwidth limited by `wondershaper <https://github.com/magnific0/wondershaper/>`_.

10Mbps = 10240Kbps, 100Mbps = 102400Kbps, 1000Mbps = 1024000Kbps

.. code-block:: bash

  wondershaper -a lo -u 10240

Intel(R) Xeon(R) Platinum 8269CY CPU @ 2.50GHz

+-----------+---------+--------+--------+--------+---------+
| bandwidth |  phase  |  2^18  |  2^20  |  2^22  |  2^24   |
+===========+=========+========+========+========+=========+
|           | offline | 0.012  | 0.012  | 0.012  | 0.014   |
|    LAN    +---------+--------+--------+--------+---------+
|           | online  | 0.495  | 2.474  | 10.765 | 44.368  |
+-----------+---------+--------+--------+--------+---------+
|           | offline | 0.012  | 0.012  | 0.024  | 0.014   |
|  100Mbps  +---------+--------+--------+--------+---------+
|           | online  | 2.694  | 11.048 | 46.983 | 192.37  |
+-----------+---------+--------+--------+--------+---------+
|           | offline | 0.016  | 0.019  | 0.0312 | 0.018   |
|  10Mbps   +---------+--------+--------+--------+---------+
|           | online  | 25.434 | 100.68 | 415.94 | 1672.21 |
+-----------+---------+--------+--------+--------+---------+

Warning: In `KKRT16 <https://eprint.iacr.org/2016/799.pdf>` paper(Figure 3), the last step receiver 
directly send psi intersection result to sender. This may be attacked in malicious model,
so We suggest use KKRT16 psi protocol as one-way PSI, ie. one party get the final intersection result.
