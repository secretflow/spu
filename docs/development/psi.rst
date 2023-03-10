PSI QuickStart
===============

Quick start with SPU Private Set Intersection (PSI).

Supported Protocols
----------------------

.. The :spu_code_host:`ECDH-PSI </spu/blob/master/spu/psi/core/ecdh_psi.h>` is favorable if the bandwidth is the bottleneck.
.. If the computing is the bottleneck, you should try the BaRK-OPRF based
.. PSI :spu_code_host:`KKRT-PSI API </spu/blob/master/spu/psi/core/kkrt_psi.h>`.

+---------------+--------------+--------------+--------------+
| PSI protocols | Threat Model | Party Number |  PsiTypeCode |
+===============+==============+==============+==============+
| ECDH-PSI      | Semi-Honest  | 2P           |   1          |
+---------------+--------------+--------------+--------------+
| ECDH-OPRF-PSI | Semi-Honest  | 2P           |   -          |
+---------------+--------------+--------------+--------------+
| `KKRT`_       | Semi-Honest  | 2P           |   2          |
+---------------+--------------+--------------+--------------+
| `PCG_PSI`_    | Semi-Honest  | 2P           |   3          |
+---------------+--------------+--------------+--------------+
| `Mini-PSI`_   | Malicious    | 2P           |   -          |
+---------------+--------------+--------------+--------------+
| `DP-PSI`_     | Semi-Honest  | 2P           |   -          |
+---------------+--------------+--------------+--------------+

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

.. _PCG_PSI: https://eprint.iacr.org/2022/334.pdf
.. _KKRT: https://eprint.iacr.org/2016/799.pdf
.. _Mini-PSI: https://eprint.iacr.org/2021/1159.pdf
.. _DP-PSI: https://arxiv.org/pdf/2208.13249.pdf

Please check :ref:`/development/psi_protocol_intro.rst` for details.

PSI Example
------------

In-memory psi example
>>>>>>>>>>>>>>>>>>>>>

A simple in-memory psi example. 

:spu_code_host:`C++ simple in-memory psi example </spu/blob/master/examples/cpp/simple_in_memory_psi.cc>` .  

Streaming psi example
>>>>>>>>>>>>>>>>>>>>>

A streaming example where users could perform PSI for billion items.
Read data from in_path, and Write PSI result to out_path.
Select psi protocol from ecdh/kkrt. 

:spu_code_host:`C++ streaming psi example </spu/blob/master/examples/cpp/simple_psi.cc>` .

:spu_code_host:`Python streaming psi example </spu/blob/master/examples/python/psi/simple_psi.py>` .

dp psi example
>>>>>>>>>>>>>>>>>>>>>
example for dp psi.

:spu_code_host:`dp psi example </spu/blob/master/examples/cpp/simple_dp_psi.cc>` .

How To Run
----------

Run In-memory C++ PSI example
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

First build simple_in_memory_psi.

.. code-block:: bash

  bazel build //examples/cpp:simple_in_memory_psi

Start two terminals.

In the first terminal.

.. code-block:: bash

  simple_in_memory_psi -- -rank 0 -data_size 1000

In the second terminal.

.. code-block:: bash

  simple_in_memory_psi -- -rank 1 -data_size 1000

Run Streaming C++ PSI example
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Start two terminals.

Taking ECDH-PSI as an example, protocol is 1. KKRT-PSI sets protocol to 2, BC22 PCG-PSI sets protocol to 3.

Get PSI result in rank 0.

First build simple_psi.

.. code-block:: bash
  
  bazel build //examples/cpp:simple_psi

In the first terminal.

.. code-block:: bash

  simple_psi -rank 0 -protocol 1 -in_path ./examples/data/psi_1.csv -field_names id -out_path /tmp/p1.out 

In the second terminal.

.. code-block:: bash

  simple_psi -rank 1 -protocol 1 -in_path ./examples/data/psi_2.csv -field_names id -out_path /tmp/p2.out 

Run Streaming Python PSI example
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

PSI protocols support ECDH_PSI_2PC, KKRT_PSI_2PC, and BC22_PSI_2PC.

Get PSI result in rank 0.

Start two terminals.

In the first terminal.

.. code-block:: bash

  python3 ./examples/python/psi/simple_psi.py --rank 0 --protocol ECDH_PSI_2PC --in_path ./examples/data/psi_1.csv --field_names id --out_path /tmp/p1.out 

In the second terminal.

.. code-block:: bash

  python3 ./examples/python/psi/simple_psi.py --rank 1 --protocol ECDH_PSI_2PC --in_path ./examples/data/psi_2.csv --field_names id --out_path /tmp/p2.out 

Run DP PSI c++ example
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

First build simple_dp_psi.

.. code-block:: bash

  bazel build //examples/cpp:simple_dp_psi

Start two terminals.

In the first terminal.

.. code-block:: bash

  simple_dp_psi -rank 0 -in_path ./examples/data/psi_1.csv -field_names id  

In the second terminal.

.. code-block:: bash

  simple_dp_psi -rank 1 -in_path ./examples/data/psi_2.csv -field_names id -out_path /tmp/p1.out 

Run Unbalanced PSI python example
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Start two terminals.

In the first terminal.

.. code-block:: bash

  python3 ./examples/python/psi/unbalanced_psi.py --rank 1 --in_path ./examples/data/psi_1.csv --field_names id  

In the second terminal.

.. code-block:: bash

  python3 ./examples/python/psi/unbalanced_psi.py --rank 0 -in_path ./examples/data/psi_2.csv -field_names id -out_path /tmp/p1.out 


Benchmark
----------

benchmark result without data load time 

ecdh-psi Benchmark
>>>>>>>>>>>>>>>>>>

:spu_code_host:`DH-PSI benchmark code </spu/blob/master/spu/psi/core/ecdh_psi_bench.cc>`

cpu limited by docker(--cpu)

+---------------------------+-----+---------+---------+----------+----------+----------+
| Intel(R) Xeon(R) Platinum | cpu |  2^20   |  2^21   |  2^22    |  2^23    |  2^24    |
+===========================+=====+=========+=========+==========+==========+==========+
|                           | 4c  | 40.181s | 81.227s | 163.509s | 330.466s | 666.807s |
|  8269CY CPU @ 2.50GHz     +-----+---------+---------+----------+----------+----------+
|                           | 8c  | 20.682s | 42.054s | 85.272s  | 173.836s | 354.842s |
|  with curve25519-donna    +-----+---------+---------+----------+----------+----------+
|                           | 16c | 11.639s | 23.670s | 48.965s  | 100.903s | 208.156s |
+---------------------------+-----+---------+---------+----------+----------+----------+

`ipp-crypto Multi-buffer Functions <https://www.intel.com/content/www/us/en/develop/documentation/ipp-crypto-reference/top/multi-buffer-cryptography-functions/montgomery-curve25519-elliptic-curve-functions.html>`_


+---------------------------+-----+--------+--------+---------+---------+----------+
| Intel(R) Xeon(R) Platinum | cpu | 2^20   | 2^21   | 2^22    | 2^23    |   2^24   |
+===========================+=====+========+========+=========+=========+==========+
|                           | 4c  | 7.37s  | 15.32s | 31.932s | 66.802s | 139.994s |
|  8369B CPU @ 2.70GHz      +-----+--------+--------+---------+---------+----------+
|                           | 8c  | 4.3s   | 9.095s | 18.919s | 40.828s | 87.649s  |
|  curve25519(ipp-crypto)   +-----+--------+--------+---------+---------+----------+
|                           | 16c | 2.921s | 6.081s | 13.186s | 29.614s | 65.186s  |
+---------------------------+-----+--------+--------+---------+---------+----------+

kkrt-psi Benchmark
>>>>>>>>>>>>>>>>>>>

All of our experiments use a single thread for each party. 

If the bandwidth is enough, the upstream could try to perform multi-threading optimizations

bandwidth limited by `wondershaper <https://github.com/magnific0/wondershaper/>`_.

10Mbps = 10240Kbps, 100Mbps = 102400Kbps, 1000Mbps = 1024000Kbps

.. code-block:: bash

  wondershaper -a lo -u 10240

Intel(R) Xeon(R) Platinum 8269CY CPU @ 2.50GHz

+-----------+---------+---------+---------+---------+----------+
| bandwidth |  phase  |   2^18  |   2^20  |   2^22  |   2^24   |
+===========+=========+=========+=========+=========+==========+
|           | offline | 0.012s  | 0.012s  | 0.012s  | 0.014s   |
|    LAN    +---------+---------+---------+---------+----------+
|           | online  | 0.495s  | 2.474s  | 10.765s | 44.368s  |
+-----------+---------+---------+---------+---------+----------+
|           | offline | 0.012s  | 0.012s  | 0.024s  | 0.014s   |
|  100Mbps  +---------+---------+---------+---------+----------+
|           | online  | 2.694s  | 11.048s | 46.983s | 192.37s  |
+-----------+---------+---------+---------+---------+----------+
|           | offline | 0.016s  | 0.019s  | 0.0312s | 0.018s   |
|  10Mbps   +---------+---------+---------+---------+----------+
|           | online  | 25.434s | 100.68s | 415.94s | 1672.21s |
+-----------+---------+---------+---------+---------+----------+

bc22 pcg-psi Benchmark
>>>>>>>>>>>>>>>>>>>>>>

Intel(R) Xeon(R) Platinum 8269CY CPU @ 2.50GHz

+-----------+---------+---------+---------+----------+---------+---------+
| bandwidth |   2^18  |   2^20  |   2^21  |   2^22   |   2^23  |   2^24  |
+===========+=========+=========+=========+==========+=========+=========+
|    LAN    | 1.261s  | 2.191s  | 3.503s  | 6.51s    | 13.012s | 26.71s  |
+-----------+---------+---------+---------+----------+---------+---------+
|  100Mbps  | 2.417s  | 6.054s  | 11.314s | 21.864s  | 43.778s | 88.29s  |
+-----------+---------+---------+---------+----------+---------+---------+
|   10Mbps  | 18.826s | 50.038s | 96.516s | 186.097s | 369.84s | 737.71s |
+-----------+---------+---------+---------+----------+---------+---------+


Security Tips
-------------

Warning:  `KKRT16 <https://eprint.iacr.org/2016/799.pdf>`_ and 
`BC22 PCG <https://eprint.iacr.org/2022/334.pdf>`_ are semi-honest PSI protocols, 
and may be attacked in malicious model.
We recommend using KKRT16 and BC22_PCG PSI protocol as one-way PSI, i.e., one party gets the final intersection result.
