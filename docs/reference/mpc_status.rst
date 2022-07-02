Supported MPC Protocol
======================

Currently, SPU implements the following protocols.

* `ABY3 <https://eprint.iacr.org/2018/403.pdf>`_: A **honest majority** 3PC-protocol. SPU provides **semi-honest** implementation.
* `Semi2k-SPDZ* <https://eprint.iacr.org/2018/482.pdf>`_ : A **semi-honest** NPC-protocol similar to SPDZ but requires a trusted third party to generate offline randoms. By default this protocol now uses trusted first party. Hence, it should be used for debugging purposes only.
* `Cheetah* <https://eprint.iacr.org/2022/207>`_ : A fast 2pc semi-honest protocol. Since this protocol does not require a trusted third party, it requires more computation effort.

Currently, we mainly focused on bridging existing AI frameworks to SPU via XLA, an intermediate
representation where we can hook Tensorflow, Torch and Jax.

We welcome security experts to help contribute more security protocols.

Complexity
----------

.. note::
   * The complexity metrics are dumped from the source code, it's supposed to be accurate and could be used as a formal cost model.
   * In Semi2k, only *online stage* complexity is considered.

Notation:

* **n** is short for the *number of parties*.
* **k** is short for the *k in module 2^k*.

Naming convention:

* **a** for *arithmetic share*.
* **p** for *public value*.

For example, `mul_aa` is short for multiply of two arithmetic shares.

.. include:: complexity.md
   :parser: myst_parser.sphinx_
