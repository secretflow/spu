SPU Runtime
===========

Architecture
------------

Here is the big picture of SPU VM.

.. image:: ../imgs/vm_arch.png

- The top 3 blocks above *SPU VM* are applications, we could ignore them for now.
- The bottom left block is the scheduling component.
- The main block is the SPU Architecture, which is the core for secure evaluation.

Inside SPU, there are multiple layers, from bottom to up:

- **System layer** provides the basic computation and communication ability for the upper layers.
- **Crypto layer** is the key for secure computation, it's composed by 3 sub layers.

  - **Basic** or classic layer, provides classic cryptography, OT, HE also lives in this layer.
  - **Correlation** or the offline protocol layer, provides correlation like beaver triple and randbit.
  - **Protocol** or the online protocol layer, applies random correlation and runs the secure evaluation.

- **ALU layer** converts MPC protocols into a programmable machine, which has two sub layers.

  - **Ring 2^k** layer, just like normal CPU, hides cryptography layer's details and provides standard ring2k arithmetic.
  - **Fixed point** layer uses fixed point encoding to represent a fractional number and provides basic arithmetic operations over them.

- **OPS layer** is designed to be extensible, in this layer we can define multiple modules based on *ALU layer* and finally exposed to VM clients via bindings or SPU IR.

Homogeneous and Heterogeneous
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recall that SPU VM is composed of multiple physical engines, the definition of *homogeneous* and *heterogeneous* comes from *engines*' perspective.

- **Homogeneous**: a layer is *homogeneous* means that all engines run exactly the same code in this layer. The user of this layer doesn't have to distinguish between engines, they cannot and should not send/recv messages between engines, in other words, they can treat all engines the same, and programming them as one machine.
- **Heterogeneous**: in contrast, a layer is *heterogeneous* means that engines in this layer behave differently (following some protocols). The author of this layer should take care of the behavior of each engine to make things correct.

We want SPU VM to be *homogeneous*, so we can treat it as a normal virtual device when applying secure evaluation. For example, in the following computation graph, given `x`, `y`, we want to compute `f(x, y)`, the big circle represents a computing node which can evaluate f.

.. image:: ../imgs/physical_eval.svg

In secure computation mode, we have a group of servers working together to provide the functionality of `f`, as shown blow.

.. image:: ../imgs/virtual_eval.svg

The secure protocol (MPC protocol) itself is **heterogeneous**, three servers inside the big circle may behave differently, in this pic, the lower part is blue, which means three servers act and interact differently.

But they together provide a **homogeneous** interface to the upper layer, in this pic, the upper half is orange, three servers behave exactly the same, so in the whole computation DAG, the big circle could be treated as one (virtual) node.

Another reason to use **homogeneous** IR is to hide the number of parties, so the application can switch to an m-PC protocol from an n-PC protocol without code change.

One of *SPU*'s goal is to hide the heterogeneous part and expose homogeneous API.

VM Layout
---------

SPU, as a virtual device, is hosted by multiple physical devices. The relationship between physical devices and SPU is very flexible. Now let's use some examples to illustrate the possible layouts.

.. important::
   Programmers coding toward the virtual layout, the underline physical is **transparent** from the programmer's perspective. It's free to use different physical layouts, without changing a line of code.

Outsourcing
~~~~~~~~~~~

In this mode, data providers send data shares to a group of non-colluding computing providers, and computing providers cooperate to evaluate secure computation.

.. image:: ../imgs/device/outsourcing.svg

The figure to left depicts the physical layout, there are 6 physical nodes, mutually connected but untrusted to each other.

* The circle stands for data provider.
* The triangle stands for computing provider, three triangle node agree on some MPC protocol.

The figure to the right depicts the virtual layout.

* The circle has one-to-one relation to the physical nodes.
* 3 triangle node is treated as a single virtual device.

Colocated
~~~~~~~~~

In this mode, data providers also participate in the computation progress, that is, data providers are **colocated** with computing providers.

.. image:: ../imgs/device/colocated.svg

* On the left side, there are 3 physical nodes, each of which acts as data provider as well as computing provider.
* On the right side, **SPU is a pure virtual node, constructed by physical nodes**.


The number of computing nodes could be larger than that of data nodes in this mode, for example, a computing node without data source could act as a *random correlation generator*, for example:

.. image:: ../imgs/device/server_aided.svg

|

There two notable optimization in this mode.

- The **private semantic**, a computing node may have private data manipulations to accelerate MPC computation, for example, in *HESS protocol*, we can do :code:`HShare x Private` without online communication.
- The **zero share data infeed**, when a data provider tries to share data cross nodes, it can use :code:`ZeroShare + Private` trick to avoid online communication.

Hybrid
~~~~~~

This is the most general form, some data provider participate in the secure computation while others do not. 

.. image:: ../imgs/device/hybrid.svg

.. note::
  the **private semantic** and **zero share data infeed** also applies for data providers that participate in the computation.

