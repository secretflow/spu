Adding New MPC Protocols
========================

.. warning::
   SecretFlow SPU currently is under active development and the APIs provided to protocol developers may be unstable at this moment.

Introduction
------------
This document is mainly for developers who want to add custom MPC protocols in SPU.
Before reading this document, we recommend that developers have a basic understanding
of the SPU system architecture (i.e., :doc:`/development/compiler`, :doc:`/development/type_system` and :doc:`/development/runtime`)
and the `layout <https://github.com/secretflow/spu/blob/main/REPO_LAYOUT.md>`_ of the SPU code repository.
In short, SPU translates the high-level applications (such as machine learning model training) written in JAX
to an MPC-specific intermediate representation named PPHLO and then dispatches PPHLO operations to the low-level MPC protocols.
In theory, protocol developers only need to implement a basic set of MPC operation APIs to fully use the SPU infrastructure to
run machine learning or data analysis programs. That is to say, for most MPC protocol development,
it only needs to add some source code files into the libspu/mpc folder.
At present, SPU has integrated several protocols such as ABY3 and Cheetah,
which can be regarded as a guide for protocol implementation.

A walk-through guide
--------------------
We further illustrate the procedures of adding protocols step by step in a **top-down** manner.

Add a new protocol kind
~~~~~~~~~~~~~~~~~~~~~~~
When users launch the SPU backend runtime, they will specify the running MPC protocol kind
through the runtime config. The protocol kinds supported by SPU are enumerations defined
in `spu.proto <https://github.com/secretflow/spu/blob/main/src/libspu/spu.proto>`_. Thus,
developers must add their protocol kinds in this protobuf file to enable SPU to be aware
of the new protocols.

.. code-block:: protobuf
  :caption: ProtocolKind enumerations

  enum ProtocolKind {
    PROT_INVALID = 0;

    REF2K = 1;

    SEMI2K = 2;

    ABY3 = 3;

    CHEETAH = 4;
  }

Register protocol
~~~~~~~~~~~~~~~~~
Each MPC protocol execution environment is abstracted as a C++ instance of an `Object <https://github.com/secretflow/spu/blob/main/src/libspu/core/object.h>`_
class in SPU. SPU constructs an MPC object when creating an **SPUContext**. Then, SPU registers a concrete protocol implementation through a factory function
named `RegisterProtocol <https://github.com/secretflow/spu/blob/main/src/libspu/mpc/factory.cc>`_ according to the runtime config. Therefore, protocol developers
need to add their functions in **RegisterProtocol** to implement protocols.

.. code-block:: c++
  :caption: RegisterProtocol function

  void Factory::RegisterProtocol(
    SPUContext* ctx, const std::shared_ptr<yacl::link::Context>& lctx) {
    // TODO: support multi-protocols.
    switch (ctx->config().protocol()) {
      ...
      ...
      case ProtocolKind::ABY3: {
        return regAby3Protocol(ctx, lctx);
      }
      case ProtocolKind::CHEETAH: {
        return regCheetahProtocol(ctx, lctx);
      }
      default: {
        SPU_THROW("Invalid protocol kind {}", ctx->config().protocol());
      }
    }
  }

Implement protocol IO interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Another function called by the factory class is the **CreateIO** function. As different protocols use different secret sharing schemas,
which means SPU has to use different ways to input/output secret data from plaintext data. As a results, developers have to implement these protocol-specific APIs
defined in `io_interface.h <https://github.com/secretflow/spu/blob/main/src/libspu/mpc/io_interface.h>`_.
Developers can check the `ABY3 implementation <https://github.com/secretflow/spu/blob/main/src/libspu/mpc/aby3/io.cc>`_ as a reference.

Understand protocol object
~~~~~~~~~~~~~~~~~~~~~~~~~~
SPU protocol `Object <https://github.com/secretflow/spu/blob/main/src/libspu/core/object.h>`_
may be the key concept for adding new protocols. Let's take a closer look at its design.
The goal of **Object** class is to realize the generalization and flexibility of developing MPC protocols through dynamic binding.
An Object instance has a series of kernels and states. A kernel and a state can be regarded as a
member function and a member variable of an Object, respectively.

.. code-block:: c++
  :caption: SPU protocol Object class

  class Object final {
    std::map<std::string, std::shared_ptr<Kernel>> kernels_;
    std::map<std::string, std::unique_ptr<State>> states_;
    ...

  public:
    explicit Object(std::string id, std::string pid = "")
        : id_(std::move(id)), pid_(std::move(pid)) {}
    ...
    ...

    // register customized kernels
    template <typename KernelT>
    void regKernel() {
      regKernel(KernelT::kBindName(), std::make_unique<KernelT>());
    }

    template <typename KernelT>
    void regKernel(const std::string& name) {
      return regKernel(name, std::make_unique<KernelT>());
    }

    // add customized states
    template <typename StateT, typename... Args>
    void addState(Args&&... args) {
      addState(StateT::kBindName(),
              std::make_unique<StateT>(std::forward<Args>(args)...));
    }
    ...
    ...
  };

Construct protocol object
~~~~~~~~~~~~~~~~~~~~~~~~~
We take the ABY3 implementation as a specific example to further explain the description above.

First of all, we can see that there is an independent aby3 directory under the `src/libspu/mpc <https://github.com/secretflow/spu/tree/main/src/libspu/mpc>`_
directory in SPU's repository layout. The aby3 directory includes the C++ source files and header
files required by the ABY3 protocol implementation. These files may be confusing at first glance.
The key to know its code organization is to open the `protocol <https://github.com/secretflow/spu/blob/main/src/libspu/mpc/aby3/protocol.cc>`_
file, which defines the **regAby3Protocol** function for registering kernels and states.
This function will be called by the factory class described in previous step.

.. code-block:: c++
  :caption: ABY3 protocol registration

  void regAby3Protocol(SPUContext* ctx,
                     const std::shared_ptr<yacl::link::Context>& lctx) {
    // register ABY3 arithmetic shares and boolean shares
    aby3::registerTypes();

    // add ABY3 required states
    ctx->prot()->addState<Z2kState>(ctx->config().field());
    ctx->prot()->addState<Communicator>(lctx);
    ctx->prot()->addState<PrgState>(lctx);

    // register public kernels
    regPV2kKernels(ctx->prot());

    // register arithmetic & binary kernels
    ...
    ctx->prot()->regKernel<aby3::AddAP>();
    ctx->prot()->regKernel<aby3::AddAA>();
    ctx->prot()->regKernel<aby3::MulAP>();
    ctx->prot()->regKernel<aby3::MulAA>();
    ...

    return obj;
  }

Inside the **regAby3Protocol** function, it does three things.

- The first is to register the protocol types. These types are defined in the `type.h <https://github.com/secretflow/spu/blob/main/src/libspu/mpc/aby3/type.h>`_ header file, \
  representing an arithmetic secret share and a boolean secret share, respectively.

- The second is to register protocol states (variables), specifically including the three states of Z2kState, \
  Communicator, and PrgState, which are used to store the ring information, communication facilities, and \
  pseudorandom number generator for protocol implementation.

- The third is to register the protocol kernels (functions). We can see that two types of kernels are registered. \
  The first type is the common kernels implemented in the `pv2k.cc <https://github.com/secretflow/spu/blob/main/src/libspu/mpc/common/pv2k.cc>`_ \
  file. The second type is implemented in `arithmetic.cc <https://github.com/secretflow/spu/blob/main/src/libspu/mpc/aby3/arithmetic.cc>`_, \
  `boolean.cc <https://github.com/secretflow/spu/blob/main/src/libspu/mpc/aby3/boolean.cc>`_ and other files under the aby3 directory.

Implement protocol kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~
In this section, we further explain why the ABY3 developer registers these two types of kernels.
In SPU, the interfaces between MPC and HAL layers are defined in the `api.h <https://github.com/secretflow/spu/blob/main/src/libspu/mpc/api.h>`_
file, which consists of a set of operations with public or secret operands (referred as **basic APIs** for the rest of this document).
As long as a protocol developer implements basic APIs, he/she can use the SPU full-stack infrastructure
to run high-level applications, e.g., training complex neural network models.

.. code-block:: c++
  :caption: Some SPU MPC basic APIs

  ...
  Value mul_pp(SPUContext* ctx, const Value& x, const Value& y);
  Value mul_sp(SPUContext* ctx, const Value& x, const Value& y);
  Value mul_ss(SPUContext* ctx, const Value& x, const Value& y);
  Value and_pp(SPUContext* ctx, const Value& x, const Value& y);
  Value and_sp(SPUContext* ctx, const Value& x, const Value& y);
  Value and_ss(SPUContext* ctx, const Value& x, const Value& y);
  ...

Among the basic APIs, some protocols working on Rings share the same logic on some operations processing public operands,
so SPU developers pre-implement these APIs as kernels and place them in the common directory.
As a result, the ABY3 developer can directly register these kernels through the **regPV2kKernels** function.

.. code-block:: c++
  :caption: Pre-implemented *and_pp* kernel

  class AndPP : public BinaryKernel {
   public:
    // kernel name for dynamic binding
    static constexpr const char* kBindName() { return "and_pp"; }

    // define cost model
    ce::CExpr latency() const override { return ce::Const(0); }
    ce::CExpr comm() const override { return ce::Const(0); }

    // protocol implementation
    NdArrayRef proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                  const NdArrayRef& rhs) const override {
      // sanity check
      SPU_ENFORCE(lhs.eltype() == rhs.eltype());
      return ring_and(lhs, rhs).as(lhs.eltype());
    }
  };


.. code-block:: c++
  :caption: Register *and_pp* kernel in regPV2kKernels function

  ...
  obj->regKernel<MulPP>();
  obj->regKernel<MatMulPP>();
  // and_pp kernel is implemented as an AndPP class
  obj->regKernel<AndPP>();
  obj->regKernel<XorPP>();
  ...

Besides, ABY3 protocol-specific operations need to be implemented by developers as kernels to register.
For example, the multiplication of two arithmetic secret shares of ABY3 is implemented as the **MulAA** kernel located in the
`arithmetic.cc <https://github.com/secretflow/spu/blob/main/src/libspu/mpc/aby3/arithmetic.cc>`_ source file.
When kernels are implemented and registered, a new protocol is finally added.

.. code-block:: c++
  :caption: ABY3 *mul_aa* kernel for arithmetic share multiplication

  NdArrayRef MulAA::proc(KernelEvalContext* ctx, const NdArrayRef& lhs,
                         const NdArrayRef& rhs) const {
    // get required states
    const auto field = lhs.eltype().as<Ring2k>()->field();
    auto* comm = ctx->getState<Communicator>();
    auto* prg_state = ctx->getState<PrgState>();

    // dispatch the real implementation to different fields
    return DISPATCH_ALL_FIELDS(field, [&]() {
      // the real protocol implementation
      ...
    });
  }

Testing
~~~~~~~
After a protocol is added, the developer usually wants to test whether the protocol works as expected.
There are two ways to test the protocol functionality in SPU. The first way is to run python examples.
SPU has provided users with a series of application `examples <https://github.com/secretflow/spu/tree/main/examples/python>`_.
If a protocol fully implements SPU's basic APIs, the developer can run these high-level examples to verify
whether the low-level protocol development is correct.

The second way is to write and run unittest. Some protocols do not cover all the basic APIs and cannot run examples,
or developers only want to test the functionalities of some specific MPC operations (such as addition and multiplication).
In these cases it is more practical to run unittest. SPU developers have construct a general test frameworks in
`api_test.cc <https://github.com/secretflow/spu/blob/main/src/libspu/mpc/api_test.cc>`_ and
`ab_api_test.cc <https://github.com/secretflow/spu/blob/main/src/libspu/mpc/ab_api_test.cc>`_.
Developers of new protocols need to instantiate these frameworks to test their own protocol functionalities.
Developers can refer to the `protocol_test.cc <https://github.com/secretflow/spu/blob/main/src/libspu/mpc/aby3/protocol_test.cc>`_
file in the aby3 directory to learn how to write their own protocol test files.
