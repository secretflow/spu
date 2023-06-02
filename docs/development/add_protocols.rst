Adding New MPC Protocols
========================

.. warning::
   SecretFlow SPU currently is under active development and the APIs provided to protocol developers may be unstable at this moment.

Introduction
------------
This document is mainly for developers who want to add custom MPC protocols in SPU. 
Before reading this document, we recommend that developers have a basic understanding 
of the SPU system architecture (i.e., :ref:`/development/compiler.rst`, :ref:`/development/type_system.rst` and :ref:`/development/runtime.rst`) 
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
in `spu.proto <https://github.com/secretflow/spu/blob/main/libspu/spu.proto>`_. Thus, 
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

Generate protocol object
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each MPC protocol execution environment is abstracted as a C++ instance of an `Object <https://github.com/secretflow/spu/blob/main/libspu/core/object.h>`_ 
class in SPU. SPU generates an MPC object through a factory function named `CreateContext <https://github.com/secretflow/spu/blob/main/libspu/mpc/factory.h>`_
according to the runtime config. Therefore, protocol developers need to add their functions in **CreateCompute** to generate protocol objects.

.. code-block:: c++
  :caption: CreateCompute function

  std::unique_ptr<SPUContext> Factory::CreateContext(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx) {
    switch (conf.protocol()) {
      ...
      ...
      case ProtocolKind::ABY3: {
        return makeAby3Protocol(conf, lctx);
      }
      case ProtocolKind::CHEETAH: {
        return makeCheetahProtocol(conf, lctx);
      }
      default: {
        SPU_THROW("Invalid protocol kind {}", conf.protocol());
      }
    }
    return nullptr;
  }

Understand protocol object
~~~~~~~~~~~~~~~~~~~~~~~~~~
SPU protocol `Object <https://github.com/secretflow/spu/blob/main/libspu/core/object.h>`_
may be the key concept for adding new protocols. Let's take a closer look at its design.
The goal of Object class is to realize the generalization and flexibility of developing MPC protocols through dynamic binding. 
An Object instance has a series of kernels and states. A kernel and a state can be regarded as a 
member function and a member variable of an Object, respectively.

.. code-block:: c++
  :caption: SPU protocol Object class

  class Object final {
    std::map<std::string_view, std::shared_ptr<Kernel>> kernels_;
    std::map<std::string_view, std::unique_ptr<State>> states_;
    ...

  public:
    explicit Object(std::string id, std::string pid = "")
        : id_(std::move(id)), pid_(std::move(pid)) {}
    ...
    ...

    // register customized kernels
    template <typename KernelT>
    void regKernel(std::string_view name) {
      return regKernel(name, std::make_unique<KernelT>());
    }

    // add customized kernels
    template <typename StateT, typename... Args>
    void addState(Args&&... args) {
      addState(StateT::kBindName,
              std::make_unique<StateT>(std::forward<Args>(args)...));
    }
    ...
    ...
  };

Construct protocol object
~~~~~~~~~~~~~~~~~~~~~~~~~
We take the ABY3 implementation as a specific example to further explain the description above.

First of all, we can see that there is an independent aby3 directory under the `libspu/mpc <https://github.com/secretflow/spu/tree/main/libspu/mpc>`_
directory in SPU's repository layout. The aby3 directory includes the C++ source files and header 
files required by the ABY3 protocol implementation. These files may be confusing at first glance. 
The key to know its code organization is to open the `protocol <https://github.com/secretflow/spu/blob/main/libspu/mpc/aby3/protocol.cc>`_
file, which defines the **makeAby3Protocol** function for generating an Object instance. 
This function will be called by the factory class described in previous step.

.. code-block:: c++
  :caption: ABY3 protocol Object instance generation

  std::unique_ptr<Object> makeAby3Protocol(
      const RuntimeConfig& conf,
      const std::shared_ptr<yacl::link::Context>& lctx) {
    // register ABY3 arithmetic shares and boolean shares
    aby3::registerTypes();

    // instantiate Object instance  
    auto obj =
        std::make_unique<Object>(fmt::format("{}-{}", lctx->Rank(), "ABY3"));

    // add ABY3 required states
    obj->addState<Z2kState>(conf.field());
    obj->addState<Communicator>(lctx);
    obj->addState<PrgState>(lctx);

    // register public kernels and api kernels
    regPV2kKernels(obj.get());
    regABKernels(obj.get());

    // register arithmetic & binary kernels
    ...
    obj->regKernel<aby3::AddAP>();
    obj->regKernel<aby3::AddAA>();
    obj->regKernel<aby3::MulAP>();
    obj->regKernel<aby3::MulAA>();
    ...

    return obj;
  }

Inside the **makeAby3Protocol** function, it does three things. 

- The first is to register the protocol types. These types are defined in the `type.h <https://github.com/secretflow/spu/blob/main/libspu/mpc/aby3/type.h>`_ header file, \
  representing an arithmetic secret share and a boolean secret share, respectively. 

- The second is to register protocol states (variables), specifically including the three states of Z2kState, \
  Communicator, and PrgState, which are used to store the ring information, communication facilities, and \
  pseudorandom number generator for protocol implementation. 

- The third is to register the protocol kernels (functions). We can see that three types of kernels are registered. \
  The first type is the kernels implemented in the `pv2k.cc <https://github.com/secretflow/spu/blob/main/libspu/mpc/common/pv2k.cc>`_ \
  file, using **Pub2k** as the naming prefix of kernel classes. The second type is the kernels implemented in the \
  `ab_api.cc <https://github.com/secretflow/spu/blob/main/libspu/mpc/common/ab_api.cc>`_ file, using **ABProt** as the \ 
  naming prefix of kernel classes. The third type is implemented in `arithmetic.cc <https://github.com/secretflow/spu/blob/main/libspu/mpc/aby3/arithmetic.cc>`_, \
  `boolean.cc <https://github.com/secretflow/spu/blob/main/libspu/mpc/aby3/boolean.cc>`_ and other files under the aby3 directory.

Implement protocol kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~
In this section, we further explain why the ABY3 developer registers these three types of kernels. 
In SPU, the interfaces between MPC and HAL layers are defined in the `api.cc <https://github.com/secretflow/spu/blob/main/libspu/mpc/api.cc>`_
file, which consists of a set of operations with public or secret operands (referred as **basic APIs** for the rest of this document). 
As long as a protocol developer implements basic APIs, he/her can use the SPU full-stack infrastructure 
to run high-level applications, e.g., training complex neural network models.

.. code-block:: c++
  :caption: Some SPU MPC basic APIs

  ...
  ArrayRef mul_pp(Object* ctx, const ArrayRef&, const ArrayRef&);
  ArrayRef mul_sp(Object* ctx, const ArrayRef&, const ArrayRef&);
  ArrayRef mul_ss(Object* ctx, const ArrayRef&, const ArrayRef&);
  ArrayRef and_pp(Object* ctx, const ArrayRef&, const ArrayRef&);
  ArrayRef and_sp(Object* ctx, const ArrayRef&, const ArrayRef&);
  ArrayRef and_ss(Object* ctx, const ArrayRef&, const ArrayRef&);
  ...

Among the basic APIs, some protocols working on Rings share the same logic on some operations processing public operands, 
so SPU developers pre-implement these APIs as kernels and place them in the common directory. 
As a result, the ABY3 developer can directly register these kernels through the **regPV2kKernels** function.

.. code-block:: c++
  :caption: Pre-implemented *and_pp* kernel

  class AndPP : public BinaryKernel {
   public:
    // kernel name for dynamic binding
    static constexpr char kBindName[] = "and_pp";

    // define cost model
    ce::CExpr latency() const override { return ce::Const(0); }
    ce::CExpr comm() const override { return ce::Const(0); }

    // protocol implementation
    ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                  const ArrayRef& rhs) const override {
      // used for logging and tracing
      SPU_TRACE_MPC_LEAF(ctx, lhs, rhs);
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
  // and_pp kernel is implemented as a AndPP class
  obj->regKernel<AndPP>();
  obj->regKernel<XorPP>();
  ...

Moreover, as we can see that the basic APIs do not have (arithmetic/boolean) secret sharing semantics. 
A series of popular protocols (such as ABY3) use arithmetic secret sharing, boolean secret sharing, 
and their conversions to achieve secret operations defined in the basic APIs. Therefore, we call this type of protocol **AB protocols**, 
and further pre-implement the basic APIs of AB protocols as kernels by dispatching secret operations to 
arithmetic/boolean secret sharing operations. For example, the multiplication of two secret operands will be 
further decomposed into the multiplication of two arithmetic secret shares or the AND of two boolean secret shares.
As ABY3 is a type of AB protocol, the ABY3 developer can directly call the **regABKernels** function to register these kernels.

.. code-block:: c++
  :caption: Pre-implemented *mul_ss* kernel for AB protocols

  class ABProtMulSS : public BinaryKernel {
    ...
    // decompose a secret multiplication into secret share operations
    ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                  const ArrayRef& rhs) const override {
      ...
      if (_LAZY_AB) {
        if (_IsB(lhs) && _NBits(lhs) == 1 && _IsB(rhs) && _NBits(rhs) == 1) {
          return _AndBB(lhs, rhs);
        }
        return _MulAA(_2A(lhs), _2A(rhs));
      }
      return _MulAA(lhs, rhs);
    }
  };

Finally, ABY3 protocol-specific operations need to be implemented by developers as kernels to register. 
For example, the multiplication of two arithmetic secret shares of ABY3 is implemented as the **MulAA** kernel located in the 
`arithmetic.cc <https://github.com/secretflow/spu/blob/main/libspu/mpc/aby3/arithmetic.cc>`_ source file.
When kernels are implemented and registered, a new protocol is finally added.

.. code-block:: c++
  :caption: ABY3 *mul_aa* kernel for arithmetic share multiplication
  
  ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                      const ArrayRef& rhs) const {
    // get required states
    const auto field = lhs.eltype().as<Ring2k>()->field();
    auto* comm = ctx->getState<Communicator>();
    auto* prg_state = ctx->getState<PrgState>();

    // dispatch the real implementation to different fields
    return DISPATCH_ALL_FIELDS(field, "aby3.mulAA", [&]() {
    // the real protocol implementation    
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
`api_test.cc <https://github.com/secretflow/spu/blob/main/libspu/mpc/api_test.cc>`_ and 
`ab_api_test.cc <https://github.com/secretflow/spu/blob/main/libspu/mpc/ab_api_test.cc>`_. 
Developers of new protocols need to instantiate these frameworks to test their own protocol functionalities. 
Developers can refer to the `protocol_test.cc <https://github.com/secretflow/spu/blob/main/libspu/mpc/aby3/protocol_test.cc>`_ 
file in the aby3 directory to learn how to write their own protocol test files.
