# Trivial Symmetric PIR

- Author(s): [Huan Zou](https://github.com/zouhuan1215) 
- Last updated:  July 29, 2023

## Abstract

This proposal implements trivial symmetric PIR:

- **Trivial** refers to that the communication complexity between the server and the client is $O(n)$, where $n$ is the number of items in the server's database. Namely, the client has to download all the database items from the server.

- **Symmetric** [[1]](#ref1) refers to that both the server's privacy is protected besides the client's privacy. That is, the server cannot learn which query the client asks for and the client cannot learn the database items beyond the ones it has asked for. 


## 1. Protocol Description

The basic idea of the protocol comes from [[2]](#ref2), which extends a PSI protocol to a labeled PSI protocol by utilizing the PSI's underlying oblivious pseudo-random function (OPRF). See the figure in the following.

![Extending PSI to Labeled PSI](https://github.com/secretflow/spu/assets/41979254/d34face4-1baa-4941-9397-02504f2dcb21)

This protocol chooses the ECDH PSI protocol from [[3]](#ref3) as the concrete PSI scheme to implement the labeled PSI. See the figure in the following.

<img width="820" alt="Labeled ECDH PSI" src="https://github.com/secretflow/spu/assets/41979254/6f19279e-ee74-4766-abd4-c03db28c791b">

In particular, this protocol does not compress the server's database items to improve communication efficiency. Because improving the communication complexity from linear to sublinear requires fully homomorphic encryption and polynomial interpolations (see [[4]](#ref4)), which may incur prohibitively expensive computation overheads.

## 2. Security Proofs

This protocol is secure in the honest-but-curious model, where the client and the server are assumed to follow the steps of the protocol honestly, but try to extract as much information as possible afterwards from the protocol transcript. 

This protocol is based on the ECDH PSI scheme proposed by [[3]](#ref3) (Figure 1), whose security proof relies on random model under a one-more gap diffie-hellman assumption.
The labeled version of the corresponding ECDH PSI scheme can be found at Section 5 of [[2]](#ref2), and whose security is shown by Theorem 3 of [[2]](#ref2). 

## 3. Protocol Implementation

The protocol implementation has the following layers:

```
[layer4] examples/trivial_spir_fullyonline // shows how to invoke the trivial_spir APIs
↑
[layer3] trivial_spir				// assembles the components to implement a pir server/client
↑			
[layer2] trivial_spir_components	// implements individual steps of trivial_spir
↑
[layer1] ecdh_oprf					// implements the ecdh oprf
```

There might be two improvements in the future: 
- replace the `ecdh oprf` with more efficient oprf (e.g., the OT-based ones). This requires changes between layer1 and layer2.
- implements pir as an online/offline paradigm. This requires changes between layer3 and layer4.

## 4. References

<a id="ref1">[1]</a>: Chengyu Lin, Zeyu Liu, Tal Malkin: XSPIR: Efficient Symmetrically Private Information Retrieval from Ring-LWE. ESORICS (1) 2022

<a id="ref2">[2]</a>: Stanislaw Jarecki, Xiaomin Liu: Fast Secure Computation of Set Intersection. SCN 2010

<a id="ref3">[3]</a>: Amanda Cristina Davi Resende, Diego F. Aranha: Faster Unbalanced Private Set Intersection. Financial Cryptography 2018

<a id="ref4">[4]</a>: Kelong Cong, Radames Cruz Moreno, Mariana Botelho da Gama, Wei Dai, Ilia Iliashenko, Kim Laine, Michael Rosenberg: Labeled PSI from Homomorphic Encryption with Reduced Computation and Communication. CCS 2021
