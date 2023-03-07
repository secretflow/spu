PSI Protocols Introduction
==========================

SecretFlow SPU implements the following PSI protocols,

- Semi-honest ECDH-based two-party PSI protocol [HFH99]_
- Semi-honest ECDH-based three-party PSI protocol
- Semi-honest OT-based two-party PSI protocol [KKRT16]_
- Semi-honest OT-based two-party PSI protocol (with improved communication efficiency) [BC22]_
- Differentially Private (DP) PSI Protocol [DP-PSI]_
- Unbalanced PSI Protocol

ECDH-PSI
--------

The semi-honest DH-PSI protocol is due to Huberman, Franklin, and Hogg [HFH99]_, 
but with roots as far back as Meadows [Mea86]_. It is a semi-honest protocol that
requires exponentiations in a Diffie-Hellman group proportional to the number of items in the sets.

As a general rule, OT-based PSI protocols are (significantly) faster but require more communication 
than Diffie-Hellman-based PSI protocols. 
In some scenarios, communication cost is overwhelmingly more important than computation cost.

DH-PSI protocol based on the Decisional Diffie-Hellman assumption:

- Agree on a group G, with a generator g.
- The assumption: for random a,b,c cannot distinguish :math:`(g^a, g^b, g^{ab})` from :math:`(g^a, g^b, g^c)`

Several candidate groups are widely used, such as subgroups of the multiplication group of a finite
field and elliptic curve groups. In practice, carefully chosen elliptic curves like
Curve25519 [Ber06]_ offer a good balance between security and performance.

.. figure:: ../imgs/dh_psi.png

1. For each element :math:`x_i` in its set, Alice applies the hash function and then exponentiates it 
   using its key :math:`\alpha`, thus computing :math:`{H(x_i)}^\alpha` . Alice sends 
   :math:`\{\{H(x_i)\}^\alpha\}_{i=1}^{n_1}` to Bob.

2. For each element :math:`{H(x_i)}^\alpha`  received from Alice in the previous step, Bob exponentiates 
   it using its key :math:`\beta`, computing :math:`{H(x_i)}^{\alpha\beta}`. 
   Bob sends :math:`{\{\{H(x_i)\}^{\alpha\beta}\}}_{i=1}^{n_1}` to Alice.

3. For each element :math:`y_i` in its set, Bob applies the hash function and then exponentiates it 
   using its key :math:`\beta`, thus computing :math:`{H(y_i)}^\beta` . 
   Bob sends the set :math:`\{\{H(y_i)\}^\beta\}_{i=1}^{n_2}` to Alice.

4. For each element :math:`{H(y_i)}^\beta`  received from Bob in the previous step, Alice exponentiates 
   it using its key :math:`\alpha`, computing :math:`{H(y_i)}^{\beta\alpha}` .   

5. Alice compares two set :math:`{\{\{H(x_i)\}^{\alpha\beta}\}}_{i=1}^{n_1}` 
   and :math:`{\{\{H(y_i)\}^{\beta\alpha}\}}_{i=1}^{n_2}`  and gets intersection.

The Elliptic Curve groups, supported in secretflow SPU PSI module.

+-------------+------------------------+------------------------------------------------------+
| EC group    | Reference              | CryptoLib                                            |
+=============+========================+======================================================+
| Curve25519  | [Ber06]_               | `LibSoidum <https://doc.libsodium.org/>`_            |
|             |                        +------------------------------------------------------+
|             |                        | [ipp-crypto]_ (Intel® CPU support AVX-512 IFMA)      |
+-------------+------------------------+------------------------------------------------------+
| Secp256k1   | [SEC2-v2]_             | `OpenSSL <https://www.openssl.org>`_                 |
+-------------+------------------------+------------------------------------------------------+
|   SM2       | GBT.32918.1-2016       | `OpenSSL <https://www.openssl.org>`_                 |
|             +------------------------+                                                      |
|             | ISO/IEC 14888-3:2018   |                                                      |
+-------------+------------------------+------------------------------------------------------+
|   FourQ     | [FourQ]_               | `FourQlib <https://github.com/microsoft/FourQlib>`_  |
+-------------+------------------------+------------------------------------------------------+

ECDH-PSI (3P)
-------------

We implement our own three-party PSI protocol based on ECDH. Note that our implementation has known
leakage, please use at your own risk.

Assume Alice, Bob, Charlie (receiver) want to perform 3P PSI, in addition to the final output, our 
protocol leaks the intersection size of Alice's data and Bob's data to Charlie.

.. figure:: ../imgs/dh_psi_3p.png

Note that at the beginning of ECDH-PSI protocol, we assume the input data from both Alice and Charlie are 
shuffled (It's not necessary to shuffle Bob's set).

Protocol:

1. For i-th element in its set, Alice calculates :math:`H(x_i)^\alpha` and sends to Bob.

2. For i-th element, Bob calculates :math:`H(x_i)^{\alpha\beta}` and 
   :math:`H(y_i)^\beta`, then shuffles them randomly and sends them to Alice.

3. For i-th element, Alice calculates :math:`H(y_i)^{\alpha\beta}` and gets the intersection of 
   :math:`H(x_i)^{\alpha\beta} \cap H(y_i)^{\alpha\beta}` (we denote the intersection as 
   :math:`I^{\alpha\beta}`), then sends :math:`I^{\alpha\beta}` to Charlie.

4. For i-th element, Charlie sends :math:`H(z_i)^{\gamma}` to Bob, Bob calculates and sends to 
   Alice :math:`H(z_i)^{\beta\gamma}`, finally Alice calculates and sends to 
   Charlie :math:`H(z_i)^{\alpha\beta\gamma}`.

5. Charlie calculates :math:`I^{\alpha\beta\gamma}` and compares :math:`I^{\alpha\beta\gamma}` with
   :math:`H(z_i)^{\alpha\beta\gamma}`.

KKRT16-PSI
----------

[KKRT16]_ is semi-honest OT-based PSI, based on OT Extension, BaRK-OPRF and CuckooHash. 
[KKRT16]_ is the first PSI protocol requiring only one minute for the case of larger sets 
( :math:`2^{24}` items each) of long strings (128 bits). 

We use 3-way stash-less CuckooHash proposed in [PSZ18]_.

.. figure:: ../imgs/kkrt16_psi.png

1. Sender and Receiver Agree on CuckooHash :math:`h_1,h_2,h_3: {\{0,1\}}^{*} \rightarrow [m]`
2. Receiver inserts each x into bin :math:`h_1(x)`, :math:`h_2(x)` or :math:`h_3(x)`
3. Sender inserts each y into bin :math:`h_1(y)`, :math:`h_2(y)` and :math:`h_3(y)`
4. Run BaRK-OPRF, Receiver gets :math:`F_{s,k_i}(x)`,Sender gets :math:`F_{s,k_i}(y)`, for :math:`bin_i`
5. Sender sends all :math:`\{F_{s,k_i}(y)\}` values to Receiver
6. Receiver compares two BaRK-OPRFs set and obtains the intersection.

 
BC22 PCG-PSI
------------

Pseudorandom Correlation Generator (PCG), is a primitive introduced in the work of Boyle et
al. [BCG+19b]_, [BCGI18]_, [SGRR19]_, [BCG+19a]_, [CIK+20]_. The goal of PCG is to compress long sources
of correlated randomness without violating security. 

Boyle et al. have designed multiple concretely efficient PCGs
for specific correlations, such as vector oblivious linear evaluation (VOLE) or batch oblivious linear
evaluation (BOLE). These primitives are at the heart of modern secure computation protocols with low
communication overhead.The VOLE functionality allows a receiver to learn a secret linear combination
of two vectors held by a sender and constructed (with sublinear communication) under variants
of the syndrome decoding assumption.

[BC22]_ uses PCG to speed up private set intersection protocols, minimizing computation and communication.
We implement semi-honest version psi in [BC22]_ and use PCG/VOLE from [WYKW21]_ . [BC22]_ PSI protocol 
requires only 30 seconds for the case of larger sets ( :math:`2^{24}` items each) of long strings (128 bits), 
and reduces 1/3 communication than [KKRT16]_.

.. figure:: ../imgs/pcg_psi.png

1. Sender and Receiver agree on :math:`(3,2)`-Generalized CuckooHash :math:`h_1,h_2: {\{0,1\}}^{*} \rightarrow [m]`

2. Receiver inserts each x into bin :math:`h_1(x)` or :math:`h_2(x)`

3. Sender inserts each y into bin :math:`h_1(y)` and :math:`h_2(y)`

4. Run PCG/VOLE from [WYKW21]_, :math:`w_i = \Delta * u_i + v_i`,  Receiver gets :math:`w_i` and :math:`\Delta`, 
   Sender gets :math:`u_i` and :math:`v_i`, for each :math:`bin_i`

5. Receiver sends Masked Bin Polynomial Coefficients to Sender, and receives BaRK-OPRF values

6. Sender sends all BaRK-OPRF values for each :math:`{\{y_i\}}_{i=1}^{n_2}` to Receiver

7. Receiver compares two BaRK-OPRFs sets and gets intersection.

Differentially Private PSI
--------------------------

We also implement a Differentially Private (DP) Private Set Intersection (PSI)
Protocol. Our implementation bases on ECDH-PSI, and provides:

- Differentially private PSI results.

This feature is currently under test, please use at your own risk!  

Why PSI with differentially private results? If we want a scheme that protects
both the private inputs and output privacy, an ideal way is to use `circuit
PSI`, which is a typical PSI variant that allows secure computation (e.g. MPC or
HE) on the PSI result without revealing it. `PSTY19
<https://eprint.iacr.org/2019/241.pdf>`_ However those protocols are expensive
in terms of efficiency.  

DP-PSI is a way of utilizing the up-sampling and sub-sampling mechanism to add
calibrated noises to the PSI results, without revealing its concise value.  

The protocol is listed below, assume Alice has a (hashed and shuffled) set
:math:`X` and Bob has a (hashed and shuffled) :math:`Y`.  

.. figure:: ../imgs/dp_psi.png

Note that we use "encrypt" to denote the process of calculating :math:`y\gets
x^a`.

Protocol:

1. Alice and Bob first encrypts their own dataset, and gets :math:`X^a` and
   :math:`Y^b` separately.
   
2. Alice sends :math:`X^a` to Bob.
   
3. Bob performs random subsampling on :math:`Y^b`, gets :math:`Y_*^b` and sends it
   to Alice. In the meantime, on receiving :math:`X^a` from Alice, Bob
   re-encrypts it with :math:`b`, gets :math:`X^{ab}`. Then it samples a random
   permutation :math:`\pi` to permute Alice's set, and sends permuted
   :math:`\pi(X^{ab})` back to Alice.
   
4. On receiving :math:`Y_*^b` and :math:`\pi(X^{ab})` from Bob, Alice re-encrypts
   :math:`Y_*^b` and gets :math:`Y_*^{ab}`, then calculates the intersection
   :math:`I_*^{ab}\gets\pi(X^{ab})\cap Y_*^{ab}`.
   
5. Alice randomly subsamples the intersection, gets :math:`I_{**}^{ab}`, and
   then finds their corresponding index in :math:`Y_*^b`. Then randomly adds
   non-intersection index to this set.
   
6. Alice sends the index set to Bob, then Bob reveals the final results.

In the end, this scheme ensures that the receiver (Bob) only learns the noised
intersection, without the ability of pointing out whether an element is in the
actual set intersection or not.  

Note that multiple invocations of DP-PSI inevitably weaken the privacy
protection, therefore, we strongly suggest that user should implement a
protection mechanism to prevent multiple DP-PSI executions on the same input
value.  

+---------------------------+--------+---------+---------+---------+-----------+
| Intel(R) Xeon(R) Platinum | 2^20   | 2^21    | 2^22    | 2^23    |   2^24    |
+===========================+========+=========+=========+=========+===========+
|   DP-PSI                  | 9.806s | 20.134s | 42.067s | 86.580s | 170.359s  |
+---------------------------+--------+---------+---------+---------+-----------+

For DP, our default privacy protection strength is :math:`\epsilon=3`. For more
details, please refer to the original paper: [DP-PSI]_

Unbalanced PSI
--------------

Ecdh-OPRF based PSI
>>>>>>>>>>>>>>>>>>>

[RA18]_ section 3 introduces Basic Unbalanced PSI(Ecdh-OPRF based) protocol proposed in [BBCD+11]_ that relaxes 
the security of the [JL10]_ to be secure against semi-honest adversaries. The protocol has two phases, the preprocessing phase and the online phase. The
authors introduced many optimizations to push as much computation and communication cost to
the preprocessing phase as possible.

An Oblivious Pseudorandom Function (OPRF) is a two-party protocol between client and server for computing the 
output of a Pseudorandom Function (PRF). [draft-irtf-cfrg-voprf-10]_ specifies OPRF, VOPRF, and POPRF protocols 
built upon prime-order groups.

.. figure:: ../imgs/ecdh_oprf_psi.png

- Offline Phase
  
  1. For each element :math:`y_i` in its set, Bob applies PRF using 
     private key :math:`\beta`, i.e. computing :math:`H_2(y_i,{H_1(y_i)}^\beta)` . 
  
  2. Bob sends :math:`\{\{H_2(y_i,{H_1(y_i)}^\beta)\}\}_{i=1}^{n_2}` to Alice in shuffled order.
   
- Online Phase
  
  1. For each element :math:`x_i` in its set, Alice applies the hash function and then exponentiates 
     it using its blind key :math:`r_i`, thus computing :math:`{H_1(x_i)}^{r_i}`. Alice sends 
     :math:`\{\{H_1(x_i)\}^{r_i}\}_{i=1}^{n_1}` to Bob.
  2. For each element :math:`H_1(x_i)^{r_i}` received from Alice in the previous step, Bob exponentiates 
     it using its key :math:`\beta`, computing :math:`{H_1(x_i)}^{r_i\beta}`. 
     Bob sends :math:`{\{\{H_1(x_i)\}^{\{r_i\}\beta}\}}_{i=1}^{n_1}` to Alice.
  3. Alice receives :math:`{\{\{H_1(x_i)\}^{r_i\beta}\}}_{i=1}^{n_1}` from Bob, and unblinds it using :math:`r_i`,
     gets :math:`\{\{\{H_1(x_i)\}^\beta\}\}_{i=1}^{n_1}`, computes OPRF :math:`\{\{H_2(x_i,{H_1(x_i)}^\beta)\}\}_{i=1}^{n_1}`.
  4. Alice compares two sets :math:`\{\{H_2(x_i,{H_1(x_i)}^\beta)\}\}_{i=1}^{n_1}`
     and :math:`\{\{H_2(y_i,{H_1(y_i)}^\beta)\}\}_{i=1}^{n_2}` and gets intersection.

Labeled PSI
>>>>>>>>>>>

Somewhat homomorphic encryption (SHE) can be used to build efficient (labeled) Private Set Intersection 
protocols in the unbalanced setting, where one of the sets is much larger than the other. 
[CMGD+21]_ introduces several optimizations and improvements to the protocols of 
[CLR17]_, [CHLR18]_, resulting in improved running time and improved communication complexity in the 
sender's set size.

Microsoft `APSI (Asymmetric PSI) <https://github.com/microsoft/APSI>`_  library provides a PSI functionality 
for asymmetric set sizes based on the latest [CMGD+21]_.  APSI uses the BFV([FV12]_) encryption scheme implemented 
in the Microsoft [SEAL]_ library.

SecretFlow SPU wraps `APSI <https://github.com/microsoft/APSI>`_ library, can be used for 

- Unbalanced PSI
- Malicious PSI
- Labeled PSI
- Keyword PIR

.. figure:: ../imgs/labeled_psi.png

- Setup Phase
  
  1. **Choose ItemParams**, TableParams, QueryParams, SEALParams.
  2. **Sender's OPRF**: The sender samples a key :math:`\beta` for the OPRF, updates its items set 
     to :math:`\{\{H_2(s_i,{H_1(s_i)}^\beta)\}\}_{s_i\in S}`.
  3. **Sender's Hashing**: Sender inserts all :math:`s_i\in S` into the sets :math:`\mathcal{B}[h_0(s_i)]`,
     :math:`\mathcal{B}[h_1(s_i)]` and :math:`\mathcal{B}[h_2(s_i)]`.
  4. **Splitting**: For each set :math:`\mathcal{B}[i]`, the sender splits it into bin bundles, denoted as
     :math:`\mathcal{B}[i,1]`, ..., :math:`\mathcal{B}[i,k]`.
  5. **Computing Coeffcients**: 
   
     - **Matching Polynomial**: For each bin bundle :math:`\mathcal{B}[i,j]`, the sender computes the 
       matching polynomial over :math:`\mathbb{F}_t`.
     - **Label Polynomial**: If the sender has labels associated with its set, then for each bin bundle 
       :math:`\mathcal{B}[i,j]`, the sender interpolates the label polynomial over :math:`\mathbb{F}_t`.
   
- Intersection Phase
  
  1. Receiver Encrypt :math:`r_i \in R`.

     - **Receiver's OPRF**: Receiver and Sender run ecdh-OPRF protocol, get 
       :math:`\{\{H_2(r_i,{H_1(r_i)}^\beta)\}\}_{r_i\in R}`.
     - **Receiver's CuckooHash**: Receiver performs cuckoo hashing on the set :math:`R` into CuckooTable C with m bins
       using h1; h2; h3 has the hash functions.
     - **Packing**: Receiver packs items in CuckooTable C into a FHE plaintext polynomial.
     - **Windowsing**: the receiver computes the component-wise query powers.
     - **Encrypt**: The receiver uses *FHE.Encrypt* to encrypt query powers and sends the ciphertexts to the sender.

  2. **Sender Homomorphically evaluate Matching Polynomial**: The sender receives the collection of
     ciphertexts and homomorphically evaluates Matching Polynomial. If Labeled PSI is desired, Sender homomorphically evaluates 
     Label Polynomial. The sender sends evaluated ciphertexts to Receiver.
  3. **Receiver Decrypt and Get result**: receiver receives and decrypts the matching ciphertexts, and labels 
     ciphertexts if needed, outputs the matching set and labels.

Labeled PSI Parameters

+-----+------------------------------------+---------------------------------------------------------------------+
|     | Params                             | function                                                            |
+=====+=============+======================+=====================================================================+
| 1   | ItemParams  |                      |                                                                     |
+-----+-------------+----------------------+---------------------------------------------------------------------+
|                   | felts_per_item       | how many Microsoft SEAL batching slots should represent each item   |
|                   |                      | = item_bit_size / plain_modulus_bits                                |
|                   |                      | item_bit_size = stats_params + log(ns)+log(nr)                      |
+-----+-------------+----------------------+---------------------------------------------------------------------+
| 2   | TableParams |                      |                                                                     |
+-----+-------------+----------------------+---------------------------------------------------------------------+
|                   | hash_func_count      | cuckoo hash count. if nr>1,hash_func_count = 3                      |
|                   |                      | nr=1-> hash_func_count=1 means essentially disabling cuckoo hashing |
+-------------------+----------------------+---------------------------------------------------------------------+
|                   | table_size           | positive multiple of floor(poly_modulus_degree/felts_per_item)      |
+-------------------+----------------------+---------------------------------------------------------------------+
|                   | max_items_per_bin    | how many items fit into each row of the sender's bin bundles        |
+-----+-------------+----------------------+---------------------------------------------------------------------+
| 3   | QueryParams |                      |                                                                     |
+-----+-------------+----------------------+---------------------------------------------------------------------+
|                   | ps_low_degree        | any number between 0 and max_items_per_bin                          |
|                   |                      | If set to zero, the Paterson-Stockmeyer algorithm is not used       |
|                   |                      | ps_low_degree > 1, use Paterson-Stockmeyer algorithm                |
+-------------------+----------------------+---------------------------------------------------------------------+
|                   | query_powers         | how many items fit into each row of the sender's bin bundles        |
|                   |                      | ref Challis and Robinson (2010) to determine good source powers     |
+-----+-------------+----------------------+---------------------------------------------------------------------+
| 4   | SEALParams  |                      |                                                                     |
+-----+-------------+----------------------+---------------------------------------------------------------------+
|                   | poly_modulus_degree  | 2048 /  4096 / 8192                                                 |
+-------------------+----------------------+---------------------------------------------------------------------+
|                   | plain_modulus(_bits) | 16(65535) / 22(bits)                                                |
+-------------------+----------------------+---------------------------------------------------------------------+
|                   | coeff_modulus_bits   | {48} / {48, 30, 30} / {56, 56, 56, 50}                              |
+-------------------+----------------------+---------------------------------------------------------------------+


Reference
------------

.. [BCGI18] E. Boyle, G. Couteau, N. Gilboa, and Y. Ishai. Compressing vector OLE. In ACM CCS 2018,
   pages 896–912. ACM Press, October 2018.

.. [BCG+19a] E. Boyle, G. Couteau, N. Gilboa, Y. Ishai, L. Kohl, P. Rindal, and P. Scholl. Efficient two-round
   OT extension and silent non-interactive secure computation. In ACM CCS 2019, pages 291–308.
   ACM Press, November 2019.

.. [BCG+19b] E. Boyle, G. Couteau, N. Gilboa, Y. Ishai, L. Kohl, P. Rindal, and P. Scholl. 
   Efficient two-round OT extension and silent non-interactive secure computation. In ACM CCS 2019,
   pages 291–308. ACM Press, November 2019.

.. [BC22] Private Set Intersection from Pseudorandom Correlation Generators

.. [Ber06] Daniel J. Bernstein. Curve25519: new diffie-hellman speed records. In In Public
   Key Cryptography (PKC), Springer-Verlag LNCS 3958, page 2006, 2006. (Cited on page 4.)

.. [BBCD+11] Baldi, P., Baronio, R., Cristofaro, E.D., Gasti, P., Tsudik, G.: Countering GATTACA:
   Efficient and Secure Testing of Fully-sequenced Human Genomes. In: ACM
   Conference on Computer and Communications Security. pp. 691–702. ACM (2011)

.. [CIK+20] G. Couteau, Y. Ishai, L. Kohl, E. Boyle, P. Scholl, and N. Gilboa. Efficient pseudorandom
   correlation generators from ring-lpn. Springer-Verlag, 2020.

.. [CHLR18] Chen, H., Huang, Z., Laine, K., Rindal, P.: Labeled PSI from fully homomorphic encryption with malicious
   security. In: Lie, D., Mannan, M., Backes, M., Wang, X. (eds.) ACM CCS 2018. pp. 1223{1237. ACM Press (Oct
   2018). https://doi.org/10.1145/3243734.3243836

.. [CLR17] Chen, H., Laine, K., Rindal, P.: Fast private set intersection from homomorphic encryption. In: Thuraisingham,
   B.M., Evans, D., Malkin, T., Xu, D. (eds.) ACM CCS 2017. pp. 1243{1255. ACM Press (Oct / Nov 2017).
   https://doi.org/10.1145/3133956.3134061

.. [CMGD+21] Kelong Cong, Radames Cruz Moreno, Mariana Botelho da Gama, Wei Dai, Ilia Iliashenko, Kim Laine, 
   Michael Rosenberg. Labeled PSI from Homomorphic Encryption with Reduced Computation and Communication 
   CCS'21: Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications SecurityNovember 2021    

.. [DP-PSI] Differentially-Private PSI https://arxiv.org/pdf/2208.13249.pdf

.. [FourQ] Costello, C., Longa, P.: Fourq: four-dimensional decompositions on a q-curve over the mersenne prime. 
    Cryptology ePrint Archive, Report 2015/565 (2015), https://eprint.iacr.org/2015/565

.. [FV12] Fan, J., Vercauteren, F.: Somewhat practical fully homomorphic encryption. Cryptology ePrint Archive, 
   Report 2012/144 (2012), http://eprint.iacr.org/2012/144.pdf

.. [HFH99] Bernardo A. Huberman, Matt Franklin, and Tad Hogg. Enhancing privacy and trust in electronic
   communities. In ACM CONFERENCE ON ELECTRONIC COMMERCE. ACM, 1999.

.. [ipp-crypto] https://github.com/intel/ipp-crypto/ 

.. [JL10] Jarecki, S., Liu, X.: Fast Secure Computation of Set Intersection. In: SCN. LNCS,
   vol. 6280, pp. 418–435. Springer (2010)

.. [KKRT16] V. Kolesnikov, R. Kumaresan, M. Rosulek, and N. Trieu. Efficient batched oblivious PRF with
    applications to private set intersection. In ACM CCS 2016, pages 818–829. ACM Press, October 2016.

.. [Mea86] C. Meadows. A more efficient cryptographic matchmaking protocol for use in the absence of a
   continuously available third party. In 1986 IEEE Symposium on Security and Privacy, pages 134–134, April 1986.

.. [PSZ18] B. Pinkas, T. Schneider, and M. Zohner. Scalable private set intersection based on ot extension.
   ACM Transactions on Privacy and Security (TOPS), 21(2):1–35, 2018.

.. [RA18] Resende, A.C.D., Aranha, D.F.: Faster unbalanced private set intersection. In: Meiklejohn, S., 
   Sako, K. (eds.) FC2018. LNCS, vol. 10957, pp. 203{221. Springer, Heidelberg (Feb / Mar 2018)   

.. [SEAL] Microsoft SEAL (release 4.0). https://github.com/Microsoft/SEAL (Sep 2022), 
   microsoft Research, Redmond, WA.

.. [SEC2-v2] Standards for Efficient Cryptography (SEC) <http://www.secg.org/sec2-v2.pdf>

.. [SGRR19] P. Schoppmann, A. Gascón, L. Reichert, and M. Raykova. Distributed vector-OLE: Improved
    constructions and implementation. In ACM CCS 2019, pages 1055–1072. ACM Press, November 2019.

.. [WYKW21] C. Weng, K. Yang, J. Katz, and X. Wang. Wolverine: fast, scalable, and communication-efficient
   zero-knowledge proofs for boolean and arithmetic circuits. In 2021 IEEE Symposium on Security
   and Privacy (SP), pages 1074–1091. IEEE, 2021.

.. [draft-irtf-cfrg-voprf-10] Oblivious Pseudorandom Functions (OPRFs) using Prime-Order Groups. 
   https://www.ietf.org/archive/id/draft-irtf-cfrg-voprf-10.html   
