# LabelPSI

## Introduction

Fully homomorphic encryption (FHE) can be used to build efficient (labeled) Private Set Intersection protocols in the unbalanced setting,
where one of the sets is much larger than the other.

Reference:

- [Fast Private Set Intersection from Homomorphic Encryption](https://eprint.iacr.org/2017/299)
- [Labeled PSI from Fully Homomorphic Encryption with Malicious Security](https://eprint.iacr.org/2018/787)
- [Labeled PSI from Homomorphic Encryption with Reduced Computation and Communication (ACM CCS 2021)](https://eprint.iacr.org/2021/1116)
- [FHE and Private Set Intersection](https://simons.berkeley.edu/talks/fhe-and-private-set-intersection)
- [Private set intersection via somewhat homomorphic encryption by Ilia Iliashenko](https://fhe.org/talks/private-set-intersection)

Microsoft [APSI (Asymmetric PSI) library](https://github.com/microsoft/APSI) provides a PSI functionality for asymmetric set sizes based
on the latest [Labeled PSI protocol](https://eprint.iacr.org/2021/1116).

This is a wrap of [APSI library](https://github.com/microsoft/APSI), can be used for

- Unbalanced PSI
- Malicious PSI
- Labeled PSI
- Keyword PIR

## LabelPSI Protocol dataflow

|        | Client(Receiver) |         | Server(Sender)  |
| ------ | ---------------- | ------- | --------------- |
| Step 1 | Request Params   | ------> |                 |
|        |                  | <------ | Response Params |
| Step 2 |                  |         | Setup Server DB |
| Step 3 | Request OPRF     | ------> |                 |
|        |                  | <------ | Response OPRF   |
| Step 4 | Request Query    | ------> |                 |
|        |                  | <------ | Response Query  |

## LabelPSI source code

|     | file            | class              | function            |
| --- | --------------- | ------------------ | ------------------- |
| 1   | psi_params.h/cc |                    |                     |
|     |                 |                    | GetPsiParams        |
|     |                 |                    | PsiParamsToBuffer   |
|     |                 |                    | ParsePsiParamsProto |
| 2   | receiver.h/cc   |                    |                     |
|     |                 | LabelPsiReceiver   | RequestPsiParams    |
|     |                 |                    | RequestOPRF         |
|     |                 |                    | RequestQuery        |
| 3   | sender.h/cc     |                    |                     |
|     |                 | LabelPsiSender     | RunPsiParams        |
|     |                 |                    | RunOPRF             |
|     |                 |                    | RunQuery            |
| 4   | package.h/cc    |                    |                     |
|     |                 | PlainResultPackage |                     |
|     |                 | ResultPackage      |                     |
| 5   | sender_db.h/cc  |                    |                     |
|     |                 | SenderDB           | SetData             |
|     |                 |                    | GetItemCount        |
|     |                 |                    | GetBinBundleCount   |
|     |                 |                    | GetPackingRate      |

## LabelPSI Parameters

|     | file        | class                | function                                                            |
| --- | ----------- | -------------------- | ------------------------------------------------------------------- |
| 1   | ItemParams  |                      |                                                                     |
|     |             | felts_per_item       | how many Microsoft SEAL batching slots should represent each item   |
|     |             |                      | = item_bit_size / plain_modulus_bits                                |
|     |             |                      | item_bit_size = stats_params + log(ns)+log(nr)                      |
| 2   | TableParams |                      |                                                                     |
|     |             | hash_func_count      | cuckoo hash count. if nr>1,hash_func_count = 3                      |
|     |             |                      | nr=1-> hash_func_count=1 means essentially disabling cuckoo hashing |
|     |             | table_size           | positive multiple of floor(poly_modulus_degree/felts_per_item)      |
|     |             | max_items_per_bin    | how many items fit into each row of the sender's bin bundles        |
| 3   | QueryParams |                      |                                                                     |
|     |             | ps_low_degree        | any number between 0 and max_items_per_bin                          |
|     |             |                      | If set to zero, the Paterson-Stockmeyer algorithm is not used       |
|     |             |                      | ps_low_degree > 1, use Paterson-Stockmeyer algorithm                |
|     |             | query_powers         | defines which encrypted powers of the query, receiver to sender     |
|     |             |                      | ref Challis and Robinson (2010) to determine good source powers     |
| 4   | SEALParams  |                      |                                                                     |
|     |             | poly_modulus_degree  | 2048 /  4096 / 8192                                                 |
|     |             | plain_modulus(_bits) | 65535 / 65535 / 22(bits)                                            |
|     |             | coeff_modulus_bits   | {48} / {48, 30, 30} / {56, 56, 56, 50}                              |

[APSI](https://github.com/microsoft/APSI) example parameter sets in the [parameters](https://github.com/microsoft/APSI/tree/main/parameters) subdirectory.

We select three SEALParams from [APSI parameters](https://github.com/microsoft/APSI/tree/main/parameters) for different receiver and sender items size.

## Security Tips

Warning:  Labeled PSI are malicious PSI protocols, but malicious sender may do attack with his CuckooHash Table.
We suggest use Labeled PSI protocol as one-way PSI, i.e., just Client(Receiver) gets the final intersection result.
