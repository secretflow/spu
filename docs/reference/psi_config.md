# SPU PSI Configuration

## Table of Contents



- Messages
    - [BucketPsiConfig](#bucketpsiconfig)
    - [DpPsiParams](#dppsiparams)
    - [InputParams](#inputparams)
    - [MemoryPsiConfig](#memorypsiconfig)
    - [OutputParams](#outputparams)
    - [PsiResultReport](#psiresultreport)



- Enums
    - [CurveType](#curvetype)
    - [PsiType](#psitype)



- [Scalar Value Types](#scalar-value-types)



 <!-- end services -->

## Messages


### BucketPsiConfig
The Bucket-psi configuration.

```python
  config = psi.BucketPsiConfig(  # prepare config
      psi_type=PsiType.ECDH_PSI_2PC,
      broadcast_result=True,
      receiver_rank=0,
      input_params=psi.InputParams(path='/xxx/ccc.csv', select_fields=['c1', 'c2']),
      output_params=psi.OutputParams(path='/yyyy/oooo.csv', need_sort=True),
  )
  report = psi.bucket_psi(lctx, config)  # run psi and get report
```


| Field | Type | Description |
| ----- | ---- | ----------- |
| psi_type | [ PsiType](#psitype) | The psi type. |
| receiver_rank | [ uint32](#uint32) | Specified the receiver rank. Receiver can get psi result. |
| broadcast_result | [ bool](#bool) | Whether to broadcast psi result to all parties. |
| input_params | [ InputParams](#inputparams) | The input parameters of psi. |
| output_params | [ OutputParams](#outputparams) | The output parameters of psi. |
| curve_type | [ CurveType](#curvetype) | Optional, specified elliptic curve cryptography used in psi when needed. |
| bucket_size | [ uint32](#uint32) | Optional, specified the hash bucket size used in psi. |
| preprocess_path | [ string](#string) | Optional，The path of offline preprocess file. |
| ecdh_secret_key_path | [ string](#string) | Optional，secret key path of ecdh_oprf, 256bit/32bytes binary file. |
| dppsi_params | [ DpPsiParams](#dppsiparams) | Optional，Params for dp-psi |
 <!-- end Fields -->
 <!-- end HasFields -->


### DpPsiParams
The input parameters of dp-psi.


| Field | Type | Description |
| ----- | ---- | ----------- |
| bob_sub_sampling | [ double](#double) | bob sub-sampling bernoulli_distribution probability. |
| epsilon | [ double](#double) | dp epsilon |
 <!-- end Fields -->
 <!-- end HasFields -->


### InputParams
The input parameters of psi.


| Field | Type | Description |
| ----- | ---- | ----------- |
| path | [ string](#string) | The path of input csv file. |
| select_fields | [repeated string](#string) | The select fields of input data. |
| precheck | [ bool](#bool) | Whether to check select fields duplicate. |
 <!-- end Fields -->
 <!-- end HasFields -->


### MemoryPsiConfig
The In-memory psi configuration.

```python
  config = psi.MemoryPsiConfig(  # prepare config
      psi_type=PsiType.ECDH_PSI_2PC,
      broadcast_result=True,
      receiver_rank=0,
  )
  joined_list = psi.mem_psi(
      lctx, config, ['a1', 'v2', 'b3', 'v4']
  )  # run psi and get joined list
```


| Field | Type | Description |
| ----- | ---- | ----------- |
| psi_type | [ PsiType](#psitype) | The psi type. |
| receiver_rank | [ uint32](#uint32) | Specified the receiver rank. Receiver can get psi result. |
| broadcast_result | [ bool](#bool) | Whether to broadcast psi result to all parties. |
| curve_type | [ CurveType](#curvetype) | Optional, specified elliptic curve cryptography used in psi when needed. |
| dppsi_params | [ DpPsiParams](#dppsiparams) | Optional，Params for dp-psi |
 <!-- end Fields -->
 <!-- end HasFields -->


### OutputParams
The output parameters of psi.


| Field | Type | Description |
| ----- | ---- | ----------- |
| path | [ string](#string) | The path of output csv file. |
| need_sort | [ bool](#bool) | Whether to sort output file by select fields. |
 <!-- end Fields -->
 <!-- end HasFields -->


### PsiResultReport
The report of psi result.


| Field | Type | Description |
| ----- | ---- | ----------- |
| original_count | [ int64](#int64) | The data count of input. |
| intersection_count | [ int64](#int64) | The count of intersection. Get `-1` when self party can not get result. |
 <!-- end Fields -->
 <!-- end HasFields -->
 <!-- end messages -->

## Enums


### CurveType
The specified elliptic curve cryptography used in psi.

| Name | Number | Description |
| ---- | ------ | ----------- |
| CURVE_INVALID_TYPE | 0 | none |
| CURVE_25519 | 1 | Daniel J. Bernstein. Curve25519: new diffie-hellman speed records |
| CURVE_FOURQ | 2 | FourQ: four-dimensional decompositions on a Q-curve over the Mersenne prime |
| CURVE_SM2 | 3 | SM2 is an elliptic curve based cryptosystem (ECC) published as a Chinese National Standard as GBT.32918.1-2016 and published in ISO/IEC 14888-3:2018 |
| CURVE_SECP256K1 | 4 | parameters of the elliptic curve defined in Standards for Efficient Cryptography (SEC) http://www.secg.org/sec2-v2.pdf |




### PsiType
The algorithm type of psi.

| Name | Number | Description |
| ---- | ------ | ----------- |
| INVALID_PSI_TYPE | 0 | none |
| ECDH_PSI_2PC | 1 | DDH based PSI |
| KKRT_PSI_2PC | 2 | Efficient Batched Oblivious PRF with Applications to Private Set Intersection https://eprint.iacr.org/2016/799.pdf |
| BC22_PSI_2PC | 3 | PSI from Pseudorandom Correlation Generators https://eprint.iacr.org/2022/334 |
| ECDH_PSI_3PC | 4 | Multi-party PSI based on ECDH (Say A, B, C (receiver)) notice: two-party intersection cardinarlity leak (|A intersect B|) |
| ECDH_PSI_NPC | 5 | Iterative running 2-party ecdh psi to get n-party PSI. Notice: two-party intersection leak |
| KKRT_PSI_NPC | 6 | Iterative running 2-party kkrt psi to get n-party PSI. Notice: two-party intersection leak |
| ECDH_OPRF_UB_PSI_2PC_GEN_CACHE | 7 | ecdh-oprf 2-party Unbalanced-PSI Generate CACHE. |
| ECDH_OPRF_UB_PSI_2PC_TRANSFER_CACHE | 8 | ecdh-oprf 2-party Unbalanced-PSI transfer CACHE. |
| ECDH_OPRF_UB_PSI_2PC_OFFLINE | 9 | ecdh-oprf 2-party Unbalanced-PSI offline phase. |
| ECDH_OPRF_UB_PSI_2PC_ONLINE | 10 | ecdh-oprf 2-party Unbalanced-PSI online phase. |
| ECDH_OPRF_UB_PSI_2PC_SHUFFLE_ONLINE | 11 | ecdh-oprf 2-party Unbalanced-PSI with shuffling online phase. large set party get intersection result |
| DP_PSI_2PC | 12 | Differentially-Private PSI https://arxiv.org/pdf/2208.13249.pdf bases on ECDH-PSI, and provides: Differentially private PSI results. |


 <!-- end Enums -->
 <!-- end Files -->

## Scalar Value Types

| .proto Type | Notes | C++ Type | Java Type | Python Type |
| ----------- | ----- | -------- | --------- | ----------- |
| <div><h4 id="double" /></div><a name="double" /> double |  | double | double | float |
| <div><h4 id="float" /></div><a name="float" /> float |  | float | float | float |
| <div><h4 id="int32" /></div><a name="int32" /> int32 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint32 instead. | int32 | int | int |
| <div><h4 id="int64" /></div><a name="int64" /> int64 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint64 instead. | int64 | long | int/long |
| <div><h4 id="uint32" /></div><a name="uint32" /> uint32 | Uses variable-length encoding. | uint32 | int | int/long |
| <div><h4 id="uint64" /></div><a name="uint64" /> uint64 | Uses variable-length encoding. | uint64 | long | int/long |
| <div><h4 id="sint32" /></div><a name="sint32" /> sint32 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int32s. | int32 | int | int |
| <div><h4 id="sint64" /></div><a name="sint64" /> sint64 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int64s. | int64 | long | int/long |
| <div><h4 id="fixed32" /></div><a name="fixed32" /> fixed32 | Always four bytes. More efficient than uint32 if values are often greater than 2^28. | uint32 | int | int |
| <div><h4 id="fixed64" /></div><a name="fixed64" /> fixed64 | Always eight bytes. More efficient than uint64 if values are often greater than 2^56. | uint64 | long | int/long |
| <div><h4 id="sfixed32" /></div><a name="sfixed32" /> sfixed32 | Always four bytes. | int32 | int | int |
| <div><h4 id="sfixed64" /></div><a name="sfixed64" /> sfixed64 | Always eight bytes. | int64 | long | int/long |
| <div><h4 id="bool" /></div><a name="bool" /> bool |  | bool | boolean | boolean |
| <div><h4 id="string" /></div><a name="string" /> string | A string must always contain UTF-8 encoded or 7-bit ASCII text. | string | String | str/unicode |
| <div><h4 id="bytes" /></div><a name="bytes" /> bytes | May contain any arbitrary sequence of bytes. | string | ByteString | str |

