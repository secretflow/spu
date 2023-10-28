# The Groupby Operation

## What is groupby operation?

The groupby operation in pandas is used to split data into groups based on a set of specific attributes or columns.
It allows us to apply functions or calculations to each group, facilitating analysis and summarization of data.
The results obtained provide insights on patterns and characteristics within each group.

## How to do groupby operation using pure jax numpy (secure and fast)?

our groupby operation is achived in two major steps and one optional postprocess step:

1. segmentation: segmentation is a process of splitting data into groups based on a set of specific attributes or columns.
2. aggregation: aggregation is a process of combining data within each group to obtain a single value.
3. (optional) shuffling: shuffle the results, necessary if count statistics is not revealed.

## What does the segmentation do?

The segmentation consists of 3 steps:

1. sort the table according to the key_columns.
2. the group ends are found by finding the differences in the key columns.
3. the group numbers are assigned.

Step 1 will produce:

* key_columns_sorted
* target_columns_sorted

Step 2 will produce:

* segment_end_marks

Step 3 will produce:

* segment_ids

We call these 4 results the "segmentation information", because they entail which group each element is assigned into.

## What does the aggregation do?

Based on the segmentation information, we are now ready to aggregate the data.
For each group, we will perform sum, or min, or max, etc operation.

### How to do this fast?

The main idea is a classfical prefix sum with some modifications:

1. We use one column to indicate the membership of each sample to its group.
2. We devise a modified version of operation which has 3 properties:

    * correctness: The binary operation scanning through the list produces the final statistics we care.
                    (e.g. the addition of all elements is indeed the sum)
    * indicator awareness: The binary operation is aware of the membership of each sample to its group.
    * associativity: The binary operation is associative.

3. We use associative scan to do the prefix sum in a parallel manner.

## Why shuffle?

By the definition of MPC, we cannot reveal any information except those can be inferred from the result.
When we compute the statistics except count, we cannot infer the information about the number of elements in each group.
Hence the count of each group must be protected.

Our aggregation algorithms produce the results in form of a sparse matrix, with group statistics at the end of each group and zero else where.
Without shuffling, the count statistics can be inferred by positions of the non-zero elements.

So we need to shuffle the results to make sure that the count statistics are not revealed.

As a side effect, we also need some (cleartext) postprocessing steps to clean up the shuffled results.

## Organization of files

1. segmentation.py: contains the implementation of the segmentation algorithm.
2. aggregation.py: contains the implementation of the aggregation algorithm (without shuffling, suitable for internal use).
3. shuffle.py: contains the implementation of the shuffle algorithm.
4. utils.py: contains utility functions for the algorithms.
5. groupby_via_shuffle.py: contains the aggregation operations with shuffling (suitable for application).
6. postpocess.py: contains the (cleartext) postprocess operations to clean up the results.
