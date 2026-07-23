# About This Folder

This C++ demo re-implements the paper [Squirrel: A Scalable Secure Two-Party Computation Framework for Training Gradient Boosting Decision Tree](https://eprint.iacr.org/2023/527),
using some fundamental functions from the SPU library.
Note that we approximate the logistic function `1/(1 + exp(x)` by a 4-segments spline function, instead of the Fourier series used in the Squirrel paper.

## Disclaimer

Code under this folder is purely for research demonstration and it's **NOT designed for production**.

## Build

`bazel build -c opt experimental/squirrel/...`

## Performance

### [APS Failure at Scania Trucks](http://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks)

* Number of features 170
* train set 60000 samples, and test set 16000  samples
* XGB parameters:
  * learning rate = 1.0
  * subsampling rate = 0.8
  * tree depth = 5
  * number of trees = 1.0
  * activation = logistic
* AUC on the test set **0.9869860044376173**
* Standalone Run
  * On one terminal

    ```sh
    bazel-bin/experimental/squirrel/squirrel_demo_main --rank0_nfeatures=85 --rank1_nfeatures=85 --standalone=true --train=BinaryClassification_Aps_Test_60000_171.csv --test=BinaryClassification_Aps_Test_16000_171.csv --rank=0 --has_label=0 --lr=1.0 --subsample=0.8
    ```

  * On another terminal

    ```sh
    bazel-bin/experimental/squirrel/squirrel_demo_main --rank0_nfeatures=85 --rank1_nfeatures=85 --standalone=true --train=BinaryClassification_Aps_Test_60000_171.csv --test=BinaryClassification_Aps_Test_16000_171.csv --rank=1 --has_label=1 --lr=1.0 --subsample=0.8
    ```

* Run on distributed dataset, e.g., using the `breast_cancer` dataset from the SPU repo.
  * On one terminal

    ```sh
    bazel-bin/experimental/squirrel/squirrel_demo_main --rank0_nfeatures=15 --rank1_nfeatures=15 --standalone=false --train=examples/data/breast_cancer_a.csv --rank=0 --has_label=0 --lr=1.0 --subsample=0.8
    ```

  * On another terminal

    ```sh
    bazel-bin/experimental/squirrel/squirrel_demo_main --rank0_nfeatures=15 --rank1_nfeatures=15 --standalone=false --train=examples/data/breast_cancer_b.csv --rank=1 --has_label=1 --lr=1.0 --subsample=0.8
    ```
