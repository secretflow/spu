# Algorithm Support lists

The table below shows the capabilities currently available in SML.
In general, the following features are rarely(partly) supported in SML:

- **Early stop** for training or iterating algorithm: We do not want to reveal any intermediate information.
- Manual set of **random seed**: SPU can't handle randomness of float properly, so if random value(matrix) is needed,
user should pass it as a parameter(such as `rsvd`, `NMF`)
- **Data inspection** like counting the number of label, re-transforming the data or label won't be done.
(So we may assume a "fixed" format for input or just tell the number of classes as a parameter)
- **single-sample SGD** not implemented for the latency consideration, MiniBatch-SGD(which we just call it `sgd` in sml) will replace it.
- Jax's Ops like `eigh`, `svd` can't run in SPU directly: `svd` implemented now is expensive and can't handle matrix that is not column full-rank matrix.

|      Algorithm       |   category    |                                   Supported features                                    | Notes                                                                                                                                                                           |
|:--------------------:|:-------------:|:---------------------------------------------------------------------------------------:|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        KMEANS        |    cluster    |                         init=`random` , algorithm=`lloyd`  only                         | only run algo once for efficiency                                                                                                                                               |
|         PCA          | decomposition |  1. `power_iteration` method(not used in scikit-learn) supported<br/> 2. `rsvd` method  | 1. if method=`power_iteration`, then cov matrix will be computed first<br/>2.`rsvd` is very unstable under fixedpoint setting even in `FM128`, so only small data is supported. |
|         NMF          | decomposition |                init=`random`,  solver=`mu`,   beta_loss=`frobenius` only                |                                                                                                                                                                                 |
|       Logistic       | linear model  |               1. `sgd` solver only<br/>2.only L2 regularization supported               | 1. `sigmoid` will be evaluated approximately                                                                                                                                    |
|      Perceptron      | linear model  |              1. all regularization methods<br/>2.patience-based early stop              | 1. this early stop will not cut down the training time, it just forces the update of parameters stop                                                                            |
|        Ridge         | linear model  |                           1. `svd` and `cholesky` solver only                           |                                                                                                                                                                                 |
|    SgdClassifier     | linear model  | 1. linear regression and logistic regression only<br/>L2  regularization supported only | 1. `sigmoid` will be evaluated approximately                                                                                                                                    |
| Gaussian Naive Bayes |  naive_bayes  |                           1. not support manual set of priors                           |                                                                                                                                                                                 |
|         KNN          |   neighbors   |      1.`brute` algorithm only<br/>      `uniform` and `distance` weights supported      | 1. KD-tree or Ball-tree can't improve the efficiency in MPC setting                                                                                                             |
|    roc_auc_score     |    metric     |                                      1.binary only                                      |                                                                                                                                                                                 |
