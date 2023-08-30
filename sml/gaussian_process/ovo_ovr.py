import copy

import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import expit
from kernels import RBF


# ovo方法输出的维度不一致。难以写成vmap形式。极慢。暂时不支持。
def _ovr_decision_function(predictions, confidences, n_classes):
    n_samples = predictions.shape[1]
    votes = jnp.zeros((n_classes, n_samples), dtype=int)
    sum_of_confidences = jnp.zeros((n_classes, n_samples), dtype=jnp.float32)

    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            sum_of_confidences = sum_of_confidences.at[i].set(
                sum_of_confidences[i] - confidences[k]
            )
            sum_of_confidences = sum_of_confidences.at[j].set(
                sum_of_confidences[j] + confidences[k]
            )
            votes = votes.at[i].set(
                jnp.where(predictions[k] == 0, votes[i] + 1, votes[i])
            )
            votes = votes.at[j].set(
                jnp.where(predictions[k] == 1, votes[j] + 1, votes[j])
            )
            k += 1

    transformed_confidences = sum_of_confidences / (
        3 * (jnp.abs(sum_of_confidences) + 1)
    )
    return (votes + transformed_confidences).T


def _fit_ovo_binary(estimator, X, Y, i, j):
    """Fit a single binary estimator (one-vs-one)."""
    cond = jnp.logical_or(Y == i, Y == j)
    y = y[cond]
    # y = []
    # new_x = []
    # for xxx, yyy in zip(X, y):
    #     if yyy == i or yyy == j
    #         y.append(yyy)
    #         new_x.append(xxx)
    # y = jnp.array(y)
    # new_x = jnp.array(new_x)

    y_binary = jnp.empty(y.shape, dtype=int)
    y_binary = jnp.where(y == i, 0, 1)
    indcond = jnp.arange(X.shape[0])[cond]
    estimator1 = copy.deepcopy(estimator)
    estimator1.fit(X[indcond], y_binary)
    # estimator1.fit(new_x, y_binary)
    return estimator1


class OneVsOneClassifier:
    def __init__(self, estimator, n_classes, *, n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.classes_ = n_classes

    def fit(self, X0, y):
        X = jnp.array(X0)
        self.estimators_ = [
            _fit_ovo_binary(self.estimator, X, y, i, j)
            for i in range(len(self.classes_))
            for j in range(i + 1, len(self.classes_))
        ]

    def decision_function(self, X):
        predictions = jnp.vstack([est.predict(X) for est in self.estimators_])

        confidences = jnp.vstack(
            [est.predict_proba(X)[:, 1] for est in self.estimators_]
        )

        Y = _ovr_decision_function(predictions, confidences, len(self.classes_))
        return Y

    def predict(self, X0):
        X = jnp.array(X0)
        Y = self.decision_function(X)
        return Y.argmax(axis=1)


# 此处注释的vmap版本ovr算法,本地已经跑通,但是spu一直报错

class OneVsRestClassifier():
    def __init__(self, estimator, n_classes, * , n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.classes_ = n_classes

    def fit(self, X, y):
        self.y_binary = jnp.array([jnp.where(y == i, 0, 1) for i in range(self.classes_)])

        # y_binary_list = []
        # for i in range(int(self.classes_)):
        #     label = jnp.where(y == i, 0, 1)
        #     y_binary_list.append(label)
        # self.y_binary = jnp.array(y_binary_list)

        self.fs_ = vmap(self._fit_ovr_binary, in_axes = (None, 0) )(X, self.y_binary)

        self.estimator.approx_func = expit
        self.estimator.X_train_ = jnp.array(X)

        if self.estimator.kernel is None:  # Use an RBF kernel as default
            self.estimator.kernel_ = RBF()
        else:
            self.estimator.kernel_ = copy.deepcopy(self.estimator.kernel)

    def predict(self, X_test):
        maxima = vmap(self.ovr_predict_proba, in_axes = (0, None, 0))(self.y_binary, X_test, self.fs_)
        print(maxima)
        return maxima.argmax(axis=0)

    def ovr_predict_proba(self, y_binary, X_test, f_):
        estimator1 = copy.deepcopy(self.estimator)
        estimator1.y_train = y_binary
        estimator1.f_ = f_
        return estimator1.predict_proba(X_test)[:, 0]

    def _fit_ovr_binary(self, X, y_binary):
        estimator1 = copy.deepcopy(self.estimator)
        f_ = estimator1.fit(X, y_binary)
        return f_

# 简陋版本的ovr


# def _fit_ovr_binary(estimator, X, y, i):
#     y_binary = jnp.where(y == i, 0, 1)
#     estimator1 = copy.deepcopy(estimator)
#     estimator1.fit(X, y_binary)
#     return estimator1


# class OneVsRestClassifier:
#     def __init__(self, estimator, n_classes, *, n_jobs=None):
#         self.estimator = estimator
#         self.n_jobs = n_jobs
#         self.classes_ = n_classes

#     def fit(self, X, y):
#         self.estimators_ = [
#             _fit_ovr_binary(self.estimator, X, y, i) for i in range(self.classes_)
#         ]

#     def predict(self, X0):
#         X = jnp.array(X0)
#         maxima = jnp.zeros(X.shape[0], dtype=jnp.float32)
#         argmaxima = jnp.zeros(X.shape[0], dtype=int)
#         for i, e in enumerate(self.estimators_):
#             pred = e.predict_proba(X)[:, 0]
#             maxima = jnp.maximum(maxima, pred)
#             argmaxima = jnp.where(maxima == pred, i, argmaxima)
#         # return jnp.array([self.classes_[ind] for ind in argmaxima])
#         return argmaxima
