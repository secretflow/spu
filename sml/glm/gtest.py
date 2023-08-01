from glm import *
import numpy as np
import scipy.stats as stats
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

n_samples, n_features = 100, 5

def gen_data(noise=False):
    """
    生成随机数据作为测试数据。

    Parameters:
    ----------
    noise : bool, optional (default=False)
        是否添加噪声。

    Returns:
    -------
    X : array-like, shape (n_samples, n_features)
        特征数据。
    y : array-like, shape (n_samples,)
        目标数据。
    coef : array-like, shape (n_features + 1,)
        真实的系数，包括截距项和特征权重。

    """
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    coef = np.random.rand(n_features + 1)  # +1 是截距项
    y = X @ coef[1:] + coef[0]
    if noise:
        noise = np.random.normal(loc=0, scale=0.05, size=num_samples)
        y += noise
    return X, y, coef

def test(model, X, y, coef, num=5):
    """
    测试广义线性回归模型的拟合、预测和评分功能。

    Parameters:
    ----------
    model : object
        广义线性回归模型对象。
    X : array-like, shape (n_samples, n_features)
        特征数据。
    y : array-like, shape (n_samples,)
        目标数据。
    coef : array-like, shape (n_features + 1,)
        真实的系数，包括截距项和特征权重。
    num : int, optional (default=5)
        需要展示的系数数量。

    Returns:
    -------
    None

    """
    model.fit(X, y)
    print('真实的系数:', coef[:num])
    print("拟合的系数：", model.coef_[:num])
    print("D^2评分：", model.score(X[:num], y[:num]))
    print("X:", X[:num])
    print("样例：", y[:num])
    print("预测:", model.predict(X[:num]))

def test_glm():
    """
    测试_GeneralizedLinearRegressor模型的功能。

    """
    X, y, coef = gen_data()
    from glm import _GeneralizedLinearRegressor
    model = _GeneralizedLinearRegressor()
    test(model, X, y, coef)

def test_lbfgs():
    """
    测试使用LBFGS优化算法的_GeneralizedLinearRegressor模型的功能。

    """
    X, y, coef = gen_data()
    from glm import _GeneralizedLinearRegressor
    model = _GeneralizedLinearRegressor(solver='lbfgs')
    test(model, X, y, coef)

def test_Poisson():
    """
    测试PoissonRegressor模型的功能。

    """
    X, y, coef = gen_data()
    y = jnp.round(jnp.exp(y))
    model = PoissonRegressor()
    test(model, X, y, coef)

def test_gamma():
    """
    测试GammaRegressor模型的功能。

    """
    X, y, coef = gen_data()
    alpha = 10
    y = np.array([stats.gamma.rvs(a=alpha, scale=y_i/alpha) for y_i in y])
    y = jnp.exp(y)
    model = GammaRegressor()
    test(model, X, y, coef)

def test_Tweedie():
    """
    测试TweedieRegressor模型的功能。

    """
    X, y, coef = gen_data()
    y = jnp.round(jnp.exp(y))
    model = TweedieRegressor()
    test(model, X, y, coef)

def test_sim():
    """
    使用模拟器测试_GeneralizedLinearRegressor模型的拟合功能。

    """
    sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64)
    X, y, coef = gen_data()

    def proc(X, y):
        from glm import _GeneralizedLinearRegressor
        model = _GeneralizedLinearRegressor(solver="newton-cholesky")
        model.fit(X, y)
        coef_fit = model.coef_
        return coef_fit

    result = spsim.sim_jax(sim, proc)(X, y)

if __name__ == '__main__':
    # Run the tests
    test_gamma()
    test_sim()
    test_glm()
    test_lbfgs()
    test_Poisson()
    test_Tweedie()
