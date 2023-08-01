from glm import *
import numpy as np
import scipy.stats as stats
import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim
n_samples, n_features = 100, 5
def gen_data(noise = False):
    # 生成一些随机数据
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    coef = np.random.rand(n_features + 1)  # +1 是截距项
    # y = X @ coef[1:] + coef[0] + np.random.normal(0, 0.01, n_samples)
    y = X @ coef[1:] + coef[0]
    if noise:
        noise = np.random.normal(loc=0, scale=0.05, size=num_samples)
        y += noise
    return X,y,coef

def test(model,X,y,coef,num = 5):
    model.fit(X,y)
    print('真实的系数:',coef[:num])
    print("拟合的系数：",model.coef_[:num])
    print("D^2评分：",model.score(X[:num], y[:num]))
    print("X:",X[:num])
    print("样例：",y[:num])
    print("预测:",model.predict(X[:num]))

def test_glm():
    X,y,coef = gen_data()
    from glm import _GeneralizedLinearRegressor
    model = _GeneralizedLinearRegressor()
    test(model,X,y,coef)

def test_lbfgs():
    X,y,coef = gen_data()
    from glm import _GeneralizedLinearRegressor
    model = _GeneralizedLinearRegressor(solver='lbfgs' )
    test(model,X,y,coef)

def test_Poisson():
    X,y,coef = gen_data()
    # y = jnp.round(jnp.exp(jax.random.poisson(key=jax.random.PRNGKey(1), lam=y)))
    y = jnp.round(jnp.exp(y))
    model = PoissonRegressor()
    test(model,X,y,coef)

def test_gamma():
    X,y,coef = gen_data()
    alpha = 10
    y = np.array([stats.gamma.rvs(a=alpha,scale=y_i/alpha) for y_i in y])
    y = jnp.exp(y)
    model = GammaRegressor()
    test(model,X,y,coef)

def test_Tweedie():
    X,y,coef = gen_data()
    y = jnp.round(jnp.exp(y))
    model = TweedieRegressor()
    test(model,X,y,coef)


def test_sim():
    sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64)
    X,y,coef = gen_data()
    def proc(X,y):
        from glm import _GeneralizedLinearRegressor
        model = _GeneralizedLinearRegressor( solver="newton-cholesky")
        model.fit(X,y)
        # y_pred = model.predict(X)
        coef_fit = model.coef_
        return  coef_fit
    result = spsim.sim_jax(sim, proc)(X,y)



if __name__ == '__main__':
    # test_gamma()
    # test_sim()
    test_glm()
    # test_lbfgs()
    # test_Poisson()
    # test_Tweedie()

