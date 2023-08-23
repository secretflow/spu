import jax.numpy as jnp

class RBF():
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def __call__(self, X, Y=None):
        if Y == None:
            Y = X
        K = jnp.zeros((X.shape[0], Y.shape[0]), dtype = jnp.float32)
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                K = K.at[i, j].set((jnp.sum((X[i] - Y[j])**2)))
        return jnp.exp(- K / (2 * self.length_scale))

# X = ([[1,1,1],[2,7,5],[3,2,3],[4,0,1]])
# X1 = ([[1,1,1],[2,1,5],[3,2,3]])
# X = jnp.array(X)
# X1 = jnp.array(X1)
# a = RBF()
# print(a(X))
# print(a(X, X1))