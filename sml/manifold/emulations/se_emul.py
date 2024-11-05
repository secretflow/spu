import time
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import kneighbors_graph

import sml.utils.emulation as emulation
from sml.manifold.jacobi import normalization, se
from sml.manifold.kneighbors import mpc_kneighbors_graph
import spu.intrinsic as si

def emul_cpz(mode: emulation.Mode.MULTIPROCESS):
    try:
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        def mypermute(x,pi):
            return si.permute(x,pi)
        x = jnp.array([2, 5, 4, 2, 1])
        pi = jnp.array([3, 2, 1, 5, 4])
        sx,spi=emulator.seal(x,pi)
        pi_x = emulator.run(mypermute)(sx,spi)
        print(pi_x)

        def SE(sX, num_samples, num_features, k, num_components):
            Knn = mpc_kneighbors_graph(sX, num_samples, num_features, k)
            D, L = normalization(Knn)
            ans = se(L, num_samples, D, num_components)
            return ans

        # 设置样本数量和维度
        num_samples = 6
        num_features = 3
        k = 3
        num_components = 2
        seed = int(time.time())
        key = jax.random.PRNGKey(seed)
        X = jax.random.uniform(key, shape=(num_samples, num_features), minval=0.0, maxval=1.0)
        # X = np.array(
        #     [
        #         [0.122, 0.114, 0.64],
        #         [0.136, 0.204, 0.25],
        #         [0.11, 0.145, 0.24],
        #         [0.16, 0.81, 0.91],
        #         [0.209, 0.122, 0.76],
        #         [0.148, 0.119, 0.15],
        #     ]
        # )
        #m_ans=SE(X, num_samples, num_features, k, num_components)

        # sX = emulator.seal(X)
        # ans = emulator.run(
        #     SE,
        #     static_argnums=(
        #         1,
        #         2,
        #         3,
        #         4,
        #     ),
        # )(sX, num_samples, num_features, k, num_components)
    
        # print('ans: \n', ans.T)

        # for i in range(num_samples):
        #     print(f"\n验证第 {i+1} 个特征值和特征向量:")
        #     print("A @ v =\n", L @ Q[i, :])
        #     print("λ * v =\n", X2[i][i] * Q[i, :])

        # # sklearn test
        # affinity_matrix = kneighbors_graph(
        #     X, n_neighbors=3, mode="distance", include_self=False
        # )
        # # print('affinity_matrix1: \n',affinity_matrix.toarray())
        # # 使矩阵对称
        # affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
        # # print('affinity_matrix2: \n',affinity_matrix.toarray())
        # embedding = spectral_embedding(
        #     affinity_matrix, n_components=num_components, random_state=None
        # )
        # print('embedding: \n', embedding)
        
        # max_abs_diff = jnp.max(jnp.abs(jnp.abs(embedding) - jnp.abs(ans.T)))
        # print(max_abs_diff)

        # m_max_abs_diff = jnp.max(jnp.abs(jnp.abs(embedding) - jnp.abs(m_ans.T)))
        # print(m_max_abs_diff)
    finally:
        emulator.down()


if __name__ == "__main__":
    emul_cpz(emulation.Mode.MULTIPROCESS)
