import jax
import jax.numpy as jnp


def mpc_kneighbors_graph(
    X,  # 要计算最近邻的输入样本
    num_samples,  # 样本数量
    num_features,  # 样本维度
    n_neighbors,  # 定义最近邻的个数，不包括样本自己
    *,
    mode="distance",
    metric="minkowski",  # 距离定义为样本之间的欧几里得距离
    p=2,
):

    # 计算每两个samples之间的欧几里得距离的平方
    X_expanded = jnp.expand_dims(X, axis=1) - jnp.expand_dims(X, axis=0)
    X_expanded = jnp.square(X_expanded)
    Dis = jnp.sum(X_expanded, axis=-1)

    # 对Dis的每一行进行排序，首先计算置换，之后将置换应用于Dis
    Indix_Dis = jnp.argsort(Dis, axis=1)

    def permute_rowwise(Dis_row, Indix_row):
        return jnp.take(Dis_row, Indix_row)

    Knn = jax.vmap(permute_rowwise)(Dis, Indix_Dis)

    # 对之前求的最近邻的欧几里得距离的平方求平方根，非最近邻的距离设置为0
    Knn2 = jnp.zeros((num_samples, num_samples))
    # for i in range(num_samples):
    #     for j in range(num_samples):
    #         if j <= n_neighbors:
    #             Knn2 = Knn2.at[i, j].set(jnp.sqrt(Knn[i][j]))
    #         else:
    #             Knn2 = Knn2.at[i, j].set(0)

    # Assume Knn and Knn2 are initialized appropriately
    def update_knn_row(i, Knn_row, n_neighbors):
        def update_element(j, Knn_value):
            return jnp.where(j <= n_neighbors, jnp.sqrt(Knn_value), 0)

        # Vectorize the inner loop over `j`
        Knn_row_updated = jax.vmap(update_element, in_axes=(0, 0))(
            jnp.arange(Knn_row.shape[0]), Knn_row
        )
        return Knn_row_updated

    # Vectorize the outer loop over `i`
    Knn2 = jax.vmap(lambda i, Knn_row: update_knn_row(i, Knn_row, n_neighbors))(
        jnp.arange(num_samples), Knn
    )

    # 对Dis进行逆置换，恢复之前的顺序
    def inverse_permutation(Indix_row):
        inverse = jnp.argsort(Indix_row)
        return inverse

    Indix_Inv = jax.vmap(inverse_permutation)(Indix_Dis)
    Knn3 = jax.vmap(permute_rowwise)(Knn2, Indix_Inv)

    # 使最近邻矩阵对称
    Knn4 = 0.5 * (Knn3 + Knn3.T)
    return Knn4
