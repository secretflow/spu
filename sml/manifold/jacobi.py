import jax
import jax.numpy as jnp
import numpy as np

# def SelectElement(X, num_samples):
#     num_zero = 0
#     for i in range(num_samples):
#         for j in range(num_samples):
#             num_zero += jnp.equal(X[i][j], 0)
#     X = X.flatten()
#     X_ij = jnp.zeros((num_samples * num_samples, 2))
#     for i in range(num_samples):
#         for j in range(num_samples):
#             X_ij = X_ij.at[i * num_samples + j, 0].set(i)
#             X_ij = X_ij.at[i * num_samples + j, 1].set(j)
#     perm = jnp.argsort(X)
#     X_ij = jnp.take(X_ij, perm, axis=0)
#     return X_ij, num_zero


# def get_ij(X_ij, i, num_samples, num_zero):
#     return (
#         X_ij[num_samples * (num_samples - 1) - i - num_zero - 1][0],
#         X_ij[num_samples * (num_samples - 1) - i - 1 - num_zero][1],
#     )


def set_value_2d(x, index_1, index_2, value, n):
    # 将x[index_1][index_2]处的值设置为value，其中index_1和index_2是秘密共享的

    # 创建两维度的索引矩阵
    row_indices = jnp.zeros((n, n), dtype=jnp.int16)
    col_indices = jnp.zeros((n, n), dtype=jnp.int16)

    for i in range(n):
        for j in range(n):
            row_indices = row_indices.at[i, j].set(i)  # 设置行索引
            col_indices = col_indices.at[i, j].set(j)  # 设置列索引

    # 比较行和列索引是否等于传入的秘密共享索引
    flag_1 = jnp.equal(row_indices, index_1)
    flag_2 = jnp.equal(col_indices, index_2)

    # 同时满足两个索引条件
    flag = jnp.logical_and(flag_1, flag_2)

    # 根据标志位设置数组值
    set_x = jnp.select([flag], [value], x)

    return set_x


# def Rotation_Matrix(X, k, l, n):
#     #根据选择的X[k][l]计算旋转矩阵J
#     J = jnp.eye(n)
#     tar_elements = X[k][l]
#     tar_diff = X[k][k] - X[l][l]
#     # cos_2theta=jnp.abs(tar_diff)*jnp.reciprocal(jnp.sqrt(4*tar_elements*tar_elements+tar_diff*tar_diff))
#     cos_2theta = jnp.reciprocal(
#         jnp.sqrt(
#             1
#             + 4
#             * tar_elements
#             * tar_elements
#             * jnp.reciprocal(tar_diff)
#             * jnp.reciprocal(tar_diff)
#         )
#     )
#     cos2 = 0.5 + 0.5 * cos_2theta
#     sin2 = 0.5 - 0.5 * cos_2theta
#     flag_zero = jnp.equal(tar_elements, 0)
#     cos = jnp.sqrt(cos2) * (1 - flag_zero) + flag_zero
#     sin = (
#         (jnp.where(jnp.logical_and(tar_elements == 0, tar_diff == 0), 0, 1))
#         * jnp.sqrt(sin2)
#         * ((jnp.greater(tar_elements * tar_diff, 0)) * 2 - 1)
#     )

#     J = set_value_2d(J, k, k, cos, n)
#     J = set_value_2d(J, l, l, cos, n)
#     J = set_value_2d(J, k, l, -sin, n)
#     J = set_value_2d(J, l, k, sin, n)
#     return J


def compute_elements(X, k, l, n):
    tar_elements = X[k][l]
    tar_diff = X[k][k] - X[l][l]

    cos_2theta = jnp.reciprocal(
        jnp.sqrt(
            1
            + 4*jnp.square(tar_elements*jnp.reciprocal(tar_diff))
        )
    )
    cos2 = 0.5 + 0.5 * cos_2theta
    sin2 = 0.5 - 0.5 * cos_2theta
    flag_zero = jnp.equal(tar_elements, 0)
    cos = jnp.sqrt(cos2) * (1 - flag_zero) + flag_zero
    sin = (
        (jnp.where(jnp.logical_and(tar_elements == 0, tar_diff == 0), 0, 1))
        * jnp.sqrt(sin2)
        * ((jnp.greater(tar_elements * tar_diff, 0)) * 2 - 1)
    )

    return cos, sin


def update_J(J, k, l, cos, sin, n):
    J = set_value_2d(J, k, k, cos, n)
    J = set_value_2d(J, l, l, cos, n)
    J = set_value_2d(J, k, l, -sin, n)
    J = set_value_2d(J, l, k, sin, n)
    return J


def Rotation_Matrix(X, k, l, n, m,k_0,l_0):
    # 根据选择的X[k][l]计算旋转矩阵J
    J = jnp.eye(n)
    k_values = jnp.array(k)  # 确保 k 和 l 是 JAX 数组
    l_values = jnp.array(l)
    # 使用 vmap 进行并行化
    cos_values, sin_values = jax.vmap(compute_elements, in_axes=(None, 0, 0, None))(
        X, k_values, l_values, n
    )
    # 更新 J
    for i in range(len(k_values)):
        # t_k=k_0-i
        # t_l=l_0+i
        # J=J.at[k_values[i],k_values[i]].set(cos_values[i])
        # J=J.at[l_values[i],l_values[i]].set(cos_values[i])
        # J=J.at[k_values[i],l_values[i]].set(-sin_values[i])
        # J=J.at[l_values[i],k_values[i]].set(sin_values[i])
        # should not be here x=Value<1x1xSF32,s=0,0>, to=Pub2k<FM64>
        J = update_J(J, k_values[i], l_values[i], cos_values[i], sin_values[i], n)


    return J


def Jacobi(X, num_samples):
    Q = jnp.eye(num_samples)
    k = 0
    while k < 5:
        for i in range(1, 2 * num_samples - 2):
            if i < num_samples:
                l_0 = i
                r_0 = 0
            else:
                l_0 = num_samples - 1
                r_0 = i - l_0
            
            n = (l_0 - r_0 - 1) // 2 + 1
            l = jnp.zeros(n, dtype=jnp.int16)
            r = jnp.zeros(n, dtype=jnp.int16)
            # 选取索引各不相同的位置
            for j in range(0, n):
                l = l.at[j].set(l_0 - j)
                r = r.at[j].set(r_0 + j)
            # 计算旋转矩阵
            J = Rotation_Matrix(X, l, r, num_samples, n,l_0,r_0)
            # 用旋转矩阵更新X和Q
            X = jnp.dot(J.T, jnp.dot(X, J))
            Q = jnp.dot(J.T, Q)
        k = k + 1

    return X, Q


def se(X, num_samples, D, n_components):
    X, Q = Jacobi(X, num_samples)
    X = jnp.diag(X)
    X = jnp.array(X)
    perm = jnp.argsort(X)

    ans = jnp.take(Q, perm[1 : n_components + 1], axis=0)

    D = jnp.diag(D)
    ans = ans * jnp.reciprocal(jnp.sqrt(D))
    X = jnp.diag(X)
    return ans


def normalization(
    adjacency,  # 邻接矩阵
    norm_laplacian=True,  # 如果为 True，使用对称归一化拉普拉斯矩阵；如果为 False，使用非归一化的拉普拉斯矩阵。
):
    D = jnp.sum(adjacency, axis=1)
    D = jnp.diag(D)

    L = D - adjacency
    D2 = jnp.diag(jnp.reciprocal(jnp.sqrt(jnp.diag(D))))
    if norm_laplacian == True:
        # 归一化
        L = jnp.dot(D2, L)
        L = jnp.dot(L, D2)
    return D, L
