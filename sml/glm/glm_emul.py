import jax.numpy as jnp
import numpy as np
import sys
import os

# Add the parent directory to the system path to import custom modules
sys.path.append('../../')
import sml.utils.emulation as emulation
import spu.utils.distributed as ppd
from glm import _GeneralizedLinearRegressor, PoissonRegressor, GammaRegressor, TweedieRegressor


def emul_SGDClassifier(mode: emulation.Mode.MULTIPROCESS, num=5):
    """
    使用模拟环境执行密文化SGD分类器，并输出结果。

    Parameters:
    -----------
    mode : emulation.Mode.MULTIPROCESS
        模拟环境的运行模式，使用多进程模式运行。
    num : int, optional (default=5)
        输出结果的前num个值。

    Returns:
    -------
    None
    """

    def proc_ncSolver(x1, x2, y):
        """
        使用Newton-Cholesky算法拟合广义线性回归模型，并计算D^2评估指标和预测结果。

        Parameters:
        ----------
        x1 : array-like, shape (n_samples, n_features1)
            特征矩阵1。
        x2 : array-like, shape (n_samples, n_features2)
            特征矩阵2。
        y : array-like, shape (n_samples,)
            目标值。

        Returns:
        -------
        float
            D^2评估指标的结果。
        array-like, shape (n_samples,)
            模型的预测结果。

        """
        X = jnp.concatenate((x1, x2), axis=1)
        model = _GeneralizedLinearRegressor(solver="newton-cholesky")
        model.fit(X, y)
        return model.score(X, y), model.predict(X)

    try:
        # Specify the file paths for cluster and dataset
        CLUSTER_ABY3_3PC = os.path.join('../../', emulation.CLUSTER_ABY3_3PC)
        DATASET_MOCK_REGRESSION_BASIC = os.path.join('../../', emulation.DATASET_MOCK_REGRESSION_BASIC)

        # Create the emulator with specified mode and bandwidth/latency settings
        emulator = emulation.Emulator(
            CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # Prepare the dataset using the emulator
        (x1, x2), y = emulator.prepare_dataset(DATASET_MOCK_REGRESSION_BASIC)

        # Run the proc_ncSolver function using both plaintext and encrypted data
        raw_score, raw_result = proc_ncSolver(ppd.get(x1), ppd.get(x2), ppd.get(y))
        score, result = emulator.run(proc_ncSolver)(x1, x2, y)

        # Print the results
        print("明文D^2: %.2f" % raw_score)
        print("明文结果(前 %s)：" % num, jnp.round(raw_result[:num]))
        print("密态D^2: %.2f" % score)
        print("密态结果(前 %s)：" % num, jnp.round(result[:num]))

    finally:
        emulator.down()


if __name__ == "__main__":
    # Run the emul_SGDClassifier function in MULTIPROCESS mode
    emul_SGDClassifier(emulation.Mode.MULTIPROCESS)
