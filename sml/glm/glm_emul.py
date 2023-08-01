import jax.numpy as jnp
import numpy as np
import sys
import os
sys.path.append('../../')
import sml.utils.emulation as emulation
import spu.utils.distributed as ppd
from glm import _GeneralizedLinearRegressor,PoissonRegressor,GammaRegressor,TweedieRegressor

def emul_SGDClassifier(mode: emulation.Mode.MULTIPROCESS,num=5):
    def proc_ncSolver(x1,x2,y):
        X = jnp.concatenate((x1, x2), axis=1)
        model = _GeneralizedLinearRegressor(solver="newton-cholesky")
        model.fit(X, y)
        return model.score(X,y),model.predict(X)


    try:
        # bandwidth and latency only work for docker mode
        CLUSTER_ABY3_3PC = os.path.join('../../',emulation.CLUSTER_ABY3_3PC)
        DATASET_MOCK_REGRESSION_BASIC = os.path.join('../../',emulation.DATASET_MOCK_REGRESSION_BASIC)
        emulator = emulation.Emulator(
            CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()
        (x1, x2), y = emulator.prepare_dataset(DATASET_MOCK_REGRESSION_BASIC)
        # log_y = jnp.log(y)
        # round_logy = jnp.round(log_y)
        raw_score,raw_result = proc_ncSolver(ppd.get(x1),ppd.get(x2),ppd.get(y))
        score,result = emulator.run(proc_ncSolver)(x1, x2, y)
        print("明文D^2:%.2f" %raw_score)
        print("明文结果(前 %s)：" %num,jnp.round(raw_result[:num]))
        print("密态D^2:%.2f" %score)
        print("密态结果(前 %s)：" %num,jnp.round(result[:num]))
    finally:
        emulator.down()


if __name__ == "__main__":
    emul_SGDClassifier(emulation.Mode.MULTIPROCESS)
