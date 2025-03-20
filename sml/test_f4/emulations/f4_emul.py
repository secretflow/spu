
import sml.utils.emulation as emulation
import numpy as np


def emul_f4(mode: emulation.Mode.MULTIPROCESS):
    def proc(x, y):
        # if x == y:
        #     a = 1
        # else:
        #     a = 2
        # b = x * y
        # c = a * b
        return x==y, x/y
    
    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_FANTASTIC4_4PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        np.random.seed(42)
        x = np.random.random()
        y = np.random.random()

        x, y = emulator.seal(x, y)
        res = emulator.run(proc)(x, y)

        print(res)

    finally:
        emulator.down()

if __name__ == "__main__":
    # Run the emul_SGDClassifier function in MULTIPROCESS mode
    emul_f4(emulation.Mode.MULTIPROCESS)
    