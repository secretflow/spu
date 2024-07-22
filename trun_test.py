import random

R = 1 << 32
EXP_TIMES = 10000

def r_rand():
    return random.randint(0, R - 1)

def r_add(x, y):
    return (x + y) % R

def r_mul(x, y):
    return (x * y) % R

def r_lshift(x, shift):
    return (x << shift) % R

def r_neg(x):
    return (-x) % R

def fxp2flp(x, fxp_pre):
    return (-1) ** (x >> 63) * x / (1 << fxp_pre)

def flp2fxp(x, fxp_pre):
    return int(x * (1 << fxp_pre)) % R

def trun_plain(x, d):
    return x / d

def trun_sml(x, d):
    x0, x1 = r_rand(), r_add(x, -x0)
    return r_add(x0 / d, x1 / d)

def print_bit(x):
    print(f"{x:0b}".rjust(64, '0'))

def print_0x(x):
    print(f"{x:0x}".rjust(16, '0'))

fxp2flp(flp2fxp(0.5, 12), 12)

def exp():
    for i in range(EXP_TIMES):
        x = random.random()