import random

# a gate used to generate signal g in ExtMsb PPA circuit.
def evaluate_g(g, p):
    assert(len(p) == 4 & len(g) == 4)
    return g[3] ^ (g[2] & p[3]) ^ (g[1] & p[2] & p[3]) ^ (g[0] & p[1] & p[2] & p[3])

# a gate used to generate signal p in ExtMsb PPA circuit.
def evaluate_p(p):
    assert(len(p) == 4)
    return p[0] & p[1] & p[2] & p[3]

def lshift(x, shift):
    return (x << shift) % (1 << 64)

def rshift(x, shift):
    return x >> shift

# hitchhiking select (lshift mask, select, lshift)
def select(x, mask, stride, ith):
    return lshift(x & lshift(mask, ith * stride), (3 - ith) * stride)

# select and lshift (select, lshift)
def select2(x, mask, stride, ith):
    return lshift(x & mask, (3 - ith) * stride)

def print_bool(x):
    print(f"{x:064b}")
    return f"{x:064b}"

def print_hex(x):
    print(f"{x:016x}")
    return f"{x:016x}"
    
def evaluate_msb(x, y):
    return ((x + y) & 0x8000000000000000) >> 63

# a gate used to generate signal g in A2B PPA circuit.
def evaluate_g_gate(g, p):
    s0 = g[0]
    s1 = g[1] ^ (g[0] & p[1])
    s2 = g[2] ^ (g[1] & p[2]) ^ (g[0] & p[1] & p[2])
    s3 = g[3] ^ (g[2] & p[3]) ^ (g[1] & p[2] & p[3]) ^ (g[0] & p[1] & p[2] & p[3])
    return rshift(s0, 3) ^ rshift(s1, 2) ^ rshift(s2, 1) ^ rshift(s3, 0)

# a gate used to generate signal p in A2B PPA circuit.
def evaluate_p_gate(p):
    return rshift(p[0], 3) ^ \
        rshift(p[0] & p[1], 2) ^ \
            rshift(p[0] & p[1] & p[2], 1) ^ \
                rshift(p[0] & p[1] & p[2] & p[3], 0)
                


def evaluate_ppa_msb(x, y, k=64):
    print("--------------------------")
    print("initial")
    p = x ^ y
    g = x & y
    result = p & 0x8000000000000000
    print(f"{p=}")
    print(f"{g=}")
    p |= 1 << k
    g &= (1 << k) - 1
       
    mask = [
        0x1111111111111111,
        0x0008000800080008,
        0x0000000000008000,
    ]
    
    bit_offset = [
        1, 4, 16
    ]
    
    lev = 0
    while k > 1:
        print(f"lev: {lev} input")
        print(f"{p=}")
        print(f"{g=}")
        pops, gops = [0, 0, 0, 0], [0, 0, 0, 0]
        for i in range(4):
            pops[i] = select(p, mask[lev], bit_offset[lev], i)
            gops[i] = select(g, mask[lev], bit_offset[lev], i)
    
        p = evaluate_p(pops)
        g = evaluate_g(gops, pops)
        
        print(f"{p=}")
        print(f"{g=}")
        
        lev += 1
        k = int(k / 4)

    print(f"{p=}")
    print(f"{g=}")
    
    result = (g ^ result) >> 63
    return result

def evaluate_ppa(x, y, k=64):
    print("--------------------------")
    print("initial")
    p = x ^ y
    g = x & y
    result = p
    print(f"{p=}")
    print(f"{g=}")
    
    mask = [
        0x1111111111111111,
        0x8888888888888888,
        0x8888888888888888,
    ]
    
    mask_epilog = [
        0x8888888888888888,         # last level mask for aggereated gate
        0x7777777777777777,         # last level mask for single gate
    ]
    
    bit_offset = [
        1, 4, 16
    ]

    print(f"lev: 0")
    lev = 0
    pops, gops = [0, 0, 0, 0], [0, 0, 0, 0]
    for i in range(4):
        pops[i] = select(p, mask[lev], bit_offset[lev], i)
        gops[i] = select(g, mask[lev], bit_offset[lev], i)
    p = evaluate_p_gate(pops)
    g = evaluate_g_gate(gops, pops)
    print(f"{p=}")
    print(f"{g=}")
    
    print(f"lev: 1")
    lev = 1
    pops, gops = [0, 0, 0, 0], [0, 0, 0, 0]
    for i in range(4):
        pops[i] = select2(p, mask[lev], bit_offset[lev], i)
        gops[i] = select2(g, mask[lev], bit_offset[lev], i)
    p = evaluate_p(pops) ^ p & (~mask[lev])
    g = evaluate_g(gops, pops) ^ g & (~mask[lev])
    print(f"{p=}")
    print(f"{g=}")
    
    print(f"lev: 2")
    lev = 2
    pops, gops = [0, 0, 0, 0], [0, 0, 0, 0]
    for i in range(4):
        pops[i] = select2(p, mask[lev], bit_offset[lev], i)
        gops[i] = select2(g, mask[lev], bit_offset[lev], i)
    p = evaluate_p(pops) ^ p & (~mask[lev])
    g = evaluate_g(gops, pops) ^ g & (~mask[lev])
    print(f"{p=}")
    print(f"{g=}")
    
    print(f"lev: 3")
    gops0 = g & mask_epilog[0]
    gops0 = lshift(gops0, 1) ^ lshift(gops0, 2) ^ lshift(gops0, 3)
    pops1, gops1 = p & mask_epilog[1], g & mask_epilog[1]
    g = (gops1 ^ pops1 & gops0) ^ (g & mask_epilog[0])
    c = lshift(g, 1)
    print(f"{p=}")
    print(f"{g=}")
    print(f"{c=}")
    
    result = result ^ c
    return result

def exp(times):
    match_count = 0
    positive_count = 0
    for i in range(times):
        x = random.randint(0, 0xFFFFFFFFFFFFFFFF)
        y = random.randint(0, 0xFFFFFFFFFFFFFFFF)
        z = (x + y) % (1 << 64)
        msbz = (z & 0x8000000000000000) >> 63
        msbppa = evaluate_ppa_msb(x, y, 64)
        if msbz == msbppa:
            match_count += 1
        # else:
        #     print("Mismatch!")
        #     print(x, y)
        if msbz == 0:
            positive_count += 1
    print(f"match rate: {match_count / times}")
    print(f"positive rate: {positive_count / times}")
    
def exp_a2b(times):
    match_count = 0
    positive_count = 0
    for i in range(times):
        x = random.randint(0, 0xFFFFFFFFFFFFFFFF)
        y = random.randint(0, 0xFFFFFFFFFFFFFFFF)
        z = (x + y) % (1 << 64)
        r = evaluate_ppa(x, y, 64)
        if z == r:
            match_count += 1
        
    print(f"match rate: {match_count / times}")

# exp_a2b(100)
x = 15967636371270604181
y = 2479107702438947437
print("result...")
print((x + y) % (1 << 64))
print(evaluate_ppa(x, y, 63))
        
        
    