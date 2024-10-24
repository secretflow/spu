# Semi2k
|kernel |     latency      |              comm               |
|-------|------------------|---------------------------------|
|a2b    |(log(K)+1)\*log(N)|(2\*log(K)+1)\*2\*K\*(N-1)\*(N-1)|
|b2a    |1                 |K\*(N-1)                         |
|a2p    |1                 |K\*(N-1)                         |
|b2p    |1                 |K\*(N-1)                         |
|add_bb |log(K)+1          |log(K)\*K\*2+K                   |
|add_aa |0                 |0                                |
|add_ap |0                 |0                                |
|mul_aa |1                 |K\*2\*(N-1)                      |
|mul_ap |0                 |0                                |
|mmul_aa|1                 |K\*2\*(N-1)\*m\*n                |
|mmul_ap|0                 |0                                |
|trunc_a|1                 |K\*(N-1)                         |
|xor_bb |0                 |0                                |
|xor_bp |0                 |0                                |
|and_bb |1                 |K\*2\*(N-1)                      |
|and_bp |0                 |0                                |
# Aby3
|kernel | latency  |     comm     |
|-------|----------|--------------|
|a2b    |log(K)+1+1|log(K)\*K+K\*2|
|b2a    |TODO      |TODO          |
|a2p    |1         |K             |
|b2p    |1         |K             |
|add_bb |log(K)+1  |log(K)\*K\*2+K|
|add_aa |0         |0             |
|add_ap |0         |0             |
|mul_aa |1         |K             |
|mul_ap |0         |0             |
|mmul_aa|1         |K\*m\*n       |
|mmul_ap|0         |0             |
|trunc_a|3         |4\*K          |
|xor_bb |0         |0             |
|xor_bp |0         |0             |
|and_bb |1         |K             |
|and_bp |0         |0             |
