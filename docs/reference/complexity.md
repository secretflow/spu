# Semi2k
| kernel  |     latency      |              comm               |
|---------|------------------|---------------------------------|
|a2b      |(log(k)+1)\*log(n)|(2\*log(k)+1)\*2\*k\*(n-1)\*(n-1)|
|b2a      |1                 |k\*(n-1)                         |
|a2p      |1                 |k\*(n-1)                         |
|b2p      |1                 |k\*(n-1)                         |
|add_bb   |log(k)+1          |log(k)\*k                        |
|add_aa   |0                 |0                                |
|add_ap   |0                 |0                                |
|mul_aa   |1                 |k\*2\*(n-1)                      |
|mul_ap   |0                 |0                                |
|mmul_aa  |TODO              |TODO                             |
|mmul_ap  |0                 |0                                |
|truncpr_a|0                 |0                                |
|xor_bb   |0                 |0                                |
|xor_bp   |0                 |0                                |
|and_bb   |1                 |k\*2\*(n-1)                      |
|and_bp   |0                 |0                                |
# Aby3
| kernel  | latency  |      comm       |
|---------|----------|-----------------|
|a2b      |log(k)+1+1|log(k)\*k\*2+k\*2|
|b2a      |2         |2\*k\*k+k        |
|a2p      |1         |k                |
|b2p      |1         |k                |
|add_bb   |log(k)+1  |log(k)\*k\*2+k   |
|add_aa   |0         |0                |
|add_ap   |0         |0                |
|mul_aa   |1         |k                |
|mul_ap   |0         |0                |
|mmul_aa  |TODO      |TODO             |
|mmul_ap  |0         |0                |
|truncpr_a|3         |4\*k             |
|xor_bb   |0         |0                |
|xor_bp   |0         |0                |
|and_bb   |1         |k                |
|and_bp   |0         |0                |
