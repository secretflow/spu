# exp

|    id    |Pass Rate|  Description   |
|----------|--------:|----------------|
|Testcase 1|   0.9219|uniform(-10, 20)|

# reciprocal

|    id    |Pass Rate|     Description      |
|----------|--------:|----------------------|
|Testcase 1|    1.000|uniform(0, 1e3)       |
|Testcase 2|    0.259|uniform(1e3, 1e6)     |
|Testcase 3|    0.000|uniform(1e6, 1e9)     |
|Testcase 4|    1.000|uniform(-1e3, 0)      |
|Testcase 4|    1.000|uniform(-2**18, 2**18)|

# div

|    id    |Pass Rate|                   Description                    |
|----------|--------:|--------------------------------------------------|
|Testcase 1|   0.4339|both x and y belongs to uniform(-2**18, 2**18)    |
|Testcase 2|   1.0000|x: uniform(-2**18, 2**18), y: uniform(-2**9, 2**9)|
|Testcase 3|   0.6447|x: uniform(-2**9, 2**9), y: uniform(-2**18, 2**18)|

# log

|    id    |Pass Rate|   Description   |
|----------|--------:|-----------------|
|Testcase 1|        1|uniform(0, 2**18)|

# logistic

|    id    |Pass Rate|  Description   |
|----------|--------:|----------------|
|Testcase 1|    0.967|uniform(-10, 10)|
