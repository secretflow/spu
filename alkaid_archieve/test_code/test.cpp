#include <iostream>
#define EQ_U64(x) static_cast<uint64_t>(x)

int main() {
    uint64_t x = 8055442489310247718;
    x = (EQ_U64(1) << 63) | x;
    std::cout << x << std::endl;

    int64_t y = 8055442489310247718;
    y = EQ_U64((EQ_U64(1) << 63) | y);
    std::cout << EQ_U64(y) << std::endl;
    return 0;
}