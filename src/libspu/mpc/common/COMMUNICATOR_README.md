SPU内置了两种计量通信的方法。
1. libspu/mpc/common/communicator.h提供了Communicator类作为对yacl通信机制的封装，该类会自动记录通信量和轮数。这一类通信计量方式被我们应用于C++侧的基准测试。
    此类通信计量方式存在以下问题：
    1. 该类默认不记录sendAsync和recv的通信轮数和通信量。
    2. 该类默认仅记录粗略的均摊发送通信量。以reduce、allReduce、bcast、gather为例，假设参与方为n方，待发送数组长度为l，忽略数组元素大小：reduce记录通信l，allReduce记录通信nl，bcast记录通信l，gather记录通信l。
2. yacl link context提供了通信机制。该类会自动记录通信量和轮数。这一类通信计量方式被我们应用于Python侧的基准测试。
此类通信计量方式的问题在于其统计发生在send和receive粒度，受集合通信实施方式和调用影响。在这里，两个并行调用的send将会被记录两轮通信轮数（send_actions+=2），而实际上仅有一轮影响。

反映到Alkaid基准测试，上述问题带来了下列挑战：
1. 部分协议（包括ABY3）需要调用sendAsync和recv，而由这些调用带来的通信轮数和通信量开销实际未计入C++侧的统计。尽管SPU提供了addCommStatsManually用于手动统计通信量，但可惜并非所有调用sendAsync和recv的关键位置都合理手动统计了通信量（e.g. boolean B2V; conversion PPA）。为解决这个问题，我们为sendAsync和recv注入了通信量统计（并抹去了ABY3中手动统计的通信量以确保正确）。
2. Alkaid的部分协议的非对称性质决定了协议的执行时间实际受关键路径，而简单的均摊开销不能反映关键路径长度。除此之外，仅统计发送也不合理。
3. 