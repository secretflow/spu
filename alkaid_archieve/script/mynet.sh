# https://zhuanlan.zhihu.com/p/588714270

#!/bin/bash
######
# Taken from https://github.com/emp-toolkit/emp-readme/blob/master/scripts/throttle.sh
######

# tc qdisc show dev lo

## replace DEV=lo with your card (e.g., eth0)
DEV=lo 
if [ "$1" == "del" ]
then
	sudo tc qdisc del dev $DEV root
fi

if [ "$1" == "lan" ]
then
sudo tc qdisc del dev $DEV root

# Alkaid 1 gbps bandwidth and 1 ms rtt
# sudo tc qdisc add dev $DEV root handle 1: tbf rate 1gbit burst 100000 limit 10000
# sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 0.5msec
# Cheetah trun test 1 gbps bandwidth and 2 ms rtt
sudo tc qdisc add dev $DEV root handle 1: tbf rate 1gbit burst 100000 limit 10000
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 1msec
fi

if [ "$1" == "wan" ]
then
sudo tc qdisc del dev $DEV root
# Alkaid 160 mbps bandwidth and 0.1 s rtt
# sudo tc qdisc add dev $DEV root handle 1: tbf rate 160mbit burst 100000 limit 10000
# sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 50msec
# Cheetah trun test 1 gbps bandwidth and 40 ms rtt
sudo tc qdisc add dev $DEV root handle 1: tbf rate 1gbit burst 100000 limit 10000
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 20msec
fi

if [ "$1" == "man" ]
then
sudo tc qdisc del dev $DEV root
# # Alkaid 400 mbps bandwidth and 10 ms rtt
# sudo tc qdisc add dev $DEV root handle 1: tbf rate 400mbit burst 100000 limit 10000
# sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 5msec
# Cheetah trun test 100 mbps bandwidth and 40 ms rtt
sudo tc qdisc add dev $DEV root handle 1: tbf rate 100mbit burst 100000 limit 10000
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 20msec
fi