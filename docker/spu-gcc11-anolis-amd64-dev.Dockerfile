FROM openanolis/anolisos:8.4-x86_64

# GCC version
ARG DEVTOOLSET_VERSION=11

# First update repos
RUN yum -y update

# Update other packages
RUN yum -y update

RUN yum makecache

# install devtools and [enable it](https://access.redhat.com/solutions/527703)
RUN yum install -y \
    gcc-toolset-${DEVTOOLSET_VERSION}-gcc \
    gcc-toolset-${DEVTOOLSET_VERSION}-gcc-c++ \
    gcc-toolset-${DEVTOOLSET_VERSION}-binutils \
    gcc-toolset-${DEVTOOLSET_VERSION}-libatomic-devel \
    gcc-toolset-${DEVTOOLSET_VERSION}-libasan-devel \
    gcc-toolset-${DEVTOOLSET_VERSION}-libubsan-devel \
    && echo "source scl_source enable gcc-toolset-${DEVTOOLSET_VERSION}" > /etc/profile.d/enable_gcc_toolset.sh

# install common tools
RUN yum install -y vim git wget unzip which \
    && yum install -y libtool autoconf make cmake ninja-build \
    && yum install -y nasm

# install python3-devtools
RUN yum install -y python38 python38-devel
RUN ln -s /usr/bin/python3 /usr/bin/python

# NOTE, ant aci will not source /etc/profile.d, so add scl to `/etc/profile.d/` does not work
ENV PATH /opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# install python packages
RUN python3 -m pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN python3 -m pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip config set install.trusted-host "mirrors.aliyun.com pypi.tuna.tsinghua.edu.cn"
RUN python3 -m pip install --upgrade pip
# it's not in pip freeze list, but required by setuptools
RUN python3 -m pip install wheel
COPY requirements.txt /tmp
RUN python3 -m pip install --requirement /tmp/requirements.txt

# install bazel 
RUN yum install -y java-11-openjdk-devel.x86_64
RUN wget https://github.com/bazelbuild/bazel/releases/download/5.2.0/bazel-5.2.0-installer-linux-x86_64.sh \
    && chmod +x ./bazel-5.2.0-installer-linux-x86_64.sh && ./bazel-5.2.0-installer-linux-x86_64.sh

RUN echo -e "\
[CentOSAppStream] \n\
name=CentOS-8.5.2111 - AppStream - mirrors.aliyun.com \n\
baseurl=http://mirrors.aliyun.com/centos-vault/8.5.2111/AppStream/\$basearch/os/ \n\
gpgcheck=0 \n\
gpgkey=http://mirrors.aliyun.com/centos/RPM-GPG-KEY-CentOS-Official \n\
" > /etc/yum.repos.d/CentOS-AppStream.repo

RUN dnf install -y epel-release

RUN yum install -y lcov


# run as root for now
WORKDIR /home/admin/

CMD [ "/sbin/init" ]
