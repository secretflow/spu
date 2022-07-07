FROM centos:centos7

# GCC version
ARG DEVTOOLSET_VERSION=11

RUN yum install -y centos-release-scl
RUN yum install -y epel-release

# install devtools and [enable it](https://access.redhat.com/solutions/527703)
RUN yum install -y \
    devtoolset-${DEVTOOLSET_VERSION}-gcc \
    devtoolset-${DEVTOOLSET_VERSION}-gcc-c++ \
    devtoolset-${DEVTOOLSET_VERSION}-binutils \
    devtoolset-${DEVTOOLSET_VERSION}-libatomic-devel \
    devtoolset-${DEVTOOLSET_VERSION}-libasan-devel \
    devtoolset-${DEVTOOLSET_VERSION}-libubsan-devel \
    && echo "source scl_source enable devtoolset-${DEVTOOLSET_VERSION}" > /etc/profile.d/enable_gcc_toolset.sh

# install common tools
RUN yum install -y vim git wget unzip which java-11-openjdk-devel.x86_64 \
    && yum install -y libtool autoconf make cmake3 ninja-build lcov \
    && yum install -y nasm

RUN ln -s /usr/bin/cmake3 /usr/bin/cmake

# install python3-devtools
RUN yum install -y rh-python38-python-devel.x86_64 rh-python38-python-pip.noarch \
    && echo "source scl_source enable rh-python38" > /etc/profile.d/enable_py_toolset.sh

RUN yum clean all


ENV PATH /opt/rh/devtoolset-${DEVTOOLSET_VERSION}/root/usr/bin:/opt/rh/rh-python38/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# install python packages
RUN python3 -m pip install --upgrade pip
# it's not in pip freeze list, but required by setuptools
RUN python3 -m pip install wheel
COPY requirements.txt /tmp
RUN python3 -m pip install --requirement /tmp/requirements.txt

# install bazel 
RUN wget https://github.com/bazelbuild/bazel/releases/download/5.2.0/bazel-5.2.0-installer-linux-x86_64.sh \
    && chmod +x ./bazel-5.2.0-installer-linux-x86_64.sh && ./bazel-5.2.0-installer-linux-x86_64.sh

RUN wget --no-check-certificate https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.gz \
    && tar zxf nasm-2.15.05.tar.gz \
    && cd nasm-2.15.05 \
    && ./configure \
    && make install \
    && rm -rf nasm-2.15.05 \
    && rm -rf nasm-2.15.05.tar.gz

# run as root for now
WORKDIR /home/admin/

CMD [ "/bin/bash" ]
