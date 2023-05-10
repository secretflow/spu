FROM ubuntu:jammy

LABEL maintainer="secretflow-contact@service.alipay.com"

# change dash to bash as default shell
RUN ln -sf /usr/bin/bash /bin/sh

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y gcc-11 g++-11 libasan6 \
    git wget curl unzip autoconf make lld-15 \
    cmake ninja-build nasm vim-common libgl1 libglib2.0-0 iproute2 \
    && apt-get clean \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 \
    && ln -s /usr/bin/ld.lld-15 /usr/bin/ld.lld

# install conda
COPY requirements.txt /tmp/requirements.txt
RUN  sed -i "s/tensorflow=/tensorflow-cpu=/g" /tmp/requirements.txt
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh \
    && bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_23.1.0-1-Linux-x86_64.sh \
    && /root/miniconda3/bin/conda init \
    && /root/miniconda3/bin/pip install -r /tmp/requirements.txt \
    && /root/miniconda3/bin/pip cache purge

# Add conda to path
ENV PATH="/root/miniconda3/bin:${PATH}" 

# Add gcc-11 libstdc++ to miniconda
RUN rm -f /root/miniconda3/lib/libstdc++.so.6 \
    && ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /root/miniconda3/lib/libstdc++.so.6

# This is due to a stupid bazel build https://github.com/bazelbuild/rules_python/issues/691
RUN ln -s /root/miniconda3/bin/python3 /usr/bin/python3

# install bazel 
RUN wget https://github.com/bazelbuild/bazel/releases/download/5.1.1/bazel-5.1.1-installer-linux-x86_64.sh \
    && bash ./bazel-5.1.1-installer-linux-x86_64.sh && rm -f ./bazel-5.1.1-installer-linux-x86_64.sh

# run as root for now
WORKDIR /home/admin/

ENTRYPOINT [ "/bin/bash", "-l" ]
