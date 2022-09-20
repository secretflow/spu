#! /bin/bash
#
# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
RED='\033[0;31m'
NC='\033[0m'

function build_spu(){
    echo -e "${RED}Start build wheel package...${NC}"

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        docker run --rm --mount type=bind,source="$(pwd)",target=/home/admin/dev/ \
        -w /home/admin/dev \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --cap-add=NET_ADMIN \
        --privileged=true \
        --entrypoint "./build_wheel_entrypoint.sh" \
        registry.hub.docker.com/secretflow/sf-gcc11-centos7-amd64-release:latest
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        sh ./build_wheel_entrypoint.sh
    fi    

    (($? != 0)) && { echo -e "${RED}Build exited with non-zero.${NC}"; exit 1; }

    spu_wheel_name=$(<./spu_wheel.name)
    spu_wheel_path="./${spu_wheel_name//sf-spu/sf_spu}"
}

function install_spu(){
    echo -e "${RED}Installing $spu_wheel_path...${NC}" 

    python3 -m pip install $spu_wheel_path --force-reinstall
}

function upload_spu(){
    echo -e "${RED}Uploading package $spu_wheel_path to pypi...${NC}"

    twine upload -r pypi $spu_wheel_path
}


iflag=
uflag=
while getopts iu name
do
    case $name in
    i)  iflag=1;;
    u)  uflag=1;;
    ?)  printf "Usage: %s: [-i] [-u]\n" $0
        echo "-i   build and install locally."
        echo "-u   build and upload to pypi."
        exit 2;;
    esac
done
build_spu
if [ ! -z "$iflag" ]; then
    install_spu
fi
if [ ! -z "$uflag" ]; then
    upload_spu
fi

