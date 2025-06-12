# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import List, Tuple, Union
import numpy as np

from spu.experimental import reveal as spu_reveal


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    syslog = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s]-[%(levelname)s]-[%(filename)s:%(lineno)d]: %(message)s'
    )
    syslog.setFormatter(formatter)
    logger.addHandler(syslog)
    logger.propagate = False

    return logger


def sml_reveal(
    x: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]],
) -> Union[np.ndarray, List[np.ndarray]]:
    if isinstance(x, list) or isinstance(x, tuple):
        return [spu_reveal(item) for item in x]
    else:
        return spu_reveal(x)
