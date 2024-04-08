# Copyright 2024 Ant Group Co., Ltd.
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

import argparse
import sys

import numpy as np
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prob_label", help="prob-label-test.csv", nargs='?')

if __name__ == "__main__":
    args = parser.parse_args()
    assert args.prob_label
    data = np.genfromtxt(args.prob_label, delimiter=',')
    n = data.shape[1]
    assert n % 2 == 0
    for idx in range(n // 2):
        prob = data[:, idx * 2]
        y = data[:, idx * 2 + 1]
        fpr, tpr, thresholds = metrics.roc_curve(y, prob, pos_label=1)
        med = np.median(thresholds)
        print("{} AUC {}".format(args.prob_label, metrics.auc(fpr, tpr)), end=' ')
    print()
