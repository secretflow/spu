import subprocess
import glob
import os
import itertools
from enum import Enum
import time
import sys
import threading
from extract_inf import split_and_extract
import builtins
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import re

"""include datetime in every print"""
# Save the original print function
original_print = builtins.print

def print_with_timestamp(*args, sep=' ', end='\n', file=None, flush=True):
    # Get the current time formatted as desired
    current_time = datetime.now(ZoneInfo('Asia/Shanghai')).strftime("[%Y-%m-%d %H:%M:%S]")
    
    # Prepend the timestamp to the original arguments
    original_args = (current_time, ) + args
    
    # Call the original print function with the new arguments
    original_print(*original_args, sep=sep, end=end, file=file, flush=flush)

# Override the built-in print with the new function
builtins.print = print_with_timestamp

sys.stdout = open('test-log.txt', 'w')

test_file_list = ["sml/svm/emulations/svm_emul.py"]

test_file_dict = {}
for test_file in test_file_list:
    file_inf = test_file.split("/")
    test_file_name = file_inf[-1].split('.')[0]
    test_file_command = "/".join(["bazel-bin", "sml", file_inf[-3], f"{file_inf[-2]}", f"{test_file_name}"])
    test_file_dict[test_file] = test_file_command

# bandwidth = "50000"
# latency = "20"
# config = "2pc_semi2k"
# change = "True"

config_combine = [
    ("5", "100", "2pc_semi2k", "False"),
    ("5", "100", "2pc_semi2k", "True"),
    ("5", "20", "2pc", "False"),
    ("5", "20", "2pc", "True"),
    ("5", "200", "2pc_semi2k", "False"),
    ("5", "200", "2pc_semi2k", "True"),
    ("5", "300", "2pc_semi2k", "False"),
    ("5", "300", "2pc_semi2k", "True"),
]

for test_file in test_file_list:
    # for i in range(5):
    #     for bandwidth in ["300", "100"]:
    #         for config in ["2pc_semi2k", "2pc"]:
    #             for change in ["False", "True"]:
    for (bandwidth, latency, config, change) in config_combine:
        with open("file-to-be-modified/" + test_file, "r") as file:
            ori_context = file.read()
        changed_context = ori_context.replace("{bandwidth}", bandwidth).replace("{latency}", latency).replace("{config}", config).replace("{change}", change)
        with open("../" + test_file, "w") as file:
            file.write(changed_context)
        original_work_dir = os.getcwd()
        work_dir = ".."
        os.chdir(work_dir)
        print(f"running with bandwidth: {bandwidth}, latency: {latency}, config: {config}, change: {change}")
        process = subprocess.Popen(test_file_dict[test_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = process.stdout.read().decode("utf-8")
        print(output)
        os.chdir(original_work_dir)
