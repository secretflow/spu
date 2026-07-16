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

refer_folder = "test-log-new/new-auto-test-algebra"
output_folder = "test-log-new/new-auto-test-algebra-r2"
for log_sub_directory in glob.glob(os.path.join(output_folder, '*/')):
    extract_result_path = os.path.join(log_sub_directory, 'extract_result.json')
    with open(extract_result_path, 'r') as file:
        extract_result = json.load(file)
    extract_result_path_ref = os.path.join(log_sub_directory.replace(output_folder, refer_folder), 'extract_result.json')
    with open(extract_result_path_ref, 'r') as file:
        extract_result_ref = json.load(file)
    
    for func_name in extract_result:
        for pass_name in extract_result[func_name]:
            if len(extract_result[func_name][pass_name]["deduplication"]) != 1:
                print(f"Function {func_name} pass {pass_name} deduplication not 1")
            if pass_name not in extract_result_ref[func_name]:
                print(f"Function {func_name} pass {pass_name} not in ref")
        for pass_name in extract_result_ref[func_name]:
            if pass_name not in extract_result[func_name]:
                print(f"Function {func_name} pass {pass_name} not in new")
        if func_name not in extract_result_ref:
            print(f"Function {func_name} not in ref")
    for func_name in extract_result_ref:
        for pass_name in extract_result_ref[func_name]:
            if pass_name not in extract_result[func_name]:
                print(f"Function {func_name} pass {pass_name} not in new")
        for pass_name in extract_result[func_name]:
            if pass_name not in extract_result_ref[func_name]:
                print(f"Function {func_name} pass {pass_name} not in ref")
        if func_name not in extract_result:
            print(f"Tests for Function {func_name} not in new")