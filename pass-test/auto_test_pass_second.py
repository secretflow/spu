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
class mod(Enum):
    sml = 1
    example = 2 

"""Define the parameters"""
test_mod = mod.sml                                      # test on sml or example
repeat_time = 1                                         # repeat time for each test, usually 1 for only testing communication cost
output_folder = "test-log-new/new-auto-test-all-SEMI2K-second"            # specify the output folder
pass_ref_folder = "test-log-new/new-auto-test-all-SEMI2K"            # specify the folder for the reference on the first passoption_choice
thread_num = 6                                         # number of threads
protocal = "SEMI2K"                                    # SEMI2K/ABY3/CHEETAH/SECURENN
passoptions_path = '/home1/leiyu.lyc/ppu/pass-test/pass_options.txt' # path to pass options
# number of parties
if protocal == "ABY3":
    partynum = 3
else:
    partynum = 2

hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

def IR_record(output_folder_path, pass_option):
    passoption_folder_path = os.path.join(output_folder_path, pass_option)
    with open(os.path.join(passoption_folder_path, 'extract_result.json'), 'r') as file:
        test_result_dict = json.load(file)
    IR_record_folder_path = os.path.join(output_folder_path, "IR_record")
    if pass_option == "baseline":
        if not os.path.exists(IR_record_folder_path):
            os.makedirs(IR_record_folder_path)
        # save pphlo for each function to a separate file for IR deduplication in the later test
        for function_name in test_result_dict:
            pphlo_log = ""
            for IR_instruction in test_result_dict[function_name]['pphlo']:
                IR_instruction = re.sub(r'dense<".*?">', 'dense<neglected>', IR_instruction)
                IR_instruction = re.sub(r'dense<\[\[.*?\]\]>', 'dense<neglected>', IR_instruction)
                pphlo_log += re.sub(r'dense<".*?">', 'dense<neglected>', IR_instruction)
            pphlo_record_dict = {"baseline": {"pphlo": hash(pphlo_log.strip()), "deduplication": ["baseline"]}}
            with open(os.path.join(IR_record_folder_path, f'pphlo_{function_name}.json'), 'w') as file:
                json.dump(pphlo_record_dict, file)

def run_test_single(test_command, output_passoption_folder_path, pass_option="baseline", sml_copy="sml"):
    if not os.path.exists(output_passoption_folder_path):
        os.makedirs(output_passoption_folder_path)
    for repeat_i in range(repeat_time):
        output_file = os.path.join(output_passoption_folder_path, "test_" + test_command.split("/")[-1] + f"_{repeat_i}.txt")
        print(f"Running passoptions {pass_option} for {output_file} with repeat {repeat_i} in {sml_copy}")
        if test_mod == mod.sml:
            with open(output_file, "w") as file:
                process = subprocess.Popen(test_command, stdout=file, stderr=subprocess.STDOUT)
                process.communicate()
        else:
            with open(output_file, "w") as file:
                nodectl_process = subprocess.Popen("bazel-bin/examples/python/utils/nodectl up".split(" "), stdin=subprocess.PIPE, stdout=file, stderr=subprocess.STDOUT, text=True)
                time.sleep(3)
                process = subprocess.run(test_command.split(" "), stderr=subprocess.STDOUT)
                nodectl_process.communicate("down")

def run_with_queues(sml_copy, config_queue):
    for pass_option, test_file, pass_baseline, (test_command, output_folder_path, not_skip_list) in config_queue:
        output_passoption_folder_path = os.path.join(output_folder_path, pass_option)
        test_command = test_command.replace("sml", sml_copy)
        if pass_option == "baseline":
            # Run the baseline test without pass options
            with open(os.path.join(original_work_dir, "file-to-be-modified", test_file), 'r') as src_file:
                test_codes_base = src_file.readlines()
            modified_code = []
            for line in test_codes_base:
                # specify the partynum and protocal
                line = line.replace('{partynum}', str(partynum)).replace('{protocalchosen}', protocal)
                # do not execute the pphlo test
                line = line.replace('{pphlo_dict}', "(None, None)")
                # only test with functions specified
                line = line.replace('{skip_control}', f", not_skip_list={not_skip_list}")
                # change the import of sml to the sml_copy
                if sml_copy != "sml":
                    line = line.replace('import sml.', f'import {sml_copy}.').replace('from sml.', f'from {sml_copy}.')
                modified_code.append(line)
                if 'copts = spu_pb2.CompilerOptions()' in line:
                    modified_code.append(line)
                    indent = len(line) - len(line.lstrip())
                    modified_code.append(indent * " " + f'copts.{pass_baseline} = True\n')
            with open(test_file.replace("sml", sml_copy), 'w') as dst_file:
                dst_file.writelines(modified_code)
            run_test_single(test_command, output_passoption_folder_path, pass_option=pass_option, sml_copy=sml_copy)
            split_and_extract(output_passoption_folder_path, partynum)
            IR_record(output_folder_path, pass_option)
        else:
            IR_record_folder_path = os.path.join(output_folder_path, "IR_record")
            with open(os.path.join(original_work_dir, "file-to-be-modified", test_file), 'r') as src_file:
                test_codes_base = src_file.readlines()
            modified_code = []
            for line in test_codes_base:
                # specify the partynum and protocal
                line = line.replace('{partynum}', str(partynum)).replace('{protocalchosen}', protocal).replace('{protocalchosen}', protocal)
                # specify the PPHLO to be compared
                line = line.replace('{pphlo_dict}', f'("{IR_record_folder_path}", "{pass_option}")')
                # only test with functions specified
                line = line.replace('{skip_control}', f", not_skip_list={not_skip_list}")
                # change the import of sml to the sml_copy
                if sml_copy != "sml":
                    line = line.replace('import sml.', f'import {sml_copy}.').replace('from sml.', f'from {sml_copy}.')
                modified_code.append(line)
                if 'copts = spu_pb2.CompilerOptions()' in line:
                    indent = len(line) - len(line.lstrip())
                    modified_code.append(indent * " " + f'copts.{pass_baseline} = True\n')
                    modified_code.append(indent * " " + f'copts.{pass_option} = True\n')

            with open(test_file.replace("sml", sml_copy), 'w') as dst_file:
                dst_file.writelines(modified_code)
            run_test_single(test_command, output_passoption_folder_path, pass_option=pass_option, sml_copy=sml_copy)

if __name__ == "__main__":
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

    """get all possible pass options"""
    with open(passoptions_path, 'r') as file:
        pass_options_list = [line.strip() for line in file if line.strip()]
        # disable_hlocse_index = pass_options_list.index("disable_hlocse")
        # pass_options_list = pass_options_list[disable_hlocse_index + 1:]
        # pass_options_list = [pass_option for pass_option in pass_options_list if pass_option.split("_")[-1] == "0"]

    """Change the working directory"""
    original_work_dir = os.getcwd()
    work_dir = ".."
    os.chdir(work_dir)

    # create the output folder
    output_folder_path = os.path.join(original_work_dir, output_folder)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    """print the output to the log file"""
    sys.stdout = open(os.path.join(output_folder_path, 'test-log2.txt'), 'w')

    # copies of sml for mutlithreading
    sml_copies = ["sml"] + [f"sml{i}" for i in range(1, thread_num)]

    """Specify the test cases here!!!"""
    if test_mod == mod.sml:
        test_file_dir = "sml"
        test_file_list = glob.glob(os.path.join(test_file_dir, "*", "*", "*_test*"))
        # test_file_list = ["sml/ensemble/tests/adaboost_test.py"]
        # test_file_list = ["sml/preprocessing/tests/preprocessing_test.py"]
        # test_file_delet_list = ["sml/linear_model/tests/quantile_test.py",
        #                   "sml/ensemble/tests/forest_test.py",
        #                   "sml/feature_selection/tests/chi2_test.py",
        #                   "sml/ensemble/tests/adaboost_test.py",
        #                   "sml/preprocessing/tests/preprocessing_test.py"]
        # test_file_delet_list = ["sml/metrics/classification/classification_test.py"]
        # test_file_list = [file for file in test_file_list if file in test_file_delet_list]
        # test_file_list = ["sml/cluster/tests/kmeans_test.py"]
        # test_file_list = ["sml/ensemble/tests/adaboost_test.py"]
        # test_file_list = ["sml/metrics/regression/regression_test.py"]
        # test_file_list = ["sml/naive_bayes/tests/gnb_test.py", "sml/ensemble/tests/forest_test.py"]
    else:
        test_file_list = ["examples/python/ml/ss_lr/ss_lr.py"]

    """define the command to be runned"""
    """Create the log folder for each test file"""
    test_file_dict = {}
    for test_file in test_file_list:
        file_inf = test_file.split("/")
        test_file_name = file_inf[-1].split('.')[0]
        if test_mod == mod.sml:
            test_file_command = "/".join(["bazel-bin", "sml", file_inf[-3], f"{file_inf[-2]}", f"{test_file_name}"])
        else:
            test_file_command = "/".join(["bazel-bin", "examples", file_inf[-4], file_inf[-3], file_inf[-2], f"{test_file_name}"])
        pass_ref_dict = {}
        # for IR_dict_path in glob.glob(os.path.join(original_work_dir, pass_ref_folder, test_file_name, "IR_record", f"pphlo_*.json")):
        #     with open(IR_dict_path, 'r') as file:
        #         IR_dict = json.load(file)
        #     for pass_option in IR_dict.keys():
        #         if pass_option != "baseline" and pass_option not in pass_ref_list:
        #             pass_ref_list.append(pass_option)
        with open(os.path.join(os.path.join(original_work_dir, pass_ref_folder, test_file_name, "extract_result.json")), "r") as file:
            extract_result = json.load(file)
        for function_name in extract_result.keys():
            for pass_option in extract_result[function_name].keys():
                if pass_option != "baseline":
                    if pass_option not in pass_ref_dict:
                        pass_ref_dict[pass_option] = [function_name]
                    else:
                        pass_ref_dict[pass_option].append(function_name)
        test_file_dict[test_file] = {}
        for pass_baseline in pass_ref_dict.keys():
            test_file_log_path = os.path.join(output_folder_path, "|".join([test_file_name, pass_baseline]))
            if not os.path.exists(test_file_log_path):
                os.makedirs(test_file_log_path)
            test_file_dict[test_file][pass_baseline] = (test_file_command, test_file_log_path, pass_ref_dict[pass_baseline])
    # Baseline is run seperately to for the IR deduplication
    # If the baseline join the mutlithread with other passes, it is possible the test with passoption runs before the baseline
    # Queues for threads of baseline tests running in different sml copies
    queues = {path: [] for path in sml_copies}
    i = 0
    for test_file in test_file_dict.keys():
        for pass_baseline in test_file_dict[test_file].keys():
            queues[sml_copies[i % len(sml_copies)]].append(("baseline", test_file, pass_baseline, test_file_dict[test_file][pass_baseline]))
            i += 1
    
    # Run the baseline test in multithreading
    threads = []
    for sml_copy in sml_copies:
        t = threading.Thread(target=run_with_queues, args=(sml_copy, queues[sml_copy]), name=f"Worker-{sml_copy}")
        t.start()
        threads.append(t)
    # Wait for all threads to complete
    for t in threads:
        t.join()
    print("Baseline tests finished")

    # Queues for threads running in different sml copies
    queues = {path: [] for path in sml_copies}
    i = 0
    for pass_option in pass_options_list:
        for test_file in test_file_dict.keys():
            for pass_baseline in test_file_dict[test_file].keys():
                queues[sml_copies[i % len(sml_copies)]].append((pass_option, test_file, pass_baseline, test_file_dict[test_file][pass_baseline]))
                i += 1
    
    # Run the test in multithreading
    threads = []
    for sml_copy in sml_copies:
        t = threading.Thread(target=run_with_queues, args=(sml_copy, queues[sml_copy]), name=f"Worker-{sml_copy}")
        t.start()
        threads.append(t)
