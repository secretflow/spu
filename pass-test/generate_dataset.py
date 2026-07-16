import re
import glob
import os
import json

from extract_inf import split_log_party, extract_log
from analysis_inf import communication_costs_analysis

import copy

"""for all pass test for one test case"""
def split_and_extract_testcase(log_sub_directory, party_nums):
    pass_to_extract_list = []
    for IR_record in glob.glob(os.path.join(log_sub_directory, 'IR_record', 'pphlo_*.json')):
        with open(IR_record, 'r') as file:
            IR_record = json.load(file)
            pass_to_extract_list = pass_to_extract_list + list(IR_record.keys())
    pass_to_extract_list = list(set(pass_to_extract_list))
    for pass_to_extract in pass_to_extract_list:
        if pass_to_extract == "baseline":
            pass_to_extract_path = os.path.join(log_sub_directory, pass_to_extract)
            """Split the raw log file with logs from different parteis into different log files"""
            split_log_party(pass_to_extract_path, party_nums, hlo_log=True)

            """extract information from splitted log files and save the information to json file"""
            extract_log(pass_to_extract_path, hlo_log=True)
        else:
            pass_to_extract_path = os.path.join(log_sub_directory, pass_to_extract)
            """Split the raw log file with logs from different parteis into different log files"""
            split_log_party(pass_to_extract_path, party_nums)

            """extract information from splitted log files and save the information to json file"""
            extract_log(pass_to_extract_path)
        
    
    test_result_dict = {}
    for IR_record in glob.glob(os.path.join(log_sub_directory, 'IR_record', 'pphlo_*.json')):
        function_name = "_".join(os.path.splitext(os.path.basename(IR_record))[0].split("_")[1:])
        test_result_dict_func = {}
        with open(IR_record, 'r') as file:
            IR_record = json.load(file)
        for pass_to_extract in IR_record.keys():
            pass_to_extract_path = os.path.join(log_sub_directory, pass_to_extract)
            with open(os.path.join(pass_to_extract_path, 'extract_result.json'), 'r') as file:
                extract_file_content = json.load(file)
                if function_name in extract_file_content.keys():
                    test_result_dict_func[pass_to_extract] = {"profile":extract_file_content[function_name]["profile"], "deduplication": IR_record[pass_to_extract]["deduplication"]}
                else:
                    print(f"Function {function_name} in {pass_to_extract_path} does not have log file")
        test_result_dict[function_name] = test_result_dict_func
    with open(os.path.join(log_sub_directory, 'extract_result.json'), 'w') as file:
        json.dump(test_result_dict, file)

if __name__ == "__main__":
    """Define the parameters"""
    # define the number of parties which is used to split the log files
    # Currently, even if the number of parties exceeds the correct number, the code can still work
    party_nums = 2
    # define the path of the log files
    log_directory = 'test-log-model/SEMI2K'
    for log_sub_directory in glob.glob(os.path.join(log_directory, '*/')):
        split_and_extract_testcase(log_sub_directory, party_nums)
    actions_collect, bytes_collect = communication_costs_analysis(log_directory)
    func_dict = {}
    for log_sub_directory in glob.glob(os.path.join(log_directory, '*/')):
        with open(os.path.join(log_sub_directory, 'extract_result.json'), 'r') as file:
            extract_result = json.load(file)
        for func_name in extract_result.keys():
            func_dict[func_name] = (log_sub_directory, False)

    algebra_dict_actions = copy.deepcopy(func_dict)
    for case in actions_collect:
        ratio = case[1][2]
        if ratio < 1:
            func_name = case[0].split("-")[1]
            if func_name in algebra_dict_actions.keys():
                algebra_dict_actions[func_name] = (algebra_dict_actions[func_name][0], True)
            else:
                print(f"Function name {func_name} not found in algebra_dict_actions")
    
    with open('dataset/algebra_dict_actions.json', 'w') as file:
        json.dump(algebra_dict_actions, file)
    
    algebra_dict_bytes = copy.deepcopy(func_dict)
    for case in bytes_collect:
        ratio = case[1][2]
        if ratio < 1:
            func_name = case[0].split("-")[1]
            if func_name in algebra_dict_bytes.keys():
                algebra_dict_bytes[func_name] = (algebra_dict_bytes[func_name][0], True)
            else:
                print(f"Function name {func_name} not found in algebra_dict_bytes")

    with open('dataset/algebra_dict_bytes.json', 'w') as file:
        json.dump(algebra_dict_bytes, file)
    