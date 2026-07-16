import re
import glob
import os
import json

def hash_test():
    log_directory_nohash = "test-log-new/new-auto-test"
    log_directory_hash = "test-log-new/new-auto-test-hash"
    for log_sub_directory in glob.glob(os.path.join(log_directory_nohash, "*/")):
        with open(os.path.join(log_sub_directory, "extract_result.json"), "r") as file:
            extract_result_nohash = json.load(file)
        with open(os.path.join(log_sub_directory.replace(log_directory_nohash, log_directory_hash), "extract_result.json"), "r") as file:
            extract_result_hash = json.load(file)
        for function_name in extract_result_nohash.keys():
            for pass_to_extract in extract_result_nohash[function_name].keys():
                if pass_to_extract not in extract_result_hash[function_name].keys():
                    print(f"Function {function_name} with pass {pass_to_extract} nohash test does not have corresponding log file in hash test")
            for pass_to_extract in extract_result_hash[function_name].keys():
                if pass_to_extract not in extract_result_nohash[function_name].keys():
                    print(f"Function {function_name} with pass {pass_to_extract} hash test does not have corresponding log file in nohash test")

def deduplication_test():
    log_directory_hash = "test-log-new/new-auto-test-hash"
    for log_sub_directory in glob.glob(os.path.join(log_directory_hash, "*/")):
        testcase_name = log_sub_directory.split("/")[-2]
        if testcase_name == "preprocessing_test":
            old_test_path = "test-log/preprocessing-add"
        elif testcase_name in ["adaboost_test", "chi2_test", "quantile_test", "forest_test"]:
            old_test_path = "test-log/sml-add"
        else:
            old_test_path = "test-log/sml-more-add"
        for IR_record in glob.glob(os.path.join(log_sub_directory, "IR_record", "pphlo_*.json")):
            function_name = "_".join(os.path.splitext(os.path.basename(IR_record))[0].split("_")[1:])
            with open(IR_record, "r") as file:
                IR_dict = json.load(file)
            IR_dict_oripipeline = {}
            for pass_test in glob.glob(os.path.join(old_test_path, "*/")):
                pass_test_name = pass_test.split("/")[-2]
                if pass_test_name != "baseline":
                    with open(os.path.join(pass_test, "pass_closed.txt"), "r") as file:
                        pass_test_name = file.read().strip()
                with open(os.path.join(pass_test, "extract_result.json"), "r") as file:
                    extract_result = json.load(file)
                if function_name in extract_result.keys():
                    if "pphlo" in extract_result[function_name].keys():
                        pphlo_log = ""
                        for IR_instruction in extract_result[function_name]["pphlo"]:
                            IR_instruction = re.sub(r'dense<".*?">', 'dense<neglected>', IR_instruction)
                            IR_instruction = re.sub(r'dense<\[\[.*?\]\]>', 'dense<neglected>', IR_instruction)
                            pphlo_log += re.sub(r'dense<".*?">', 'dense<neglected>', IR_instruction)
                        pphlo_cur = hash(pphlo_log.strip())
                        if pphlo_cur not in IR_dict_oripipeline.values():
                            IR_dict_oripipeline[pass_test_name] = pphlo_cur
            with open(os.path.join("tmp-exp/deduplication_test", f"pphlo_{function_name}.json"), "w") as file:
                json.dump(IR_dict_oripipeline, file)

# load the result and compare
def deduplication_test_step2():
    log_directory_hash = "test-log-new/new-auto-test-all"
    for log_sub_directory in glob.glob(os.path.join(log_directory_hash, "*/")):
        testcase_name = log_sub_directory.split("/")[-2]
        with open(os.path.join(log_sub_directory, "extract_result.json"), "r") as file:
            extract_result_hash = json.load(file)
        for function_name in extract_result_hash.keys():
            with open(os.path.join("tmp-exp/deduplication_test", f"pphlo_{function_name}.json"), "r") as file:
                IR_dict_oripipeline = json.load(file)
            for pass_test in IR_dict_oripipeline.keys():
                if pass_test not in extract_result_hash[function_name].keys():
                    print(f"Function {function_name} with pass {pass_test} does not have corresponding deduplicated result in deduplicated test")
            for pass_test in extract_result_hash[function_name].keys():
                if pass_test not in IR_dict_oripipeline.keys():
                    print(f"Function {function_name} with pass {pass_test} does not have corresponding deduplicated result in original test")

# check the result among different protocals
def compare_protocals():
    log_directory_ABY3 = "test-log-new/new-auto-test-all"
    log_directory_cheetah = "test-log-new/new-auto-test-all-CHEETAH"
    log_directory_SEMI2K = "test-log-new/new-auto-test-all-SEMI2K"
    for log_sub_directory in glob.glob(os.path.join(log_directory_ABY3, "*/")):
        testcase_name = log_sub_directory.split("/")[-2]
        with open(os.path.join(log_sub_directory, "extract_result.json"), "r") as file:
            extract_result_ABY3 = json.load(file)
        with open(os.path.join(log_sub_directory, "extract_result.json").replace(log_directory_ABY3, log_directory_cheetah), "r") as file:
            extract_result_cheetah = json.load(file)
        with open(os.path.join(log_sub_directory, "extract_result.json").replace(log_directory_ABY3, log_directory_SEMI2K), "r") as file:
            extract_result_SEMI2K = json.load(file)
        # compare between ABY3 and CHEETAH
        for function_name in extract_result_ABY3.keys():
            for pass_test in extract_result_ABY3[function_name].keys():
                if pass_test not in extract_result_cheetah[function_name].keys():
                    print(f"Function {function_name} with pass {pass_test} in ABY3 does not have corresponding deduplicated result in CHEETAH")
            for pass_test in extract_result_cheetah[function_name].keys():
                if pass_test not in extract_result_ABY3[function_name].keys():
                    print(f"Function {function_name} with pass {pass_test} in CHEETAH does not have corresponding deduplicated result in ABY3")
        # compare between ABY3 and SEMI2K
        for function_name in extract_result_ABY3.keys():
            for pass_test in extract_result_ABY3[function_name].keys():
                if pass_test not in extract_result_SEMI2K[function_name].keys():
                    print(f"Function {function_name} with pass {pass_test} in ABY3 does not have corresponding deduplicated result in SEMI2K")
            for pass_test in extract_result_SEMI2K[function_name].keys():
                if pass_test not in extract_result_ABY3[function_name].keys():
                    print(f"Function {function_name} with pass {pass_test} in SEMI2K does not have corresponding deduplicated result in ABY3")

# sort results among different functions, get the function with the most number of passes with muatated IR
def sort_functions():
    log_directory_ABY3 = "test-log-new/new-auto-test-all"
    # log_directory_ABY3 = "test-log-new/new-auto-test-all-CHEETAH"
    function_dict = {}
    for log_sub_directory in glob.glob(os.path.join(log_directory_ABY3, "*/")):
        with open(os.path.join(log_sub_directory, "extract_result.json"), "r") as file:
            extract_result_ABY3 = json.load(file)
        for function_name in extract_result_ABY3.keys():
            function_dict[function_name] = len(extract_result_ABY3[function_name].keys())
    function_dict = dict(sorted(function_dict.items(), key=lambda item: int(item[1]), reverse=True))
    print(len(function_dict))

# get all the pass options with muatated IR
def get_all_pass_mutated():
    """get all the pass options with muatated IR that can be run in runtime"""
    # log_directory_ABY3 = "test-log-new/new-auto-test-all"
    log_directory_ABY3 = "test-log-new/new-auto-test-all-second"
    pass_list = []
    for log_sub_directory in glob.glob(os.path.join(log_directory_ABY3, "*/")):
        with open(os.path.join(log_sub_directory, "extract_result.json"), "r") as file:
            extract_result_ABY3 = json.load(file)
        for function_name in extract_result_ABY3.keys():
            for pass_test in extract_result_ABY3[function_name].keys():
                if pass_test not in pass_list:
                    pass_list.append(pass_test)
    print(sorted(pass_list))

    """get all the pass options with muatated IR without checking runnable in runtime"""
    pass_list_nocheckruntime = []
    for log_sub_directory in glob.glob(os.path.join(log_directory_ABY3, "*/")):
        testcase_name = log_sub_directory.split("/")[-2]
        # compare between ABY3 and CHEETAH
        for IR_record in glob.glob(os.path.join(log_sub_directory, "IR_record", "pphlo_*.json")):
            function_name = "_".join(os.path.splitext(os.path.basename(IR_record))[0].split("_")[1:])
            with open(IR_record, "r") as file:
                IR_dict = json.load(file)
            for pass_test in IR_dict.keys():
                # if pass_test == "disable_mlir_mhlo_createlegalizegeneraldotpass":
                #     print(function_name)
                if pass_test not in pass_list_nocheckruntime:
                    pass_list_nocheckruntime.append(pass_test)
    print(sorted(pass_list_nocheckruntime))
        
if __name__ == "__main__":
    # hash_test()
    # deduplication_test()
    # deduplication_test_step2()
    # compare_protocals()
    sort_functions()
    # get_all_pass_mutated()