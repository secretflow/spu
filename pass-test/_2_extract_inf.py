import re
import glob
import os
import json

"""Split the raw log file with logs from different parteis into different log files"""
def split_log_party(log_sub_directory, party_nums, pphlo_log = True, hlo_log = False):
    extracted_inf_directory = os.path.join(log_sub_directory, "extracted_inf")
    if not os.path.exists(extracted_inf_directory):
        os.makedirs(extracted_inf_directory)
    log_file_raw_list = glob.glob(os.path.join(log_sub_directory, '*.txt'))
    
    for log_file in log_file_raw_list:
        repeat_i = log_file.split("_")[-1].split(".")[0]
        # read information from log file
        with open(log_file, 'r', encoding='utf-8') as file:
            log_content = file.readlines()

        party_logs = {}
        for party_i in range(party_nums):
            party_logs[f'Party_{party_i}'] = []
        
        if pphlo_log:
            pphlo_logs = []
            pphlo_flag = False

        if hlo_log:
            hlo_logs = []
        for line in log_content:
            # delete the . occurs at the beginning of line
            # these . are generated from pytest
            if line.startswith("."):
                while line.startswith("."):
                    line = line[1:]
            if pphlo_log and pphlo_flag:
                pphlo_logs.append(line)
                if line.startswith("}"):
                    pphlo_flag = False
            else:
                if 'Party_' in line:
                    current_party = re.search(r'Party_\d+', line).group(0)
                    party_logs[current_party].append(line.replace(current_party + "|", ''))
                # Only extract the pphlo logs from the first repeat
                if pphlo_log and repeat_i == "0":
                    if line.startswith("module"):
                        pphlo_flag = True
                        pphlo_logs.append(line)
                if hlo_log:
                    if "HLO_IR" in line:
                        hlo_logs.append(line.replace("HLO_IR|", ''))


        """Save the splitted log files"""
        for party, logs in party_logs.items():
            with open(os.path.join(log_sub_directory, "extracted_inf", f'{party.split("_")[-1]}_party_{repeat_i}_repeat.txt'), 'w', encoding='utf-8') as file:
                file.writelines(logs)
        # Only save the pphlo logs from the first repeat
        if pphlo_log and repeat_i == "0":
            with open(os.path.join(log_sub_directory, "extracted_inf", 'pphlo.txt'), 'w', encoding='utf-8') as file:
                file.writelines(pphlo_logs)
        if hlo_log:
            # Only save the hlo logs from the first repeat
            if repeat_i == "0":
                with open(os.path.join(log_sub_directory, "extracted_inf", 'hlo.txt'), 'w', encoding='utf-8') as file:
                    file.writelines(hlo_logs)

"""extract information from splitted log files and save the information to json file"""
def extract_log(log_sub_directory, pphlo_log = True, hlo_log = False, operator_profile = False):
    extracted_inf_directory = os.path.join(log_sub_directory, "extracted_inf")
    if pphlo_log:
        """extract information from pphlo log files"""
        pphlo_log_file = os.path.join(extracted_inf_directory, 'pphlo.txt')
        unittest_pphlo_dict = {}
    if hlo_log:
        """extract information from hlo log files"""
        hlo_log_file = os.path.join(extracted_inf_directory, 'hlo.txt')
        unittest_hlo_dict = {}
    
    if pphlo_log:
        unittest_pphlo_flag = False
        unittest_pphlo_fragment = []
        with open(pphlo_log_file, 'r', encoding='utf-8') as file:
            log_content = file.readlines()
        for line in log_content:
            if unittest_pphlo_flag == True:
                unittest_pphlo_fragment.append(line)
            if line.startswith("module"):
                unittest_pphlo_flag = True
                unittest_pphlo_fragment = [line]
            if line.startswith("}"):
                unittest_pphlo_flag = False
                try:
                    function_name = re.search(r"module\s+@jit_(.*?)\s+attributes", unittest_pphlo_fragment[0]).group(1)
                except:
                    function_name = re.search(r"module\s+@\"jit_(.*?)\"\s+attributes", unittest_pphlo_fragment[0]).group(1)
                    function_name = function_name.replace("<", "_").replace(">", "_").replace("(", "_").replace(")", "_").replace("'", "_").replace(",", "_")
                unittest_pphlo_dict[function_name] = unittest_pphlo_fragment
    
    if hlo_log:
        unittest_hlo_flag = False
        unittest_hlo_fragment = []
        with open(hlo_log_file, 'r', encoding='utf-8') as file:
            log_content = file.readlines()
        for line in log_content:
            if unittest_hlo_flag == True:
                unittest_hlo_fragment.append(line)
            if line.startswith("Start printing HLO IR for"):
                unittest_hlo_flag = True
                function_name = line.strip().split("for ")[-1]
                unittest_hlo_fragment = []
            if line.startswith("End of printing HLO IR for"):
                unittest_hlo_flag = False
                function_name = function_name.replace("<", "_").replace(">", "_").replace("(", "_").replace(")", "_").replace("'", "_").replace(",", "_").replace("[", "_").replace("]", "_").replace(" ", "_")
                unittest_hlo_dict[function_name] = unittest_hlo_fragment[:-1]

    """extract information from splitted log files"""
    test_result_dict = {}
    log_file_list = glob.glob(os.path.join(extracted_inf_directory, '*party*.txt'))
    # sort the log files by the repeat index first and then the number of parties
    log_file_list.sort(key=lambda x: (int(re.search(r'(\d+)_repeat', x).group(1)), int(re.search(r'(\d+)_party', x).group(1))))
    for log_file in log_file_list:
        parts = os.path.splitext(os.path.basename(log_file))[0].split('_')
        party_num, repeat_num = parts[0], parts[-2]
        unittest_list = []
        unittest_flag = False
        unittest_fragment = []

        with open(log_file, 'r', encoding='utf-8') as file:
            log_content = file.readlines()
        for line in log_content:
            if unittest_flag == True:
                unittest_fragment.append(line)
            if line.find("[Profiling]") != -1:
                unittest_flag = True
                unittest_fragment = [line]
            if line.find("Link details") != -1:
                unittest_flag = False
                unittest_list.append(unittest_fragment)

        # test_result_all_repeat = {}
        for function_count, unittest_fragment in enumerate(unittest_list):
            test_result = {}
            # function_name = test_name + f"_{function_count}"
            function_name = re.search(r"\bexecution\b\s+(.*?)\s+\bcompleted\b", unittest_fragment[0]).group(1)
            total_time_match = re.search(r"total time (\d+\.\d+s)", unittest_fragment[0]).group(1)
            link_details_match = re.search(
                r"Link details: total send bytes (\d+), recv bytes (\d+), send actions (\d+), recv actions (\d+)",
                unittest_fragment[-1]
            )
            test_result['total_time'] = total_time_match
            test_result['send_bytes'] = link_details_match.group(1)
            test_result['recv_bytes'] = link_details_match.group(2)
            test_result['send_actions'] = link_details_match.group(3)
            test_result['recv_actions'] = link_details_match.group(4)
            if operator_profile:
                operator_profile_result = {}
                HLO_profile_flag = False
                HAL_profile_flag = False
                MPC_profile_flag = False
                operator_details_pattern = r"([^,]+),\s*executed\s*(\d+)\s*times,\s*duration\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)s,\s*send bytes\s*(\d+)\s*recv bytes\s*(\d+),\s*send actions\s*(\d+),\s*recv actions\s*(\d+)"
                for line in unittest_fragment:
                    if line.find("HLO profiling") != -1:
                        HLO_profile_flag = True
                        HLO_profile_dict = {}
                        continue
                    if line.find("HAL profiling") != -1:
                        HLO_profile_flag = False
                        operator_profile_result["HLO_profile"] = HLO_profile_dict
                        HAL_profile_flag = True
                        HAL_profile_dict = {}
                        continue
                    if line.find("MPC profiling") != -1:
                        HAL_profile_flag = False
                        operator_profile_result["HAL_profile"] = HAL_profile_dict
                        MPC_profile_flag = True
                        MPC_profile_dict = {}
                        continue
                    if line.find("Link details") != -1:
                        operator_profile_result["MPC_profile"] = MPC_profile_dict
                        break
                    if HLO_profile_flag or HAL_profile_flag or MPC_profile_flag:
                        match = re.findall(operator_details_pattern, line.split(" - ")[1])
                        one_operator_dict = {}
                        one_operator_dict["executed_times"] = match[0][1]
                        one_operator_dict["total_time"] = match[0][2]
                        one_operator_dict["send_bytes"] = match[0][3]
                        one_operator_dict["recv_bytes"] = match[0][4]
                        one_operator_dict["send_actions"] = match[0][5]
                        one_operator_dict["recv_actions"] = match[0][6]
                        if HLO_profile_flag:
                            HLO_profile_dict[match[0][0]] = one_operator_dict
                        elif HAL_profile_flag:
                            HAL_profile_dict[match[0][0]] = one_operator_dict
                        elif MPC_profile_flag:
                            MPC_profile_dict[match[0][0]] = one_operator_dict
                        else:
                            print("Error: operator profile flag error")
                            assert False
                test_result["operator_profile"] = operator_profile_result                     
            if repeat_num == "0" and party_num == "0":
                if function_name in test_result_dict.keys():
                    print(f"Function {function_name} in {extracted_inf_directory} has multiple log files")
                test_result_dict[function_name] = {"profile": {f"repeat_{repeat_num}" : {f"party_{party_num}": test_result}}}
            elif party_num == "0":
                test_result_dict[function_name]["profile"][f"repeat_{repeat_num}"] = {f"party_{party_num}": test_result}
            else:
                test_result_dict[function_name]["profile"][f"repeat_{repeat_num}"][f"party_{party_num}"] = test_result

    if pphlo_log or hlo_log:
        """match the information from pphlo (or hlo) log files and splitted log files"""
        for function_name in test_result_dict.keys():
            function_name_pphlo = function_name.replace("<", "_").replace(">", "_").replace("(", "_").replace(")", "_").replace("'", "_").replace(",", "_").replace("[", "_").replace("]", "_").replace(" ", "_")
            if pphlo_log:
                if function_name_pphlo in unittest_pphlo_dict.keys():
                    test_result_dict[function_name]['pphlo'] = unittest_pphlo_dict[function_name_pphlo]
                else:
                    print(f"Function {function_name} in {extracted_inf_directory} does not have pphlo log file")
            
            if hlo_log:
                if function_name_pphlo in unittest_hlo_dict.keys():
                    test_result_dict[function_name]['hlo'] = unittest_hlo_dict[function_name_pphlo]
                else:
                    print(f"Function {function_name} in {extracted_inf_directory} does not have hlo log file")

    # save information to json file
    with open(os.path.join(log_sub_directory, 'extract_result.json'), 'w') as file:
        json.dump(test_result_dict, file)

"""Only for one pass test for one test case"""
def split_and_extract(log_sub_directory, party_nums):
    """Split the raw log file with logs from different parteis into different log files"""
    split_log_party(log_sub_directory, party_nums)

    """extract information from splitted log files and save the information to json file"""
    extract_log(log_sub_directory)

"""for all pass test for one test case"""
def split_and_extract_testcase(log_sub_directory, party_nums):
    pass_to_extract_list = []
    for IR_record in glob.glob(os.path.join(log_sub_directory, 'IR_record', 'pphlo_*.json')):
        with open(IR_record, 'r') as file:
            IR_record = json.load(file)
            pass_to_extract_list = pass_to_extract_list + list(IR_record.keys())
    pass_to_extract_list = list(set(pass_to_extract_list))
    for pass_to_extract in pass_to_extract_list:
        if pass_to_extract != "baseline":
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
    log_directory = 'test-log/SEMI2K-all'
    for log_sub_directory in glob.glob(os.path.join(log_directory, '*/')):
        # if log_sub_directory.find("preprocessing_test") != -1:
        #     print("preprocessing_test is skipped!!!!!!!!!!!!!!!!!!!!")
        #     continue
        split_and_extract_testcase(log_sub_directory, party_nums)