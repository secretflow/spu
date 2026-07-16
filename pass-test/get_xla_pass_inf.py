import os
import glob
import json
import re

XLA_pass_directory = "../xla/xla/service"

# Get all the XLA pass in the XLA pass directory
# Currently only get the pass do not need arguements
def get_xla_pass():
    pass_record = {}
    pass_record["HloModulePass"] = {}
    pass_record["HloModulePass_consturctor"] = {}
    pass_record["OpExpanderPass"] = {}
    pass_record["OpExpanderPass_consturctor"] = {}
    pass_class = []
    for file_path in glob.glob(os.path.join(XLA_pass_directory, "*.h")):
        file_name = os.path.basename(file_path)
        # Skip the interface file
        if file_name == "hlo_pass_interface.h" or file_name == "op_expander_pass.h":
            continue
        with open(file_path, 'r') as f:
            for line in f:
                if "class" in line and ": public " in line and "{" in line:
                    try:
                        pass_name = re.search(r': public (\w+) {', line).group(1)
                        pass_class.append(pass_name)     
                    except:
                        print(f"{line} in {file_name} do not include pass class name")
                if "HloModulePass" in line:
                    try:
                        pass_name = re.search(r'class (\w+) : public', line).group(1)
                    except:
                        print(f"{line} in {file_name} do not include pass name")
                        continue
                    # Check if the pass is has consturctor
                    if pass_name + "(" in f.read():
                        pass_record["HloModulePass_consturctor"][pass_name] = file_name
                    else:
                        pass_record["HloModulePass"][pass_name] = file_name
                if "OpExpanderPass" in line:
                    try:
                        pass_name = re.search(r'class (\w+) : public', line).group(1)
                    except:
                        print(f"{line} in {file_name} do not include pass name")
                        continue
                    # Check if the pass is has consturctor
                    if pass_name + "(" in f.read():
                        pass_record["OpExpanderPass_consturctor"][pass_name] = file_name
                    else:
                        pass_record["OpExpanderPass"][pass_name] = file_name

    # print(list(set(pass_class)))
    print(len(pass_record["HloModulePass"]))
    print(len(pass_record["HloModulePass_consturctor"]))
    print(len(pass_record["OpExpanderPass"]))
    print(len(pass_record["OpExpanderPass_consturctor"]))
    with open("extract_XLAPass.json", 'w') as file:
        json.dump(pass_record, file)

    # """ CPU passes """
    # print("Start to get CPU passes!!!")
    # pass_record_cpu = {}
    # pass_record_cpu["HloModulePass"] = {}
    # pass_record_cpu["HloModulePass_consturctor"] = {}
    # pass_record_cpu["OpExpanderPass"] = {}
    # pass_record_cpu["OpExpanderPass_consturctor"] = {}
    # pass_class = []
    # for file_path in glob.glob(os.path.join(XLA_pass_directory, "cpu", "*.h")):
    #     file_name = os.path.basename(file_path)
    #     with open(file_path, 'r') as f:
    #         for line in f:
    #             if "class" in line and ": public " in line and "{" in line:
    #                 try:
    #                     pass_name = re.search(r': public (\w+) {', line).group(1)
    #                     pass_class.append(pass_name)
    #                     if pass_name == 'Pass':
    #                         print(f"{line} in {file_name} is a pass!!!!!!!!!!!!!!!!!!")       
    #                 except:
    #                     print(f"{line} in {file_name} do not include pass class name")
    #             if "HloModulePass" in line:
    #                 try:
    #                     pass_name = re.search(r'class (\w+) : public', line).group(1)
    #                 except:
    #                     print(f"{line} in {file_name} do not include pass name")
    #                     continue
    #                 # Check if the pass is has consturctor
    #                 if pass_name + "(" in f.read():
    #                     pass_record_cpu["HloModulePass_consturctor"][pass_name] = file_name
    #                 else:
    #                     pass_record_cpu["HloModulePass"][pass_name] = file_name
    #             if "OpExpanderPass" in line:
    #                 try:
    #                     pass_name = re.search(r'class (\w+) : public', line).group(1)
    #                 except:
    #                     print(f"{line} in {file_name} do not include pass name")
    #                     continue
    #                 # Check if the pass is has consturctor
    #                 if pass_name + "(" in f.read():
    #                     pass_record_cpu["OpExpanderPass_consturctor"][pass_name] = file_name
    #                 else:
    #                     pass_record_cpu["OpExpanderPass"][pass_name] = file_name
    # # print(list(set(pass_class)))
    # print(len(pass_record_cpu["HloModulePass"]))
    # print(len(pass_record_cpu["HloModulePass_consturctor"]))
    # print(len(pass_record_cpu["OpExpanderPass"]))
    # print(len(pass_record_cpu["OpExpanderPass_consturctor"]))

    # """ GPU passes """
    # print("Start to get GPU passes!!!")
    # pass_record_gpu = {}
    # pass_record_gpu["HloModulePass"] = {}
    # pass_record_gpu["HloModulePass_consturctor"] = {}
    # pass_record_gpu["OpExpanderPass"] = {}
    # pass_record_gpu["OpExpanderPass_consturctor"] = {}
    # pass_class = []
    # for file_path in glob.glob(os.path.join(XLA_pass_directory, "gpu", "*.h")) + glob.glob(os.path.join(XLA_pass_directory, "gpu", "transforms", "*.h")):
    #     file_name = os.path.basename(file_path)
    #     with open(file_path, 'r') as f:
    #         for line in f:
    #             if "class" in line and ": public " in line and "{" in line:
    #                 try:
    #                     pass_name = re.search(r': public (\w+) {', line).group(1)
    #                     pass_class.append(pass_name)
    #                     if pass_name == 'Pass':
    #                         print(f"{line} in {file_name} is a pass!!!!!!!!!!!!!!!!!!")       
    #                 except:
    #                     print(f"{line} in {file_name} do not include pass class name")
    #             if "HloModulePass" in line:
    #                 try:
    #                     pass_name = re.search(r'class (\w+) : public', line).group(1)
    #                 except:
    #                     print(f"{line} in {file_name} do not include pass name")
    #                     continue
    #                 # Check if the pass is has consturctor
    #                 if pass_name + "(" in f.read():
    #                     pass_record_gpu["HloModulePass_consturctor"][pass_name] = file_name
    #                 else:
    #                     pass_record_gpu["HloModulePass"][pass_name] = file_name
    #             if "OpExpanderPass" in line:
    #                 try:
    #                     pass_name = re.search(r'class (\w+) : public', line).group(1)
    #                 except:
    #                     print(f"{line} in {file_name} do not include pass name")
    #                     continue
    #                 # Check if the pass is has consturctor
    #                 if pass_name + "(" in f.read():
    #                     pass_record_gpu["OpExpanderPass_consturctor"][pass_name] = file_name
    #                 else:
    #                     pass_record_gpu["OpExpanderPass"][pass_name] = file_name
    # # print(list(set(pass_class)))
    # print(len(pass_record_gpu["HloModulePass"]))
    # print(len(pass_record_gpu["HloModulePass_consturctor"]))
    # print(len(pass_record_gpu["OpExpanderPass"]))
    # print(len(pass_record_gpu["OpExpanderPass_consturctor"]))

    # number_all = 0
    # for key in pass_record:
    #     number_all += len(pass_record[key])
    # for key in pass_record_cpu:
    #     number_all += len(pass_record_cpu[key])
    # for key in pass_record_gpu:
    #     number_all += len(pass_record_gpu[key])
    # print(number_all)
        
# Get options in the algebraic simplifier
def get_algebraic_simplifier_options():
    algebraic_simplifier_options_path = os.path.join(XLA_pass_directory, "algebraic_simplifier.h")
    options_dict = {}
    with open(algebraic_simplifier_options_path, 'r') as f:
        for line in f:
            if line.find("_{") != -1 and line.find("};") != -1:
                value_type = line.strip().split(" ")[0]
                option_inf = line.strip().split(" ")[1].split("_")
                option_name = "_".join(option_inf[:-1])
                option_default_value = option_inf[-1][1:-2]
                options_dict[option_name] = (option_default_value, value_type)
    with open("extract_AlgebraOption.json", 'w') as file:
        json.dump(options_dict, file)


if __name__ == "__main__":
    # get_xla_pass()
    get_algebraic_simplifier_options()

