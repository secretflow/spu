import re
import shutil
import json

number_record = 28
add_record = 0

# ADD .bazel will make the file to be compiled by bazel
bazel_path_input = 'file-to-be-modified/BUILD_ori'
bazel_path_output = '../libspu/compiler/front_end/BUILD.bazel'

# Add new passes to SPU
def pass_adder_new(pass_to_be_added, indent):
    global add_record
    modified_lines_add = []
    passoption_name_list = []
    for pass_new in pass_to_be_added:
        passoption_name = "able_" + pass_new.lower() + f"_{add_record}"
        passoption_name_list.append(passoption_name)
        modified_lines_add.append(indent * " " + f'if (compiler_options.{passoption_name}()) ' + "{")
        modified_lines_add.append(indent * " " + f'  pipeline.AddPass<{pass_new}>();')
        modified_lines_add.append(indent * " " + '}')
    add_record += 1
    return modified_lines_add, passoption_name_list
        

def pass_adder_modifier(passfile_path_input, passfile_path_output, optionsfile_path_input, optionsfile_path_output, passoption_list_path, passadd_list_path=None, algebra_modify=False):
    global number_record
    with open(passfile_path_input, 'r') as file:
        content = file.read()
    
    # Load the pass to be added from the json file
    if passadd_list_path:
        with open(passadd_list_path, 'r') as file:
            pass_to_be_added_json = json.load(file)
        pass_to_be_added = list(pass_to_be_added_json["HloModulePass"].keys()) + list(pass_to_be_added_json["OpExpanderPass"].keys())
        file_to_be_imported = list(pass_to_be_added_json["HloModulePass"].values()) + list(pass_to_be_added_json["OpExpanderPass"].values())
        file_to_be_imported = list(set(file_to_be_imported))
    
    # Load the algebraic simplifier options from the json file
    if algebra_modify:
        with open('extract_AlgebraOption.json', 'r') as file:
            algebraic_simplifier_options_dict = json.load(file)

    # Extract the section between the start and end comments
    start_flag = False
    lines = []
    imported_file = []
    option_start_flag = False
    algebraic_simplifier_option_block = []
    for line in content.split('\n'):
        if passadd_list_path:
            if line.find("#include ") != -1:
                imported_file.append(line.strip())
        if algebra_modify:
            if line.find("// Simplifier options") != -1:
                option_start_flag = True
            if option_start_flag:
                algebraic_simplifier_option_block.append(line)
                if line.find("// End of simplifier options") != -1:
                    option_start_flag = False
        if '// End of modifying Flags to enable/disable passes' in line:
            break
        if start_flag:
            lines.append(line)
        if '// Start to modify Flags to enable/disable passes' in line:
            start_flag = True

    # Extract the pass blocks and store them in a dictionary
    pass_block_dict = {}
    pass_flag = 0
    pass_block = []
    passoption_name = ''
    for line in lines:
        if pass_flag == 1:
            pass_block.append(line)
            if line.find(');') != -1:
                pass_block_dict[passoption_name] = pass_block
                pass_flag = 0
                pass_block = []
                passoption_name = ''
        ### A Special pass in hlo_importer_ori.cc contains "simplification" serves as a templete
        elif (line.find('AddPass') != -1 or line.find('addPass') != -1)and line.find('"simplification"') == -1:
            pass_block.append(line)
            try:
                passoption_name =  "disable_" + re.findall(r'pipeline\.AddPass<([^>]+)>', line)[0].lower().replace("::", "_")
            except:
                passoption_name = "disable_" + re.findall(r'addPass\(([^()]+)\(\)\)', line)[0].lower().replace("::", "_")
                # print(re.findall(r'.addPass(([^>]+))', line))
                # passoption_name = "disable_" + re.findall(r'.addPass(([^>]+))', line)[0].lower()
            if line.find(');') != -1:
                if passoption_name in pass_block_dict:
                    passoption_name = passoption_name + "_1"
                    pass_block_dict[passoption_name] = pass_block
                else:
                    pass_block_dict[passoption_name] = pass_block
                pass_block = []
                passoption_name = ''
            else:
                pass_flag = 1
        else:
            pass_block_dict["no_" + line] = line
    
    modified_lines = []
    passoption_list = []
    # insert add test at the beginning of the fixpoint block
    inserted_begin_fixpoint = False
    if passadd_list_path:
        modified_lines_new, passoption_list_new = pass_adder_new(pass_to_be_added, 2)
        modified_lines = modified_lines + modified_lines_new
        passoption_list = passoption_list + passoption_list_new
    for key in pass_block_dict:
        if key.startswith("no_"):
            modified_lines.append(pass_block_dict[key])
        else:
            pass_block = pass_block_dict[key]
            indent = len(pass_block[0]) - len(pass_block[0].lstrip())
            # insert add test at the beginning of the fixpoint block
            if passadd_list_path and indent == 4 and inserted_begin_fixpoint == False:
                modified_lines_new, passoption_list_new = pass_adder_new(pass_to_be_added, indent)
                modified_lines = modified_lines + modified_lines_new
                passoption_list = passoption_list + passoption_list_new
                inserted_begin_fixpoint = True
            modified_lines.append(indent * " " + f'if (!compiler_options.{key}()) ' + "{")
            """ Uncomment to confirm that the opts can control whether the pass is added """
            # modified_lines.append(f'    std::cout << "Pass {key} has been added" << std::endl;')
            for pass_line in pass_block:
                modified_lines.append("  " + pass_line)
            modified_lines.append(indent * " " + "}")
            if passadd_list_path:
                modified_lines_new, passoption_list_new = pass_adder_new(pass_to_be_added, indent)
                modified_lines = modified_lines + modified_lines_new
                passoption_list = passoption_list + passoption_list_new

    # Remove all keys that start with "no_", which are the lines that are not passes
    keys_to_remove = [key for key in pass_block_dict if key.startswith("no_")]
    for key in keys_to_remove:
        del pass_block_dict[key]
    
    passoption_algebra_list = []
    algebra_modified_lines = []
    if algebra_modify:
        # Change the default value of the algebraic simplifier options according to the hlo_importer
        for line in algebraic_simplifier_option_block:
            if line.strip().startswith("options"):
                value = line.split("(")[1].split(")")[0]
                option_name = "_".join(line.split("(")[0].split("_")[1:])
                if option_name in algebraic_simplifier_options_dict:
                    algebraic_simplifier_options_dict[option_name][0] = value
                else:
                    print(f"Option {option_name} is not in the options_dict")
        
        # Generate the modified lines for the algebraic simplifier options
        indent = len(algebraic_simplifier_option_block[0]) - len(algebraic_simplifier_option_block[0].lstrip())
        for option in algebraic_simplifier_options_dict:
            value, value_type = algebraic_simplifier_options_dict[option]
            option_formatted = option.replace('_', '')
            if value_type == "bool":
                if value == "true":
                    value_set = ["false"]
                else:
                    value_set = ["true"]
                algebra_set_option = [f"set{value_set[0]}_{option_formatted}"]
            elif value_type.startswith("int"):
                value_set = [str(int(value) - 1), str(int(value) + 1)]
                algebra_set_option = [f"setless_{option_formatted}", f"setmore_{option_formatted}"]
            elif value_type.startswith("double"):
                value_set = [str(float(value) - 1.0), str(float(value) + 1.0)]
                algebra_set_option = [f"setless_{option_formatted}", f"setmore_{option_formatted}"]
            else:
                print(f"Unknown value type {value_type}")
            for i in range(len(value_set)):
                algebra_modified_lines.append(indent * " " + f'if (compiler_options.{algebra_set_option[i]}()) ' + "{")
                algebra_modified_lines.append(indent * " " + f'  options.{"set_" + option}({value_set[i]});')
                algebra_modified_lines.append(indent * " " + '}')
            passoption_algebra_list = passoption_algebra_list + algebra_set_option

    # the corresponding file to be imported, there is not file has been added in original version
    file_to_be_imported_bazel = []

    # Replace the original lines surrounded by the start and end comments with the modified lines
    with open(passfile_path_output, 'w') as file:
        start_flag = False
        for line in content.split('\n'):
            if passadd_list_path:
                if '#include "xla/service/hlo.pb.h"' in line:
                    file.write(line + '\n')
                    for file_h in file_to_be_imported:
                        import_code = f'#include "xla/service/{file_h}"'
                        if import_code not in imported_file:
                            file.write(import_code + '\n')
                            file_to_be_imported_bazel.append('"' + f"@xla//xla/service:{file_h.split('.')[0]}" + '"')
                    continue
            if algebra_modify:
                if line.find("// End of simplifier options") != -1:
                    file.write(line + '\n')
                    for modified_line in algebra_modified_lines:
                        file.write(modified_line + '\n')
            if '// Start to modify Flags to enable/disable passes' in line:
                start_flag = True
                file.write(line + '\n')
                for modified_line in modified_lines:
                    file.write(modified_line + '\n')
            elif '// End of modifying Flags to enable/disable passes' in line:
                start_flag = False
                file.write(line + '\n')
            elif not start_flag:
                file.write(line + '\n')

    if passadd_list_path:
        # Add the corresponding file import to be imported to the bazel BUILD file
        with open(bazel_path_input, 'r') as file:
            bazel_content = file.readlines()
        with open(bazel_path_output, 'w') as file:
            start_flag = False
            for line in bazel_content:
                file.write(line)
                if '@xla//xla/translate/hlo_to_mhlo:hlo_module_importer' in line:
                    for file_h in file_to_be_imported_bazel:
                        file.write("        " + file_h + ',\n')
                        
    with open(optionsfile_path_input, 'r') as file:
        content_options = file.read()

    # Save all the keys in the pass_block_dict into a list
    # Collect all the pass options from add test, delete test, and algebraic simplifier options
    passoption_list = passoption_list + list(pass_block_dict.keys()) + passoption_algebra_list

    # the current number is the last number in the options file
    # can be changed to the number of the last option in the file
    current_number = number_record
    with open(optionsfile_path_output, 'w') as file:
        start_flag = False
        for line in content_options.split('\n'):
            file.write(line + '\n')
            if 'bool' in line and f" = {current_number};" in line:
                file.write('\n')
                file.write('  // Modified Options for Compilation Options\n')
                for passoption in passoption_list:
                    current_number += 1
                    file.write(f'  bool {passoption} = {current_number};\n')
    number_record = current_number

    # write all options to a file
    with open(passoption_list_path, 'w') as file:
        for key in passoption_list:
            file.write(key + '\n')

if __name__ == '__main__':
    optionsfile_path_input = 'file-to-be-modified/spu_ori.proto'
    optionsfile_path_tmp = 'file-to-be-modified/spu_tmp.proto'
    # pass neeed to be added from XLA
    passadd_list_path = "extract_XLAPass.json"

    shutil.copyfile(optionsfile_path_input, optionsfile_path_tmp)
    passfile_path_input = 'file-to-be-modified/hlo_importer_ori.cc'
    passfile_path_output = '../libspu/compiler/front_end/hlo_importer.cc'
    optionsfile_path_output = '../libspu/spu.proto'
    passoption_list_path = 'pass_options_HLO.txt'
    pass_adder_modifier(passfile_path_input, passfile_path_output, optionsfile_path_tmp, optionsfile_path_output, passoption_list_path, passadd_list_path, algebra_modify=True)
    shutil.copyfile(optionsfile_path_output, optionsfile_path_tmp)

    delete_list = []
    add_list = []
    algebra_list = []
    with open(passoption_list_path, 'r') as file:
        for line in file.readlines():
            if line.startswith("disable") == True:
                delete_list.append(line.strip())
            elif line.startswith("able") == True:
                add_list.append(line.strip())
            else:
                algebra_list.append(line.strip())
    with open('pass_options_HLO_add.txt', 'w') as file:
        for line in add_list:
            file.write(line + '\n')
    
    with open('pass_options_HLO_delete.txt', 'w') as file:
        for line in delete_list:
            file.write(line + '\n')

    with open('pass_options_HLO_algebra.txt', 'w') as file:
        for line in algebra_list:
            file.write(line + '\n')

    passfile_path_input = 'file-to-be-modified/fe_ori.cc'
    passfile_path_output = '../libspu/compiler/front_end/fe.cc'
    passoption_list_path = 'pass_options_fe.txt'
    pass_adder_modifier(passfile_path_input, passfile_path_output, optionsfile_path_tmp, optionsfile_path_output, passoption_list_path)
