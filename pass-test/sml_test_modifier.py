"""Noted that this script can not modify all the tests. You should manually modify the rest of the tests!!!"""
import re
import os 

test_file_path = "sml/preprocessing/tests/preprocessing_test.py"
input_path = os.path.join("file-to-be-modified", test_file_path)
output_path = os.path.join("..", test_file_path)

def modify_fucntion(to_modify_fn_lines):
    """note that when parameter contains ), there will be problem!!!"""
    pattern = re.compile(r'(spsim\.sim_jax\(\s*sim\s*,\s*\w+(?:\s*,\s*\w+\s*=\s*[^)]+)*\))')
    match = pattern.search(to_modify_fn_lines[0])
    extracted_function = match.group(0)
    # print(extracted_function)

    indent = len(to_modify_fn_lines[0]) - len(to_modify_fn_lines[0].lstrip())
    modified_fn_lines = []
    modified_fn_lines.append(indent * " " + "\n")
    modified_fn_lines.append(indent * " " + "copts = spu_pb2.CompilerOptions()\n")
    modified_fn_lines.append(indent * " " + "\n")
    modified_fn_lines.append(indent * " " + f"spu_fn = {extracted_function[:-1]}, copts=copts)\n")
    to_modify_fn_lines[0] = to_modify_fn_lines[0].replace(extracted_function, "spu_fn")
    for line in to_modify_fn_lines:
        modified_fn_lines.append(line)
    modified_fn_lines.append(indent * " " + "print(spu_fn.pphlo)\n")
    return modified_fn_lines
        
print("Noted that this script can not modify all the tests. You should manually modify the rest of the tests!!!")
with open(input_path, 'r') as file:
    content = file.readlines()

modified_lines_all = []
spu_fn_flag = False

modified_fn_lines = []
for line in content:
    if spu_fn_flag:
        to_modify_fn_lines.append(line)
        if line.lstrip().startswith(")"):
            modified_fn_lines = modify_fucntion(to_modify_fn_lines)
            modified_lines_all.extend(modified_fn_lines)
            spu_fn_flag = False
    elif "spsim.sim_jax(sim, " in line:
        if ")\n" in line:
            to_modify_fn_lines = [line]
            modified_fn_lines = modify_fucntion(to_modify_fn_lines)
            modified_lines_all.extend(modified_fn_lines)
        else:
            spu_fn_flag = True
            to_modify_fn_lines = [line]
    else:
        modified_lines_all.append(line)
with open(output_path, 'w') as file:
    file.writelines(modified_lines_all)