import os
import glob
import json

test_file_dir = "file-to-be-modified/sml_single"

pphlo_list = []
test_file_list = glob.glob(os.path.join(test_file_dir, "*", "*", "*_test*"))
for test_file in test_file_list:
    with open(test_file, 'r') as src_file:
        for line in src_file.readlines():
            if line.find("{pphlo_list}") != -1:
                pphlo_list.append(line.split("#")[1].strip())
with open("/home1/leiyu.lyc/ppu/pass-test/test-log/sml-new2/baseline/extract_result.json", 'r') as file:
    baseline_json = json.load(file)

for pphlo in pphlo_list:
    if pphlo not in baseline_json.keys():
        print(pphlo)
        
for pphlo in baseline_json.keys():
    if pphlo not in pphlo_list:
        print(pphlo)
