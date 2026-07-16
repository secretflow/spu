import shutil
import os
extend_number = 8
ori_testcase = '../sml'
bazel_list = []
for root, dirs, files in os.walk(ori_testcase):
    for file in files:
        if file == "BUILD.bazel":
            bazel_list.append(os.path.join(root, file))
# list of python files that import sml
python_list = []
for root, dirs, files in os.walk(ori_testcase):
    for file in files:
        # if file == "BUILD.bazel":
        #     bazel_list.append(os.path.join(root, file))
        if file.endswith(".py"):
            with open(os.path.join(root, file), 'r') as f:
                file_content = f.read()
                if 'import sml.' in file_content or 'from sml.' in file_content:
                    python_list.append(os.path.join(root, file))

for i in range(extend_number):
    current_number = i + 1
    current_testcase = ori_testcase + str(current_number)
    if os.path.exists(current_testcase):
        shutil.rmtree(current_testcase)
    shutil.copytree(ori_testcase, current_testcase)
    for bazel_file in bazel_list:
        current_bazel_file = bazel_file.replace(ori_testcase, current_testcase)
        with open(current_bazel_file, 'r') as f:
            bazel_content = f.readlines()
        with open(current_bazel_file, 'w') as f:
            for line in bazel_content:
                if "//sml" in line:
                    f.write(line.replace('//sml', '//sml' + str(current_number)))
                else:
                    f.write(line)
    for python_file in python_list:
        current_python_file = python_file.replace(ori_testcase, current_testcase)
        with open(current_python_file, 'r') as f:
            bazel_content = f.readlines()
        with open(current_python_file, 'w') as f:
            for line in bazel_content:
                f.write(line.replace('import sml.', f'import sml{str(current_number)}.').replace('from sml.', f'from sml{str(current_number)}.'))
    