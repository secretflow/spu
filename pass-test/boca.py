import os, sys
import time
import random
import numpy as np
import copy
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import math
import subprocess
import re
from enum import Enum
import threading
import select
random.seed(456)
iters = 60
begin2end = 5
# begin2end = 1
md = int(os.environ.get('MODEL', 1))
fnum = int(os.environ.get('FNUM', 8))
decay = float(os.environ.get('DECAY', 0.5))
scale = float(os.environ.get('SCALE', 10))
offset = float(os.environ.get('OFFSET', 20))

import functools
print = functools.partial(print, flush=True)

# cmd2 = ' -I ../polybench/utilities -I ../polybench/linear-algebra/kernels/2mm ../polybench/utilities/polybench.c ../polybench/linear-algebra/kernels/2mm/2mm.c -lm -DPOLYBENCH_TIME -o 2mm_time'
# cmd3 = 'gcc -O2 -funswitch-loops -ftree-vectorize -fpredictive-commoning -fipa-cp-clone -finline-functions -fgcse-after-reload -I ../polybench/utilities -I ../polybench/linear-algebra/kernels/2mm ../polybench/utilities/polybench.c ../polybench/linear-algebra/kernels/2mm/2mm.c -lm -DPOLYBENCH_TIME -o 2mm_time'
# cmd4 = 'rm -rf *.o *.I *.s a.out'
# cmd5 = './2mm_time'

sys.stdout = open('boca-log2.txt', 'w')
os.chdir("..")
# nodectl_process = subprocess.Popen("bazel-bin/examples/python/utils/nodectl up".split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

class mod(Enum):
    sml = 1
    example = 2 
# original_program = "pass-test/file-to-be-modified/sml/svm/tests/svm_test.py"
# test_mod = mod.sml
original_program = "pass-test/file-to-be-modified/examples/python/ml/ss_lr/ss_lr.py"
test_mod = mod.example
program_args= {}
log_directory = 'pass-test/test-log-boca/3'

if test_mod == mod.sml:
    test_program = original_program.replace('pass-test/file-to-be-modified/', '')
    program_inf = test_program.split("/")
    test_command = "/".join(["bazel-bin", "sml", program_inf[-3], f"{program_inf[-2]}", f"{program_inf[-1].split('.')[0]}"])
else:
    test_program = original_program.replace('pass-test/file-to-be-modified/', '')
    program_inf = test_program.split("/")
    test_command = "/".join(["bazel-bin", "examples", program_inf[-4], program_inf[-3], program_inf[-2], f"{program_inf[-1].split('.')[0]}"])
    if program_args:
        for key, value in program_args.items():
            test_command += f" --{key} {value}"

current_program_count = 0                                                         
options = []
baseline_send_actions = 0

def run_progrom_with_opts(pass_options):
    if pass_options != []:
        global current_program_count
        with open(original_program, 'r') as src_file:
            test_codes_base = src_file.readlines()
        modified_code = []
        for line in test_codes_base:
            modified_code.append(line)
            if 'copts = spu_pb2.CompilerOptions()' in line:
                indent = len(line) - len(line.lstrip())
                for option in pass_options:
                    modified_code.append(indent * " " + f'copts.{option} = True\n')
        with open(test_program, 'w') as dst_file:
            dst_file.writelines(modified_code)

        output_log_path = os.path.join(log_directory, f"test_{current_program_count}")
        if not os.path.exists(output_log_path):
            os.makedirs(output_log_path)
        with open(f"{output_log_path}/pass_closed.txt", 'w') as pass_file:
            pass_file.write('\n'.join(pass_options))

        output_file = os.path.join(output_log_path, "test_" + test_command.split("/")[-1].split(" ")[0] + ".txt")
        
    else:
        ### run the original program as baseline
        with open(original_program, 'r') as src_file:
            test_codes_base = src_file.readlines()
        with open(test_program, 'w') as dst_file:
            dst_file.writelines(test_codes_base)
        
        output_log_path = os.path.join(log_directory, f"test_baseline")
        if not os.path.exists(output_log_path):
            os.makedirs(output_log_path)

        output_file = os.path.join(output_log_path, "test_" + test_command.split("/")[-1].split(" ")[0] + ".txt")
    if pass_options != []:
        print(f"Running passoptions {pass_options} for {output_file} for {current_program_count} times")
    else:
        print(f"Running passoptions {pass_options} for {output_file} for baseline")
    if test_mod == mod.sml:
        with open(output_file, "w") as file:
            process = subprocess.run(test_command, stdout=file, stderr=subprocess.STDOUT)

    else:
        with open(output_file, "w") as file:
            nodectl_process = subprocess.Popen("bazel-bin/examples/python/utils/nodectl up".split(" "), stdin=subprocess.PIPE, stdout=file, stderr=subprocess.STDOUT, text=True)
            time.sleep(3)
            process = subprocess.run(test_command.split(" "), stderr=subprocess.STDOUT)
            nodectl_process.communicate("down")
        # with open(output_file, 'w', encoding='utf-8') as f_out:
        #     # stdout, stderr = process.communicate()
        #     nodectl_process.communicate()
        #     f_out.write(nodectl_process.stdout.read())

            # while True:
            #     line = nodectl_process.stdout.read()
            #     if not line:
            #         break
            #     f_out.write(line)
            #     f_out.flush()

            # stdout_fd = nodectl_process.stdout.fileno()
            # stderr_fd = nodectl_process.stderr.fileno()
            # while True:
            #     reads, _, _ = select.select([nodectl_process.stdout, nodectl_process.stderr], [], [])
            #     for read in reads:
            #         line = read.readline()
            #         if not line:
            #             break
            #         if read == nodectl_process.stdout:
            #             f_out.write(line)
            #             f_out.flush()
            #         elif read == nodectl_process.stderr:
            #             continue
            #     if nodectl_process.poll() is not None:
            #         break
        # with open(log_directory + '/node_log.txt', 'r') as src_file:
        #     with open(output_file, 'w') as dst_file:
        #         dst_file.write(src_file.read())
        # # Clean the content in log_directory + '/node_log.txt'
        # time.sleep(5)
        # with open(log_directory + '/node_log.txt', 'w') as file:
        #     file.write('')
                
    if pass_options != []:
        current_program_count += 1
    send_actions = 0
    try:
        with open(output_file, "r") as file:
            for line in file:
                if "Party_2|Link details" in line or "Party_1|Link details" in line or "Party_0|Link details" in line:
                    link_details_match = re.search(
                        r"Link details: total send bytes (\d+), recv bytes (\d+), send actions (\d+), recv actions (\d+)",
                        line
                    )
                    send_actions += int(link_details_match.group(3))
    except:
        send_actions = 0
        print(f"Running passoptions {pass_options} for {output_file} failed !!!!!!")
    print(send_actions)
    return send_actions
    


def generate_opts(independent):
    result = []
    for k, s in enumerate(independent):
        if s == 1:
            result.append(options[k])
    independent = result

    return independent

def get_objective_score(independent):
    independent = generate_opts(independent)

    send_actions = run_progrom_with_opts(independent)
    if send_actions == 0:
        return 0
    else:
        return -(baseline_send_actions/send_actions)
    
    # speedups = []
    # step = 0
    # while (len(speedups) < 6):
    #     step += 1
    #     if step > 10:
    #         print('failed configuration!')
    #         sys.exit(0)
    #     os.system(cmd4)
    #     print('gcc -O2 ' + ' '.join(independent) + cmd2)
    #     os.system('gcc -O2 ' + ' '.join(independent) + cmd2)
    #     begin = time.time()
    #     print(cmd5)
    #     ret = os.system(cmd5)
    #     if ret != 0:
    #         continue
    #     print(ret)
    #     end = time.time()
    #     de = end - begin
    #     os.system(cmd4)
    #     os.system(cmd3)

    #     begin = time.time()
    #     os.system(cmd5)
    #     end = time.time()
    #     nu = end - begin
       
    #     print('nu:' + str(nu) + ' de:' + str(de) + ' val:' + str(nu / de))
    #     speedups.append(nu / de)

    # print(speedups)
    # return -np.median(speedups)

def generate_conf(x):
    comb = bin(x).replace('0b', '')
    comb = '0' * (len(options) - len(comb)) + comb
    conf = []
    for k, s in enumerate(comb):
        if s == '1':
            conf.append(1)
        else:
            conf.append(0)
    return conf

class get_exchange(object):
    def __init__(self, incumbent):
        self.incumbent = incumbent

    def to_next(self, feature_id):
        ans = [0] * len(options)
        for f in feature_id:
            ans[f] = 1
        for f in self.incumbent:
            ans[f[0]] = f[1] 
        return ans

def do_search(train_indep, model, eta, rnum):
    features = model.feature_importances_
    print('features')
    print(features)
    
    b = time.time()
    feature_sort = [[i, x] for i, x in enumerate(features)]
    feature_selected = sorted(feature_sort, key=lambda x: x[1], reverse=True)[:fnum]
    feature_ids = [x[0] for x in feature_sort]
    neighborhood_iterators = []    
    for i in range(2 ** fnum):
        comb = bin(i).replace('0b', '')
        comb = '0' * (fnum - len(comb)) + comb
        inc = []
        for k, s in enumerate(comb):
            if s == '1':
                inc.append((feature_selected[k][0], 1))
            else:
                inc.append((feature_selected[k][0], 0))
        neighborhood_iterators.append(get_exchange(inc))
    print('time1:' + str(time.time() - b))

    s = time.time()
    neighbors = []
    r = 0
    print('rnum:' + str(rnum))
    for i, inc in enumerate(neighborhood_iterators):
        for j in range(1 + int(rnum)):
            selected_feature_ids = random.sample(feature_ids, random.randint(0, len(feature_ids)))
            n = neighborhood_iterators[i].to_next(selected_feature_ids)
            neighbors.append(n)
    print('neighbrslen:'+str(len(neighbors)))
    print('time2:' + str(time.time()-s))
    
    pred = []
    estimators = model.estimators_
    s = time.time()
    for e in estimators:
        pred.append(e.predict(np.array(neighbors)))
    acq_val_incumbent = get_ei(pred, eta)
    print('time3:' + str(time.time()-s))
   
    return [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]

def get_ei(pred, eta):
    pred = np.array(pred).transpose(1, 0)
    m = np.mean(pred, axis=1)
    s = np.std(pred, axis=1)

    def calculate_f():
        z = (eta - m) / s
        return (eta - m) * norm.cdf(z) + s * norm.pdf(z)
    
    if np.any(s == 0.0):
        s_copy = np.copy(s)
        s[s_copy == 0.0] = 1.0
        f = calculate_f()
        f[s_copy == 0.0] = 0.0
    else:
        f = calculate_f()

    return f

def get_nd_solutions(train_indep, training_dep, eta, rnum):
    predicted_objectives = []
    model = RandomForestRegressor()
    
    model.fit(np.array(train_indep), np.array(training_dep))
    estimators = model.estimators_

    pred = []
    for e in estimators:
        pred.append(e.predict(train_indep))
    train_ei = get_ei(pred, eta)

    #get_initial_points
    configs_previous_runs = [(x, train_ei[i]) for i, x in enumerate(train_indep)]
    configs_previous_runs_sorted = sorted(configs_previous_runs, key=lambda x: x[1], reverse=True)

    # do search
    begin = time.time()
    merged_predicted_objectives = do_search(train_indep, model, eta, rnum)
    merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)
    end = time.time()
    print('search time:' + str(begin - end)) 

    begin = time.time()
    for x in merged_predicted_objectives:
        if x[0] not in train_indep:
            print('no repete time:' + str(time.time() - begin))
            return x[0], x[1]

def get_training_sequence(training_indep, training_dep, testing_indep, rnum):
    return_nd_independent, predicted_objectives = get_nd_solutions(training_indep, training_dep, testing_indep, rnum)
    return return_nd_independent, predicted_objectives

def main():
    training_indep = []
    ts = []
    initial_sample_size = 2
    rnum0 = int(os.environ.get('RNUM', 2 ** 8))
    b = time.time()
    sigma = -scale ** 2 / (2 * math.log(decay))

    # initial sampling until there is not failed configuration
    # while True:
        # while len(training_indep) < initial_sample_size:
        #     x = random.randint(0, 2 ** len(options))
        #     x = generate_conf(x)
        #     if x not in training_indep:
        #         training_indep.append(x)
        #         ts.append(time.time() - b)

        # training_dep = [get_objective_score(r) for r in training_indep]
        # if 0 not in training_dep:
        #     break
    training_dep = []
    while len(training_dep) < initial_sample_size:
        x = random.randint(0, 2 ** len(options))
        x = generate_conf(x)
        if x not in training_indep:
            score = get_objective_score(x)
            if 0 != score:
                training_indep.append(x)
                training_dep.append(score)
    steps = 0
    budget = iters
    result = 1e8

    for i, x in enumerate(training_dep):
        if result > x:
            result = x

    while initial_sample_size + steps < budget:
        steps += 1
        rnum = rnum0 * math.exp(-max(0, len(training_indep) - offset) ** 2 / (2 * sigma ** 2))
        best_solution, return_nd_independent = get_training_sequence(training_indep, training_dep, result, rnum)
        print('best_solution')
        print(best_solution)
        training_indep.append(best_solution)
        ts.append(time.time() - b)
        best_result = get_objective_score(best_solution)
        print(best_result)
        # if there is a failed configuration, we do not update the result
        if best_result == 0:
            training_indep.pop()
            ts.pop()
            continue
        training_dep.append(best_result)
        if best_result < result:
            result = best_result
    
    return training_dep, ts


if __name__ == '__main__':
    # os.chdir("..")
    # with open(log_directory + '/node_log.txt', 'w') as file:
    #     nodectl_process = subprocess.Popen("bazel-bin/examples/python/utils/nodectl up".split(" "), stdout=file, stderr=subprocess.STDOUT)
    # time.sleep(5)
    passoptions_path = '/home1/leiyu.lyc/ppu/pass-test/pass_options.txt'
    with open(passoptions_path, 'r') as file:
        for line in file:
            if line.strip():
                options.append(line.strip())
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # run the original program as baseline
    baseline_send_actions = run_progrom_with_opts([])
    stats = []
    times = []

    for i in range(begin2end):
        dep, ts = main()
        print('middle result')
        print(dep)
        stats.append(dep)
        times.append(ts)
        
    vals = []
    for j, v_tmp in enumerate(stats):
        max_s = 0
        for i, v in enumerate(v_tmp):
            max_s = min(max_s, v)
            v_tmp[i] = max_s

    print(times)
    print(stats)

    # for i in range(iters):
    #     tmp = []
    #     for j in range(begin2end):
    #         tmp.append(times[j][i])
    #     vals.append(-np.mean(tmp))

    # print(vals)

    # vals = []
    # for i in range(iters):
    #     tmp = []
    #     for j in range(begin2end):
    #         tmp.append(stats[j][i])
    #     vals.append(-np.mean(tmp))

    # print(vals)

    # vals = []
    # for i in range(iters):
    #     tmp = []
    #     for j in range(begin2end):
    #         tmp.append(stats[j][i])
    #     vals.append(-np.std(tmp))

    # print(vals)
