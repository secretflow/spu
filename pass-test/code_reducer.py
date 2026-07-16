import os
import subprocess
from extract_inf import split_log_party, extract_log
import json
import shutil
import builtins
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import random

class snippet():
    def __init__(self, operator_list, IR_inf, block_inf, partynum, protocalchosen, log_path, pass_option_mut=None, matrix="bytes"):
        self.operator_list = operator_list
        self.IR_inf = IR_inf
        self.block_inf = block_inf
        self.index_last = int(self.operator_list[-1].split(".")[1])
        self.partynum = partynum
        self.protocalchosen = protocalchosen
        self.log_path = log_path
        self.pass_option_mut = pass_option_mut
        self.matrix = matrix

        self.ori_log_path = os.path.join(self.log_path, "ori")
        self.mut_log_path = os.path.join(self.log_path, "mut")
        
    def _requirements_check(self):
        input_need_list = []
        output_need_list = []
        call_need_list = []
        for output_var_name in self.operator_list[::-1]:
            output_inf = self.IR_inf[output_var_name]
            input_var_list = output_inf["input_var_list"]
            var_type = output_inf["var_type"]
            parameters = output_inf["parameters"]
            if "to_apply=" in parameters:
                call_need_list.append(parameters.split("to_apply=")[-1])
            # deal with while
            if "condition=" in parameters and "body=" in parameters:
                remain_in, block_2 = parameters.split(", body=")
                block_1 = remain_in.split("condition=")[-1]
                call_need_list.append(block_1)
                call_need_list.append(block_2)
            if output_var_name in input_need_list and var_type != "parameter":    
                input_need_list.remove(output_var_name)
            if var_type == "operation":
                for input_var in input_var_list:
                    if input_var not in input_need_list:
                        input_need_list.append(input_var)
            if var_type == "parameter":
                if output_var_name not in input_need_list:
                    input_need_list.append(output_var_name)
        
        for output_var_name in self.operator_list:
            output_inf = self.IR_inf[output_var_name]
            input_var_list = output_inf["input_var_list"]
            var_type = output_inf["var_type"]
            if var_type == "operation":
                for input_var in input_var_list:
                    if input_var in output_need_list:
                        output_need_list.remove(input_var)
            # deal with the tuple return from original code
            if output_var_name.startswith("ROOT tuple") == False:
                output_need_list.append(output_var_name)
            else:
                for input_var in input_var_list:
                    output_need_list.append(input_var)

        self.input_need_list = input_need_list
        self.output_need_list = output_need_list
        # recuvrsively add the block needed by the call
        self.call_need_list = []
        while len(call_need_list) != 0:
            call_var = call_need_list.pop()
            self.call_need_list.append(call_var)
            if call_var != "ENTRY" and self.block_inf[call_var][1] != []:
                call_need_list += self.block_inf[call_var][1]
        # reverse the call_need_list to make sure the block is added in the right order
        self.call_need_list = self.call_need_list[::-1]

    def _input_adder(self):
        self.input_need_list.sort(key=lambda x: int(x.split(".")[1]))
        input_add_code = []
        parameter_shape_list = []
        parameter_count = 0
        for input_var in self.input_need_list:
            input_inf = self.IR_inf[input_var]
            if input_inf["var_type"] == "constant":
                input_add_code.append(input_inf["line"])
            else:
                input_shape = input_inf['shape']
                input_shape_split = input_shape.split(", ")
                # deal with the case that the input is a tuple
                if len(input_shape_split) > 1:
                    print("dealing with tuple input")
                    input_single_list = []
                    for input_i, input_shape_single in enumerate(input_shape_split):
                        if "(" in input_shape_single:
                            input_shape_single = input_shape_single.split("(")[1]
                        if ")" in input_shape_single:
                            input_shape_single = input_shape_single.split(")")[0]
                        if "/*index=5*/" in input_shape_single:
                            input_shape_single = input_shape_single.split("/*index=5*/")[1]
                        input_single = input_var.replace(".", f"-{input_i}.")
                        input_single_list.append(input_single)
                        input_add_code.append(f"{input_single} = {input_shape_single} parameter({parameter_count})")
                        parameter_shape_list.append(input_shape_single)
                        parameter_count += 1
                    input_add_code.append(f"{input_var} = {input_shape} tuple({', '.join(input_single_list)})")
                else:
                    input_add_code.append(f"{input_var} = {input_shape} parameter({parameter_count})")
                    parameter_shape_list.append(input_shape)
                    parameter_count += 1
        self.input_add_code = input_add_code 
        self.parameter_shape_list = parameter_shape_list

    def _output_adder(self):
        output_add_code = []
        return_shape_list = []
        for output_var_name in self.output_need_list:
            output_inf = self.IR_inf[output_var_name]
            return_shape_list.append(output_inf['shape'])
        output_add_code.append(f"ROOT tuple.{self.index_last + 1} = ({', '.join(return_shape_list)}) tuple({', '.join(self.output_need_list)})")
        self.output_add_code = output_add_code
        self.return_shape_list = return_shape_list
    
    def code_generator(self):
        self._requirements_check()
        self._input_adder()
        self._output_adder()
        code_main = self.input_add_code
        for output_var_name in self.operator_list:
            line_inf = self.IR_inf[output_var_name]
            # Do not add line for parameter, since it has been added in input_add_code
            if line_inf["var_type"] == "parameter":
                continue
            code_main.append(line_inf["line"])
        
        code_call = []
        for call_var in self.call_need_list:
            call_block = self.block_inf[call_var][0]
            code_call += call_block + [""]
        
        # TODO: the logic there is dirty, need to be refactored
        # If there is only one output
        if len(self.output_need_list) == 1:
            # If the last output is not ROOT, directly add last line as ROOT
            if self.operator_list[-1].find("ROOT") == -1:
                code_main[-1] = f"ROOT {self.IR_inf[self.operator_list[-1]]['line']}"
            # If the last output is ROOT, directly use last line as output_add_code. So the output_add_code is not added in code_main
            code_all = ["HloModule jit_test, entry_computation_layout={" + f"({', '.join(self.parameter_shape_list)})->{self.return_shape_list[0]}" + "}, frontend_attributes={xla.sdy.meshes={}}", ""]
            code_all += code_call
            code_all += [f"ENTRY main.{self.index_last + 1} " + "{"]
        else:
            # If the last output is ROOT, remove the ROOT
            if self.operator_list[-1].find("ROOT") != -1:
                code_main[-1] = code_main[-1].replace("ROOT ", "")
            code_main = code_main + self.output_add_code
            code_all = ["HloModule jit_test, entry_computation_layout={" + f"({', '.join(self.parameter_shape_list)})->({', '.join(self.return_shape_list)})" + "}, frontend_attributes={xla.sdy.meshes={}}", ""]
            code_all += code_call
            code_all += [f"ENTRY main.{self.index_last + 2} " + "{"]
        
        code_all += ["  " + code for code in code_main]
        code_all += ["}"]
        self.code_exec = "\n".join(code_all)
    
    def code_executer(self):
        original_work_dir = os.getcwd()
        work_dir = ".."
        os.chdir(work_dir)

        if self.pass_option_mut == None:
            print("Error: code reducer for functionality test is not implemented yet!!!!!!!!!!!!")
            return

        with open(os.path.join("pass-test", self.log_path, "IR.txt"), "w") as f:
            f.write(self.code_exec)
        input_arg_list = []
        for parameter_shape in self.parameter_shape_list:
            # Scalar
            if "[]" in parameter_shape:
                var_type = parameter_shape.split("[")[0]
                if var_type == "s32":
                    input_arg = "1"
                elif var_type == "f32":
                    input_arg = "1.0"
                elif var_type == "pred":
                    input_arg = "True"
                else:
                    print(f"Error: Unprocessed scalar type {var_type}!!!!!!!!!!!!")
            # Tensor
            else:
                var_type = parameter_shape.split("[")[0]
                shape = parameter_shape.split("[")[1].split("]")[0]
                if var_type == "s32":
                    input_arg = f"np.ones(({shape}), dtype=np.int32)"
                elif var_type == "f32":
                    input_arg = f"np.ones(({shape}))"
                elif var_type == "pred":
                    input_arg = f"np.ones(({shape}), dtype=np.bool)"
                else:
                    print(f"Error: Unprocessed tensor type {var_type}!!!!!!!!!!!!")
            input_arg_list.append(input_arg)

        with open("pass-test/file-to-be-modified/hlo_debug.py", "r") as f:
            test_codes_base = f.readlines()
        
        # Run the test code without pass modification
        modified_code = []
        for line in test_codes_base:
            if line.find("{partynum}"):
                line = line.replace('{partynum}', str(self.partynum)).replace('{protocalchosen}', self.protocalchosen)
            if line.find("{IR_to_be_tested}"):
                line = line.replace("{IR_to_be_tested}", self.code_exec)
            if line.find("{input_args}"):
                line = line.replace("{input_args}", f"[{', '.join(input_arg_list)}]")
            if line.find("{return_number}"):
                line = line.replace("{return_number}", str(len(self.return_shape_list)))
            modified_code.append(line)
            if 'copts = spu_pb2.CompilerOptions()' in line:
                indent = len(line) - len(line.lstrip())
                for pass_option in self.pass_option_mut["ori"]:
                    modified_code.append(indent * " " + f'copts.{pass_option} = True\n')
        
        with open("spu/tests/hlo_debug.py", "w") as f:
            f.writelines(modified_code)

        ori_log_path = os.path.join("pass-test", self.ori_log_path)
        os.makedirs(ori_log_path, exist_ok=True)
        with open(os.path.join(ori_log_path, "test_0.txt"), "w") as file:
            process = subprocess.Popen("bazel-bin/spu/tests/hlo_debug", stdout=file, stderr=subprocess.STDOUT)
            process.communicate()
        
        # Run the test code without pass modification
        modified_pass_code = []
        for line in modified_code:
            # ignore the mutations form baseline
            if line.find("copts.") != -1:
                continue
            modified_pass_code.append(line)
            if 'copts = spu_pb2.CompilerOptions()' in line:
                indent = len(line) - len(line.lstrip())
                for pass_option in self.pass_option_mut["mut"]:
                    modified_pass_code.append(indent * " " + f'copts.{pass_option} = True\n')
        
        with open("spu/tests/hlo_debug.py", "w") as f:
            f.writelines(modified_pass_code)
        
        mut_log_path = os.path.join("pass-test", self.mut_log_path)
        os.makedirs(mut_log_path, exist_ok=True)
        with open(os.path.join(mut_log_path, "test_0.txt"), "w") as file:
            process = subprocess.Popen("bazel-bin/spu/tests/hlo_debug", stdout=file, stderr=subprocess.STDOUT)
            process.communicate()
        os.chdir(original_work_dir)
    
    def evaluate(self):
        split_log_party(self.ori_log_path, self.partynum, pphlo_log = False)
        extract_log(self.ori_log_path, pphlo_log = False)
        split_log_party(self.mut_log_path, self.partynum, pphlo_log = False)
        extract_log(self.mut_log_path, pphlo_log = False)
        with open(os.path.join(self.ori_log_path, "extract_result.json"), "r") as f:
            ori_extract_result = json.load(f)
        with open(os.path.join(self.mut_log_path, "extract_result.json"), "r") as f:
            mut_extract_result = json.load(f)
        ori_evaluate_value = 0
        # in some cases, the reduced case can be compiled but cannot be executed
        if "test" not in ori_extract_result:
            print(f"Error: test not in ori_extract_result for {self.ori_log_path}!!!!!!!!!!!!")
            return False
        for party in ori_extract_result["test"]["profile"]["repeat_0"]:
            ori_evaluate_value += int(ori_extract_result["test"]["profile"]["repeat_0"][party][self.matrix])
        mut_evaluate_value = 0
        # in some cases, the reduced case can be compiled but cannot be executed
        if "test" not in mut_extract_result:
            print(f"Error: test not in mut_extract_result for {self.mut_log_path}!!!!!!!!!!!!")
            return False
        for party in mut_extract_result["test"]["profile"]["repeat_0"]:
            mut_evaluate_value += int(mut_extract_result["test"]["profile"]["repeat_0"][party][self.matrix])
        
        # CHEETAH has the problem that the communication bytes is random
        if self.protocalchosen == "CHEETAH" and self.matrix == "send_bytes":
            if ori_evaluate_value!= 0 and mut_evaluate_value/ori_evaluate_value < 0.99:
                return True
            else:
                return False
        else: 
            if ori_evaluate_value > mut_evaluate_value:
                return True
            else:
                return False
        
class full_snippet(snippet):
    def __init__(self, code_exec, IO_line, partynum, protocalchosen, log_path, pass_option_mut=None, matrix="bytes"):
        self.code_exec = code_exec
        self.IO_line = IO_line
        self.partynum = partynum
        self.protocalchosen = protocalchosen
        self.log_path = log_path
        self.pass_option_mut = pass_option_mut
        self.matrix = matrix

        self.ori_log_path = os.path.join(self.log_path, "ori")
        self.mut_log_path = os.path.join(self.log_path, "mut")
    
    def code_generator(self):
        parameter_shape_list = []
        for input_fragment in self.IO_line.split("layout={")[-1].split("->")[0].split(", "):
            if "(" in input_fragment:
                input_fragment = input_fragment.split("(")[1]
            if ")" in input_fragment:
                input_fragment = input_fragment.split(")")[0]
            parameter_shape_list.append(input_fragment)
        return_shape_list = []
        for output_fragment in self.IO_line.split("->")[-1].split("}, frontend")[0].split(", "):
            if "(" in output_fragment:
                output_fragment = output_fragment.split("(")[1]
            if ")" in output_fragment:
                output_fragment = output_fragment.split(")")[0]
            return_shape_list.append(output_fragment)
        self.parameter_shape_list = parameter_shape_list
        self.return_shape_list = return_shape_list
    
    def cost_compare(self):
        split_log_party(self.ori_log_path, self.partynum, pphlo_log = False)
        extract_log(self.ori_log_path, pphlo_log = False, operator_profile = True)
        split_log_party(self.mut_log_path, self.partynum, pphlo_log = False)
        extract_log(self.mut_log_path, pphlo_log = False, operator_profile = True)
        with open(os.path.join(self.ori_log_path, "extract_result.json"), "r") as f:
            ori_extract_result = json.load(f)
        with open(os.path.join(self.mut_log_path, "extract_result.json"), "r") as f:
            mut_extract_result = json.load(f)
        ori_operator_cost = {}
        for i, party in enumerate(ori_extract_result["test"]["profile"]["repeat_0"]):
            HLO_profile = ori_extract_result["test"]["profile"]["repeat_0"][party]["operator_profile"]["HLO_profile"]
            for operator in HLO_profile:
                if i == 0:
                    ori_operator_cost[operator] = int(HLO_profile[operator][self.matrix])
                else:
                    ori_operator_cost[operator] += int(HLO_profile[operator][self.matrix])

        mut_operator_cost = {}
        for i, party in enumerate(mut_extract_result["test"]["profile"]["repeat_0"]):
            HLO_profile = mut_extract_result["test"]["profile"]["repeat_0"][party]["operator_profile"]["HLO_profile"]
            for operator in HLO_profile:
                if i == 0:
                    mut_operator_cost[operator] = int(HLO_profile[operator][self.matrix])
                else:
                    mut_operator_cost[operator] += int(HLO_profile[operator][self.matrix])

        predict_factor = {}
        special_operator_none = []
        special_operator_nocommu = []
        operator_pphlo_to_hlo = {
            "equal": "compare",
            "not_equal": "compare",
            "greater_equal": "compare",
            "greater": "compare",
            "less_equal": "compare",
            "less": "compare",
            "dynamic_slice": "dynamic-slice"
        }
        for operator in ori_operator_cost:
            operator_name = operator.split(".")[1]
            if operator_name in operator_pphlo_to_hlo:
                operator_name = operator_pphlo_to_hlo[operator_name]
            if operator not in mut_operator_cost:
                print(f"Operator {operator} not found in mut_operator_cost!!!!!!!!!!!!")
                # special_operator_none.append(operator_name)
                continue
            if ori_operator_cost[operator] != 0 and mut_operator_cost[operator] == 0:
                print(f"Operator {operator} has no communication cost in mut_operator_cost!!!!!!!!!!!!")
                # special_operator_nocommu.append(operator_name)
                continue
            if ori_operator_cost[operator] > mut_operator_cost[operator]:
                # remove pphlo. prefix
                predict_factor[operator_name] = ori_operator_cost[operator]/mut_operator_cost[operator] - 1
            elif ori_operator_cost[operator] < mut_operator_cost[operator]:
                # remove pphlo. prefix
                predict_factor[operator_name] = mut_operator_cost[operator]/ori_operator_cost[operator] - 1
        for operator in mut_operator_cost:
            operator_name = operator.split(".")[1]
            if operator_name in operator_pphlo_to_hlo:
                operator_name = operator_pphlo_to_hlo[operator_name]
            if operator not in ori_operator_cost:
                print(f"Operator {operator} not found in ori_operator_cost!!!!!!!!!!!!")
                special_operator_none.append(operator_name)
                continue
            if mut_operator_cost[operator] != 0 and ori_operator_cost[operator] == 0:
                print(f"Operator {operator} has no communication cost in ori_operator_cost!!!!!!!!!!!!")
                # special_operator_nocommu.append(operator_name)
        special_operator_none = list(set(special_operator_none))
        special_operator_nocommu = list(set(special_operator_nocommu))
        print("before dealing with special operator:", predict_factor)
        if len(predict_factor) == 0:
            # a hyperparameter!!!!
            max_predict_factor = 1
            min_predict_factor = 1e2
        else:
            max_predict_factor = max(predict_factor.values())
            min_predict_factor = min(predict_factor.values())
        for operator in special_operator_none:
            predict_factor[operator] = max_predict_factor
        for operator in special_operator_nocommu:
            predict_factor[operator] = min_predict_factor
        print("after dealing with special operator:", predict_factor)
        return predict_factor

class reducer():
    def __init__(self, data, log_path_all, partynum=2, protocalchosen="SEMI2K", pass_option_mut=None, matrix="send_bytes", pattern_window_size=32, inline_call=False, verbose=False):
        self.data = data
        self.log_path_all = log_path_all
        self.pass_option_mut = pass_option_mut
        self.partynum = partynum
        self.protocalchosen = protocalchosen
        self.matrix = matrix
        self.pattern_window_size = pattern_window_size
        self.inline_call = inline_call
        self.verbose = verbose
        # counter for the number of test on reduced code
        self.cur_reduce_index = 0

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _reconstruct_from_indicies(self, indices):
        seq = list(self.IR_sequnce)
        return [seq[i] for i in indices]

    def _interesting_test(self, to_check, record_log_path, trigger_log_path):
        self._log("to_check:", to_check)
        log_name = f"log_{self.cur_reduce_index}"
        self._log("log_name:", log_name)
        self.cur_reduce_index += 1

        cur_record_log_path = os.path.join(record_log_path, log_name)
        os.makedirs(cur_record_log_path, exist_ok=True)

        snippet_instance = snippet(
            self._reconstruct_from_indicies(to_check),
            self.IR_inf,
            self.block_inf,
            self.partynum,
            self.protocalchosen,
            cur_record_log_path,
            self.pass_option_mut,
            self.matrix,
        )
        snippet_instance.code_generator()
        snippet_instance.code_executer()
        eval_result = snippet_instance.evaluate()
        if eval_result:
            shutil.copytree(cur_record_log_path, os.path.join(trigger_log_path, log_name))
        return eval_result

    @staticmethod
    def _compute_cdd_chunk_size(round_count, initial_p, clamp_to_one_on_overflow=False):
        p = initial_p * (1.582 ** round_count)
        if p <= 0:
            raise ValueError("p must be greater than 0.")
        if p >= 1:
            if clamp_to_one_on_overflow:
                print("p is greater than 1, return 1")
                return 1
            raise ValueError("p must be between 0 and 1 exclusive.")

        S_opt = -1 / math.log(1 - p)

        if S_opt < 1:
            candidates = [1]
        else:
            S_floor = math.floor(S_opt)
            S_ceil = math.ceil(S_opt)
            candidates = [S_floor, S_ceil]

        max_value = -float('inf')
        best_S = 0

        for S in candidates:
            current_value = S * ((1 - p) ** S)
            if current_value > max_value or (current_value == max_value and S > best_S):
                max_value = current_value
                best_S = S

        return best_S
    
    def profile_origin(self):
        fullcode_log_path = os.path.join(self.log_path_all, "fullcode_log")
        if not os.path.exists(fullcode_log_path):
            os.makedirs(fullcode_log_path)
        full_snippet_instance = full_snippet(("\n").join(self.data), self.data[0], self.partynum, self.protocalchosen, fullcode_log_path, self.pass_option_mut, self.matrix)
        full_snippet_instance.code_generator()
        full_snippet_instance.code_executer()
        self.predict_factor = full_snippet_instance.cost_compare()

    def _operator_extract(self, line_list):
        IR_inf = {}
        # dict of values return as tuple from block
        block_tuple_return_dict = {}
        # dict of var from get-tuple-element
        block_tuple_value_dict = {}
        for line in line_list:
            if line.endswith("{") == True or line.startswith("}") == True:
                continue
            line = line.strip()
            output_var_name, statment = line.split(" = ")
            if "), " in statment:
                shape_input_comb, parameters = statment.split("), ")
                shape_input_comb = shape_input_comb + ")"
            else:
                shape_input_comb = statment
                parameters = ""
            shape_input_comb_split = shape_input_comb.split(" ")
            shape = shape_input_comb_split[0]

            # special case where output is a tuple, it should only occur as a call
            # warning: other case may also occur (reduce, sort, etc.)
            if shape[-1] == ",":
                shape = shape_input_comb.split(") ")[0] + ")"
                input_fragment_list = shape_input_comb.split(") ")[1].split(" ")
            else:
                input_fragment_list = shape_input_comb_split[1:]
            if shape == "()":
                continue
            # Currently, iota is treated as constant, which is not verified strictly
            if "constant(" in shape_input_comb_split[1] or "iota(" in shape_input_comb_split[1]:
                var_type = "constant"
            elif "parameter(" in shape_input_comb_split[1]:
                var_type = "parameter"
            else:
                var_type = "operation"

            # TODO: after all the tuple return is processed, remove the tuple inf to avoid redundunt processing
            if output_var_name.find("get-tuple-element") != -1:
                tuple_var_name = input_fragment_list[0].split("(")[-1].split(")")[0]
                if tuple_var_name in block_tuple_return_dict:
                    index_tuple = parameters.split("=")[-1]
                    block_tuple_value_dict[output_var_name] = block_tuple_return_dict[tuple_var_name][int(index_tuple)]
                    continue

            input_var_list = []
            for input_fragment in input_fragment_list:
                if "(" in input_fragment:
                    input_fragment = input_fragment.split("(")[1]
                if ")" in input_fragment:
                    input_fragment = input_fragment.split(")")[0]
                if "," in input_fragment and var_type == "operation":
                    input_fragment = input_fragment.split(",")[0]
                
                if input_fragment in block_tuple_value_dict:
                    input_fragment_replaced = block_tuple_value_dict[input_fragment]
                    line = line.replace(input_fragment, input_fragment_replaced)
                    input_fragment = input_fragment_replaced
                input_var_list.append(input_fragment)
                if var_type == "operation":
                    # When operation is call and the return is tuple, do not update the tobe_input_list
                    if output_var_name.find("call") != -1 and shape[0] == "(":
                        pass
                    else:
                        IR_inf[input_fragment]["tobe_input_list"].append(output_var_name)
            if self.inline_call == True and output_var_name.find("call") != -1:
                call_block_name = parameters.split("to_apply=")[-1]
                IR_inf_block = self._operator_extract(self.block_inf[call_block_name][0])
                for block_output_var_name in IR_inf_block:
                    line_inf_block = IR_inf_block[block_output_var_name]
                    # directly assign the return value of the block to the call variable
                    if "ROOT" in block_output_var_name:
                        # deal with tuple return
                        if shape[0] == "(":
                            block_tuple_return_dict[output_var_name] = line_inf_block["input_var_list"]
                            for input_var in line_inf_block["input_var_list"]:
                                IR_inf[input_var]["tobe_input_list"].remove(block_output_var_name)
                            continue
                        else:
                            line_inf_block["line"] = line_inf_block["line"].replace(block_output_var_name, output_var_name)
                            IR_inf[output_var_name] = line_inf_block
                            for input_var in line_inf_block["input_var_list"]:
                                IR_inf[input_var]["tobe_input_list"][:] = [input if input != block_output_var_name else output_var_name for input in IR_inf[input_var]["tobe_input_list"]]
                            continue
                    
                    # if the block_output_var_name is a parameter, replace it with the input_var for block
                    if line_inf_block["var_type"] == "parameter":
                        parameter_var_name =  input_var_list[int(line_inf_block['input_var_list'][0])]
                        for tobe_inplaced_operator in line_inf_block['tobe_input_list']:
                            IR_inf_block[tobe_inplaced_operator]['input_var_list'][:] = [input if input != block_output_var_name else parameter_var_name for input in IR_inf_block[tobe_inplaced_operator]['input_var_list']]
                            IR_inf_block[tobe_inplaced_operator]['line'] = IR_inf_block[tobe_inplaced_operator]['line'].replace(block_output_var_name, parameter_var_name)
                            IR_inf[parameter_var_name]["tobe_input_list"].append(tobe_inplaced_operator)
                        # remove the parameter from the block
                        continue
                    IR_inf[block_output_var_name] = line_inf_block
                continue
            IR_inf[output_var_name] = {"shape": shape, "var_type": var_type, "input_var_list": input_var_list, "tobe_input_list": [], "parameters": parameters, "line": line}
        return IR_inf
    
    def operator_extract(self):
        self.IR_inf = self._operator_extract(self.block_inf["ENTRY"][0])
        IR_sequnce = []
        for operator in list(self.IR_inf.keys())[::-1]:
            # dead code elimination
            # eliminate the variable that is not used in the later code
            
            if operator.startswith("ROOT") != True and self.IR_inf[operator]["tobe_input_list"] == []:
                for input_var in self.IR_inf[operator]["input_var_list"]:
                    tobe_remove_list = [(input_var, operator)]
                    while tobe_remove_list != []:
                        target_element, remove_element = tobe_remove_list.pop()
                        if self.IR_inf[remove_element]["var_type"] == "constant":
                            continue
                        if target_element in self.IR_inf[remove_element]["tobe_input_list"]:
                            self.IR_inf[remove_element]["tobe_input_list"].remove(target_element)
                        if self.IR_inf[remove_element]["tobe_input_list"] == []:
                            tobe_remove_list += [(remove_element, input_var) for input_var in self.IR_inf[remove_element]["input_var_list"]]
                continue
            IR_sequnce.append(operator)
        self.IR_sequnce = IR_sequnce[::-1]
    
    def IR_extract(self):
        # extract the blocks and the sub-blocks needed
        self.block_inf = {}
        for line in self.data:
            if line == "" or line.startswith("HloModule") == True:
                continue
            if line.startswith(" ") == False and line != "" and line.startswith("HloModule") == False:
                if line.startswith("}"):
                    block_data.append(line)
                    self.block_inf[block_name] = (block_data, subblock_need_list)
                else:
                    block_name = line.split(" ")[0]
                    block_data = [line]
                    subblock_need_list = []
            else:
                # dealing with the case that blocks needs sub-blocks
                if line.find("to_apply=") != -1:
                    block_need = line.split("to_apply=")[-1]
                    subblock_need_list.append(block_need)
                block_data.append(line)
        self.operator_extract()

        """Uncomment the following code to test the processed unreduced code"""
        # record_log_path = os.path.join(self.log_path_all, "record_log")
        # if not os.path.exists(record_log_path):
        #     os.makedirs(record_log_path)
        # log_name = f"log_{self.cur_reduce_index}"
        # cur_record_log_path = os.path.join(record_log_path, log_name)
        # if not os.path.exists(cur_record_log_path):
        #     os.makedirs(cur_record_log_path)
        # snippet_instance = snippet(self.IR_sequnce, self.IR_inf, self.block_inf, self.partynum, self.protocalchosen, cur_record_log_path, self.pass_option_mut, self.matrix)
        # snippet_instance.code_generator()
        # snippet_instance.code_executer()
        # eval_result = snippet_instance.evaluate()

    def pattern_predict(self):
        if not hasattr(self, "predict_factor"):
            raise ValueError("Error: predict_factor is not generated yet!!!!!!!!!!!!")
        if not hasattr(self, "IR_inf"):
            raise ValueError("Error: IR_inf is not generated yet!!!!!!!!!!!!")
        pattern_dict = {}
        pattern_window_operator_list = []
        pattern_window_value = 0
        pattern_window_value_list = []
        for i, operator_inf in enumerate(self.IR_sequnce):
            if "ROOT" in operator_inf:
                operator = operator_inf.split(" ")[1].split(".")[0]
            else:
                operator = operator_inf.split(".")[0]
            if operator in self.predict_factor:
                operator_predict_factor = self.predict_factor[operator]
            else:
                operator_predict_factor = 0
            if len(pattern_window_operator_list) < self.pattern_window_size:
                pattern_window_operator_list.append(i)
                pattern_window_value += operator_predict_factor
                pattern_window_value_list.append(operator_predict_factor)
            if len(pattern_window_operator_list) == self.pattern_window_size:
                pattern_dict[pattern_window_operator_list[0]] = pattern_window_value
                pattern_window_operator_list.pop(0)
                pattern_window_operator_list.append(i)
                pattern_window_value -= pattern_window_value_list[0]
                pattern_window_value_list.pop(0)
                pattern_window_value += operator_predict_factor
                pattern_window_value_list.append(operator_predict_factor)
        self.pattern_dict = dict(sorted(pattern_dict.items(), key=lambda item: item[1], reverse=True))
    
    def reduce(self, mod="heur", onlycomplement=False, initial_p = 0.1):
        if mod not in ["heur", "ddmin", "CDD", "ProbDD", "heurmin", "heurmin_remove", "heurmin-avg", "ddmin_C", "heurmin_C"]:
            raise ValueError("Error: mod should be in [heur, ddmin, ProbDD, CDD, heurmin, heurmin-avg, ddmin_C]!!!!!!!!!!!!")
        if not hasattr(self, "IR_inf"):
            raise ValueError("Error: IR_inf is not generated yet!!!!!!!!!!!!")
        if mod == "heur" and not hasattr(self, "pattern_dict"):
            raise ValueError("Error: pattern_dict is not generated yet!!!!!!!!!!!!")
        record_log_path = os.path.join(self.log_path_all, "record_log")
        if not os.path.exists(record_log_path):
            os.makedirs(record_log_path)
        trigger_log_path = os.path.join(self.log_path_all, "trigger_log")
        if not os.path.exists(trigger_log_path):
            os.makedirs(trigger_log_path)
        
        if mod == "heur":
            for pattern_index in self.pattern_dict:
                log_name = f"log_{self.cur_reduce_index}"
                self.cur_reduce_index += 1
                cur_record_log_path = os.path.join(record_log_path, log_name)
                if not os.path.exists(cur_record_log_path):
                    os.makedirs(cur_record_log_path)
                cut_start_index = pattern_index
                cut_end_index = cut_start_index + self.pattern_window_size - 1
                snippet_instance = snippet(list(self.IR_sequnce)[cut_start_index : cut_end_index], self.IR_inf, self.block_inf, self.partynum, self.protocalchosen, cur_record_log_path, self.pass_option_mut, self.matrix)
                snippet_instance.code_generator()
                snippet_instance.code_executer()
                eval_result = snippet_instance.evaluate()
                if eval_result:
                    shutil.copytree(cur_record_log_path, os.path.join(trigger_log_path, log_name))
                    break
                
            self.delta_debug([i for i in range(cut_start_index, cut_end_index + 1)], record_log_path, trigger_log_path, onlycomplement=onlycomplement)
        elif mod == "ddmin":
            self.delta_debug([i for i in range(len(self.IR_sequnce))], record_log_path, trigger_log_path, onlycomplement=onlycomplement)
        elif mod == "ProbDD":
            self.ProbDD([i for i in range(len(self.IR_sequnce))], record_log_path, trigger_log_path, initial_p=initial_p)
        elif mod == "CDD":
            self.CDD([i for i in range(len(self.IR_sequnce))], record_log_path, trigger_log_path, initial_p=initial_p)
        elif mod == "heurmin":
            self.heurmin([i for i in range(len(self.IR_sequnce))], record_log_path, trigger_log_path)
        elif mod == "heurmin_remove":
            self.heurmin_remove([i for i in range(len(self.IR_sequnce))], record_log_path, trigger_log_path)
        elif mod == "heurmin-avg":
            self.heurmin_avg([i for i in range(len(self.IR_sequnce))], record_log_path, trigger_log_path)
        elif mod == "ddmin_C":
            self.ddmin_C([i for i in range(len(self.IR_sequnce))], record_log_path, trigger_log_path, onlycomplement=onlycomplement, initial_p=initial_p)
        elif mod == "heurmin_C":
            self.heurmin_C([i for i in range(len(self.IR_sequnce))], record_log_path, trigger_log_path, initial_p=initial_p)
        else:
            raise ValueError("Error: mod should be in [heur, ddmin, ProbDD, CDD, heurmin, heurmin-avg, ddmin_C]!!!!!!!!!!!!")
        
    
    # https://github.com/andrewchambers/ddmin-python/blob/master/ddmin.py#L8
    def delta_debug(
        self,
        interesting_indicies,
        record_log_path,
        trigger_log_path,
        granularity = 2,
        onlycomplement=False,
    ):
        test_cache = []

        round_count = 0
        while len(interesting_indicies) > 1:
            chunk_size = (len(interesting_indicies) + granularity - 1) // granularity
            self._log("round:", round_count)
            self._log(len(interesting_indicies))
            self._log(granularity)
            self._log(chunk_size)
            subsets = [
                interesting_indicies[i : i + chunk_size]
                for i in range(0, len(interesting_indicies), chunk_size)
            ]
            temp_interesting_indicies = interesting_indicies
            some_subset_is_interesting = False

            # if only test on the complement of subset, skip the test on subset
            if onlycomplement == False:
                for subset in subsets:
                    test_cache_key = ",".join([str(i) for i in subset])
                    if test_cache_key in test_cache:
                        self._log(f"{test_cache_key} in test_cache")
                        continue
                    if self._interesting_test(subset, record_log_path, trigger_log_path):
                        temp_interesting_indicies = subset
                        some_subset_is_interesting = True
                        break
                    else:
                        test_cache.append(",".join([str(i) for i in subset]))

            if not some_subset_is_interesting:
                for subset in subsets:
                    complement = sorted(set(interesting_indicies) - set(subset))
                    test_cache_key = ",".join([str(i) for i in complement])
                    if test_cache_key in test_cache:
                        self._log(f"{test_cache_key} in test_cache")
                        continue
                    if self._interesting_test(complement, record_log_path, trigger_log_path):
                        temp_interesting_indicies = complement
                        some_subset_is_interesting = True
                        break
                    else:
                        test_cache.append(",".join([str(i) for i in complement]))

            if some_subset_is_interesting:
                interesting_indicies = temp_interesting_indicies
                granularity = max(2, granularity - 1)
            else:
                if granularity == len(interesting_indicies):
                    break
                granularity = min(len(interesting_indicies), granularity * 2)
            round_count += 1

        return self._reconstruct_from_indicies(interesting_indicies)

    def heurmin(
        self,
        interesting_indicies,
        record_log_path,
        trigger_log_path,
        granularity = 2,
        onlycomplement=False,
    ):
        def _sort_pool(pool):
            for test_cache_key in pool.keys():
                value_predict = 0
                sequnce_to_check = self._reconstruct_from_indicies([int(x) for x in test_cache_key.split(",")])
                for operator in sequnce_to_check:
                    # do not use the operator name to judge the cost,
                    # since the operator name does not correspond to the oriniginal operator
                    for operator_name in self.predict_factor:
                        if " " + operator_name + "(" in self.IR_inf[operator]["line"]:
                            value_predict += self.predict_factor[operator_name]
                            break
                pool[test_cache_key] = value_predict
            return dict(sorted(pool.items(), key=lambda item: item[1], reverse=True))
        
        test_cache = []

        round_count = 0
        while len(interesting_indicies) > 1:
            chunk_size = (len(interesting_indicies) + granularity - 1) // granularity
            self._log("round:", round_count)
            self._log(len(interesting_indicies))
            self._log(granularity)
            self._log(chunk_size)
            subsets = [
                interesting_indicies[i : i + chunk_size]
                for i in range(0, len(interesting_indicies), chunk_size)
            ]
            temp_interesting_indicies = interesting_indicies
            some_subset_is_interesting = False

            complement_pool = {}
            for subset in subsets:
                complement = sorted(set(interesting_indicies) - set(subset))
                test_cache_key = ",".join([str(i) for i in complement])
                complement_pool[test_cache_key] = 0
            complement_pool = _sort_pool(complement_pool)
            for test_cache_key in complement_pool:
                complement = [int(x) for x in test_cache_key.split(",")]
                if test_cache_key in test_cache:
                    self._log(f"{test_cache_key} in test_cache")
                    continue
                if self._interesting_test(complement, record_log_path, trigger_log_path):
                    temp_interesting_indicies = complement
                    some_subset_is_interesting = True
                    break
                else:
                    test_cache.append(",".join([str(i) for i in complement]))

            if some_subset_is_interesting:
                interesting_indicies = temp_interesting_indicies
                granularity = max(2, granularity - 1)
            else:
                if granularity == len(interesting_indicies):
                    break
                granularity = min(len(interesting_indicies), granularity * 2)
            round_count += 1

        return self._reconstruct_from_indicies(interesting_indicies)
    
    def heurmin_remove(
        self,
        interesting_indicies,
        record_log_path,
        trigger_log_path,
        granularity = 2,
        onlycomplement=False,
    ):
        def _sort_pool(pool):
            for test_cache_key in pool.keys():
                value_predict = 0
                sequnce_to_check = self._reconstruct_from_indicies([int(x) for x in test_cache_key.split(",")])
                for operator in sequnce_to_check:
                    # do not use the operator name to judge the cost,
                    # since the operator name does not correspond to the oriniginal operator
                    for operator_name in self.predict_factor:
                        if " " + operator_name + "(" in self.IR_inf[operator]["line"]:
                            value_predict += self.predict_factor[operator_name]
                            break
                pool[test_cache_key] = value_predict
            return dict(sorted(pool.items(), key=lambda item: item[1]))
        
        test_cache = []

        round_count = 0
        while len(interesting_indicies) > 1:
            chunk_size = (len(interesting_indicies) + granularity - 1) // granularity
            self._log("round:", round_count)
            self._log(len(interesting_indicies))
            self._log(granularity)
            self._log(chunk_size)
            subsets = [
                interesting_indicies[i : i + chunk_size]
                for i in range(0, len(interesting_indicies), chunk_size)
            ]
            temp_interesting_indicies = interesting_indicies
            some_subset_is_interesting = False

            remove_pool = {}
            for subset in subsets:
                test_cache_key = ",".join([str(i) for i in subset])
                remove_pool[test_cache_key] = 0
            remove_pool = _sort_pool(remove_pool)
            for test_cache_key_remove in remove_pool:
                subset = [int(x) for x in test_cache_key_remove.split(",")]
                complement = sorted(set(interesting_indicies) - set(subset))
                test_cache_key = ",".join([str(i) for i in complement])
                if test_cache_key in test_cache:
                    self._log(f"{test_cache_key} in test_cache")
                    continue
                if self._interesting_test(complement, record_log_path, trigger_log_path):
                    temp_interesting_indicies = complement
                    some_subset_is_interesting = True
                    break
                else:
                    test_cache.append(",".join([str(i) for i in complement]))

            if some_subset_is_interesting:
                interesting_indicies = temp_interesting_indicies
                granularity = max(2, granularity - 1)
            else:
                if granularity == len(interesting_indicies):
                    break
                granularity = min(len(interesting_indicies), granularity * 2)
            round_count += 1

        return self._reconstruct_from_indicies(interesting_indicies)

    def heurmin_avg(
        self,
        interesting_indicies,
        record_log_path,
        trigger_log_path,
        granularity = 2,
    ):
        def _sort_pool(pool):
            for test_cache_key in pool.keys():
                value_predict = 0
                sequnce_to_check = self._reconstruct_from_indicies([int(x) for x in test_cache_key.split(",")])
                for operator in sequnce_to_check:
                    # do not use the operator name to judge the cost,
                    # since the operator name does not correspond to the oriniginal operator
                    for operator_name in self.predict_factor:
                        if " " + operator_name + "(" in self.IR_inf[operator]["line"]:
                            value_predict += self.predict_factor[operator_name]
                            break
                pool[test_cache_key] = value_predict/len(sequnce_to_check)
            return dict(sorted(pool.items(), key=lambda item: item[1], reverse=True))
        
        test_cache = []

        round_count = 0
        while len(interesting_indicies) > 1:
            chunk_size = (len(interesting_indicies) + granularity - 1) // granularity
            self._log("round:", round_count)
            self._log(len(interesting_indicies))
            self._log(granularity)
            self._log(chunk_size)
            subsets = [
                interesting_indicies[i : i + chunk_size]
                for i in range(0, len(interesting_indicies), chunk_size)
            ]
            temp_interesting_indicies = interesting_indicies
            some_subset_is_interesting = False

            avg_pool = {}
            for subset in subsets:
                test_cache_key = ",".join([str(i) for i in subset])
                avg_pool[test_cache_key] = 0
                complement = sorted(set(interesting_indicies) - set(subset))
                test_cache_key = ",".join([str(i) for i in complement])
                avg_pool[test_cache_key] = 0
                avg_pool = _sort_pool(avg_pool)
            
            # for test_cache_key in list(avg_pool.keys())[:len(subsets)]:
            for test_cache_key in avg_pool:    
                complement = [int(x) for x in test_cache_key.split(",")]
                if test_cache_key in test_cache:
                    self._log(f"{test_cache_key} in test_cache")
                    continue
                if self._interesting_test(complement, record_log_path, trigger_log_path):
                    temp_interesting_indicies = complement
                    some_subset_is_interesting = True
                    break
                else:
                    test_cache.append(",".join([str(i) for i in complement]))

            if some_subset_is_interesting:
                interesting_indicies = temp_interesting_indicies
                granularity = max(2, granularity - 1)
            else:
                if granularity == len(interesting_indicies):
                    break
                granularity = min(len(interesting_indicies), granularity * 2)
            round_count += 1

        return self._reconstruct_from_indicies(interesting_indicies)
    
    def CDD(
        self,
        interesting_indicies,
        record_log_path,
        trigger_log_path,
        initial_p=0.1,
    ):  
        # test_cache = []

        round_count = 0
        while len(interesting_indicies) > 1:
            chunk_size = self._compute_cdd_chunk_size(round_count, initial_p)
            self._log("round:", round_count)
            self._log(len(interesting_indicies))
            self._log(chunk_size)
            subsets = [
                interesting_indicies[i : i + chunk_size]
                for i in range(0, len(interesting_indicies), chunk_size)
            ]

            for subset in subsets:
                complement = sorted(set(interesting_indicies) - set(subset))
                if complement == []:
                    continue
                # test_cache_key = ",".join([str(i) for i in complement])
                # if test_cache_key in test_cache:
                #     print(f"{test_cache_key} in test_cache")
                #     continue
                if self._interesting_test(complement, record_log_path, trigger_log_path):
                    interesting_indicies = complement
                # else:
                #     test_cache.append(",".join([str(i) for i in complement]))

            if chunk_size <= 1:
                break
            round_count += 1

        return self._reconstruct_from_indicies(interesting_indicies)
    
    def ProbDD(
        self,
        interesting_indicies,
        record_log_path,
        trigger_log_path,
        initial_p=0.1,
    ):  
        def _select_subset(indices):
            # randomize then sort
            sorted_indices = sorted(
                indices,
                key=lambda x: (prob_dict[x], random.random())
            )
            
            best_subset = []
            current_gain = 0.0
            cumulative_product = 1.0
            
            for idx in sorted_indices:
                # compute gain
                temp_subset = best_subset + [idx]
                temp_product = cumulative_product * (1 - prob_dict[idx])
                temp_gain = len(temp_subset) * temp_product
                
                # update the optimal subset
                if temp_gain > current_gain:
                    best_subset = temp_subset
                    current_gain = temp_gain
                    cumulative_product = temp_product
                else:
                    break
                    
            return best_subset
        
        prob_dict = {idx: initial_p for idx in interesting_indicies}
        # test_cache = []

        while True:
            if all(p >= 0.999 for p in prob_dict.values()):
                break
            S = _select_subset(interesting_indicies)
            self._log(len(interesting_indicies))
            self._log(prob_dict)
            self._log(S)
            if not S:
                break
            if len(S) == len(interesting_indicies):
                self._log("terminate because all indicies are selected")
                break

            complement = sorted(set(interesting_indicies) - set(S))
            # test_cache_key = ",".join(map(str, complement))

            # if test_cache_key in test_cache:
            #     print("{test_cache_key} in test_cache")
            #     continue

            if self._interesting_test(complement, record_log_path, trigger_log_path):
                interesting_indicies = complement
                for idx in S:
                    if idx in prob_dict:
                        del prob_dict[idx]
            else:
                product = math.prod([1 - prob_dict[idx] for idx in S])
                
                factor = 1.0 / (1 - product) if product < 0.999 else 1.0
                
                for idx in S:
                    prob_dict[idx] = min(prob_dict[idx] * factor, 1.0)
                    
                # test_cache.append(test_cache_key)
        # del test_cache
        return self._reconstruct_from_indicies(interesting_indicies)

    def ddmin_C(
        self,
        interesting_indicies,
        record_log_path,
        trigger_log_path,
        granularity = 2,
        onlycomplement=False,
        initial_p=0.1,
    ):
        test_cache = []

        round_count = 0
        round_count_cdd = 0
        number_of_rounds_cdd = 0
        swtich_to_cdd = False
        chunk_size_cdd = len(interesting_indicies)
        while len(interesting_indicies) > 1:
            chunk_size = (len(interesting_indicies) + granularity - 1) // granularity
            if self.cur_reduce_index >= number_of_rounds_cdd:
                if chunk_size > chunk_size_cdd:
                    swtich_to_cdd = True
                    break
                self._log("Predicting the cost of CDD")
                chunk_size_cdd = self._compute_cdd_chunk_size(round_count_cdd, initial_p, clamp_to_one_on_overflow=True)
                number_of_rounds_cdd = round_count_cdd + len(interesting_indicies)//chunk_size_cdd + 1
                self._log("chunk_size_cdd:", chunk_size_cdd)
                self._log("number_of_rounds_cdd:", number_of_rounds_cdd)
                round_count_cdd += 1
            self._log("round:", round_count)
            self._log(len(interesting_indicies))
            self._log(granularity)
            self._log(chunk_size)
            subsets = [
                interesting_indicies[i : i + chunk_size]
                for i in range(0, len(interesting_indicies), chunk_size)
            ]
            temp_interesting_indicies = interesting_indicies
            some_subset_is_interesting = False

            # if only test on the complement of subset, skip the test on subset
            if onlycomplement == False:
                for subset in subsets:
                    test_cache_key = ",".join([str(i) for i in subset])
                    if test_cache_key in test_cache:
                        self._log(f"{test_cache_key} in test_cache")
                        continue
                    if self._interesting_test(subset, record_log_path, trigger_log_path):
                        temp_interesting_indicies = subset
                        some_subset_is_interesting = True
                        break
                    else:
                        test_cache.append(",".join([str(i) for i in subset]))

            if not some_subset_is_interesting:
                for subset in subsets:
                    complement = sorted(set(interesting_indicies) - set(subset))
                    test_cache_key = ",".join([str(i) for i in complement])
                    if test_cache_key in test_cache:
                        self._log(f"{test_cache_key} in test_cache")
                        continue
                    if self._interesting_test(complement, record_log_path, trigger_log_path):
                        temp_interesting_indicies = complement
                        some_subset_is_interesting = True
                        break
                    else:
                        test_cache.append(",".join([str(i) for i in complement]))

            if some_subset_is_interesting:
                interesting_indicies = temp_interesting_indicies
                granularity = max(2, granularity - 1)
            else:
                if granularity == len(interesting_indicies):
                    break
                granularity = min(len(interesting_indicies), granularity * 2)
            round_count += 1

        if swtich_to_cdd:
            self._log("switch to CDD!")
            round_count = round_count_cdd - 1
            while len(interesting_indicies) > 1:
                chunk_size = self._compute_cdd_chunk_size(round_count, initial_p, clamp_to_one_on_overflow=True)
                self._log("round:", round_count)
                self._log(len(interesting_indicies))
                self._log(chunk_size)
                subsets = [
                    interesting_indicies[i : i + chunk_size]
                    for i in range(0, len(interesting_indicies), chunk_size)
                ]

                for subset in subsets:
                    complement = sorted(set(interesting_indicies) - set(subset))
                    if complement == []:
                        continue
                    test_cache_key = ",".join([str(i) for i in complement])
                    if test_cache_key in test_cache:
                        self._log(f"{test_cache_key} in test_cache")
                        continue
                    if self._interesting_test(complement, record_log_path, trigger_log_path):
                        interesting_indicies = complement
                    else:
                        test_cache.append(",".join([str(i) for i in complement]))

                if chunk_size <= 1:
                    break
                round_count += 1


        return self._reconstruct_from_indicies(interesting_indicies)

    def heurmin_C(
        self,
        interesting_indicies,
        record_log_path,
        trigger_log_path,
        granularity = 2,
        initial_p=0.1,
    ):
        def _sort_pool(pool):
            for test_cache_key in pool.keys():
                value_predict = 0
                sequnce_to_check = self._reconstruct_from_indicies([int(x) for x in test_cache_key.split(",")])
                for operator in sequnce_to_check:
                    # do not use the operator name to judge the cost,
                    # since the operator name does not correspond to the oriniginal operator
                    for operator_name in self.predict_factor:
                        if " " + operator_name + "(" in self.IR_inf[operator]["line"]:
                            value_predict += self.predict_factor[operator_name]
                            break
                pool[test_cache_key] = value_predict
            return dict(sorted(pool.items(), key=lambda item: item[1], reverse=True))
        
        test_cache = []

        round_count = 0
        round_count_cdd = 0
        number_of_rounds_cdd = 0
        swtich_to_cdd = False
        chunk_size_cdd = len(interesting_indicies)
        while len(interesting_indicies) > 1:
            chunk_size = (len(interesting_indicies) + granularity - 1) // granularity
            if self.cur_reduce_index >= number_of_rounds_cdd:
                if chunk_size > chunk_size_cdd:
                    swtich_to_cdd = True
                    break
                self._log("Predicting the cost of CDD")
                chunk_size_cdd = self._compute_cdd_chunk_size(round_count_cdd, initial_p, clamp_to_one_on_overflow=True)
                number_of_rounds_cdd = round_count_cdd + len(interesting_indicies)//chunk_size_cdd + 1
                self._log("chunk_size_cdd:", chunk_size_cdd)
                self._log("number_of_rounds_cdd:", number_of_rounds_cdd)
                round_count_cdd += 1
            self._log("round:", round_count)
            self._log(len(interesting_indicies))
            self._log(granularity)
            self._log(chunk_size)
            subsets = [
                interesting_indicies[i : i + chunk_size]
                for i in range(0, len(interesting_indicies), chunk_size)
            ]
            temp_interesting_indicies = interesting_indicies
            some_subset_is_interesting = False

            # if only test on the complement of subset, skip the test on subset
            complement_pool = {}
            for subset in subsets:
                complement = sorted(set(interesting_indicies) - set(subset))
                test_cache_key = ",".join([str(i) for i in complement])
                complement_pool[test_cache_key] = 0
            complement_pool = _sort_pool(complement_pool)
            for test_cache_key in complement_pool:
                complement = [int(x) for x in test_cache_key.split(",")]
                if test_cache_key in test_cache:
                    self._log(f"{test_cache_key} in test_cache")
                    continue
                if self._interesting_test(complement, record_log_path, trigger_log_path):
                    temp_interesting_indicies = complement
                    some_subset_is_interesting = True
                    break
                else:
                    test_cache.append(",".join([str(i) for i in complement]))

            if some_subset_is_interesting:
                interesting_indicies = temp_interesting_indicies
                granularity = max(2, granularity - 1)
            else:
                if granularity == len(interesting_indicies):
                    break
                granularity = min(len(interesting_indicies), granularity * 2)
            round_count += 1

        if swtich_to_cdd:
            self._log("switch to CDD!")
            round_count = round_count_cdd - 1
            while len(interesting_indicies) > 1:
                chunk_size = self._compute_cdd_chunk_size(round_count, initial_p, clamp_to_one_on_overflow=True)
                self._log("round:", round_count)
                self._log(len(interesting_indicies))
                self._log(chunk_size)
                subsets = [
                    interesting_indicies[i : i + chunk_size]
                    for i in range(0, len(interesting_indicies), chunk_size)
                ]

                for subset in subsets:
                    complement = sorted(set(interesting_indicies) - set(subset))
                    if complement == []:
                        continue
                    test_cache_key = ",".join([str(i) for i in complement])
                    if test_cache_key in test_cache:
                        self._log(f"{test_cache_key} in test_cache")
                        continue
                    if self._interesting_test(complement, record_log_path, trigger_log_path):
                        interesting_indicies = complement
                    else:
                        test_cache.append(",".join([str(i) for i in complement]))

                if chunk_size <= 1:
                    break
                round_count += 1

        return self._reconstruct_from_indicies(interesting_indicies)

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

# Save the original stdout
original_stdout = sys.stdout

# ddmethod_list = ["ddmin", "onlycomplement", "CDD", "ProbDD"]
# ddmethod_list = ["onlycomplement", "CDD"]
# case_name_list = ["average-sort-semi2k-dd", "chi2-algebra-semi2k-dd", "forest-while-semi2k-dd", "kbinskmeans-algebra-semi2k-dd", "quantile-algebraic-aby3-dd", "rsvd-dimension-cheetah-dd", "rsvd-dotdec-cheetah-dd", "svm-algebra-semi2k-dd", "svm-algebra-semi2k-dd2"]
# case_name_list = ["chi2-algebra-semi2k-dd", "kbinskmeans-algebra-semi2k-dd", "quantile-algebraic-aby3-dd", "rsvd-dimension-cheetah-dd", "rsvd-dotdec-cheetah-dd", "svm-algebra-semi2k-dd", "svm-algebra-semi2k-dd2"]
# ddmethod_list = ["ProbDD"]
# case_name_list = ["ada-while-semi2k-dd", "tree-while-semi2k-dd"]
# ddmethod_list = ["CDD"]
# case_name_list = ["quantile-gather-semi2k-dd"]
# ddmethod_list = ["ddmin_C_OC"]
# case_name_list = ["average-sort-semi2k-dd", "chi2-algebra-semi2k-dd", "kbinskmeans-algebra-semi2k-dd", "rsvd-dimension-cheetah-dd", "rsvd-dotdec-cheetah-dd", "svm-algebra-semi2k-dd2", "quantile-gather-semi2k-dd", "quantile-algebraic-aby3-dd2", "tweedie-algebra-cheetah-dd"]
# case_name_list = ["rsvd-dimension-cheetah-dd", "rsvd-dotdec-cheetah-dd", "svm-algebra-semi2k-dd2", "quantile-gather-semi2k-dd"]
# case_name_list = ["tweedie-algebra-cheetah-dd"]
# ddmethod_list = ["ddmin", "onlycomplement", "CDD", "ProbDD", "heurmin", "heurmin-avg", "ddmin_C", "ddmin_C_OC"]
# ddmethod_list = ["heurmin", "heurmin-avg"]
# ddmethod_list = ["heurmin_C"]
# case_name_list = ["ada-while-semi2k-dd", "forest-while-semi2k-dd", "tree-while-semi2k-dd"]
# ten cases for final test
# case_name_list = ["average-sort-semi2k-dd", "chi2-algebra-semi2k-dd", "kbinskmeans-algebra-semi2k-dd", "svm-algebra-semi2k-dd2", "quantile-gather-semi2k-dd", "quantile-algebraic-aby3-dd2", "tweedie-algebra-cheetah-dd", "ada-while-semi2k-dd", "forest-while-semi2k-dd", "tree-while-semi2k-dd"]
# ddmethod_list = ["heurmin_remove"]

# case_name_list = ["qunatile-callin-semi2k-dd"]
# case_name_list = ["gamma-algebra-cheetah-dd", "labelbin_bin-while-semi2k-dd", "poisson-algebra-cheetah-dd"]
# ddmethod_list = ["heurmin", "heurmin_remove", "ddmin", "onlycomplement"]
# ddmethod_list = ["CDD"]

case_name_list = ["kbinskmeans-algebra-semi2k-dd"]
ddmethod_list = ["heurmin_C"]

for case_name in case_name_list:
    for ddmethod in ddmethod_list:
        # try:
        """set the hyperparameters"""
        # case_name = "rsvd-dotdec-cheetah-dd"
        # ddmethod = "ddmin"
        case_name_path = f"reduce_log/{case_name}"
        if not os.path.exists(case_name_path):
            os.makedirs(case_name_path)
        log_path_all = f"reduce_log/{case_name}/{case_name}-{ddmethod}"

        ### load the test case
        case_dir = f"reduce_case/{case_name}.json"
        if os.path.exists(case_dir):
            with open(case_dir, "r") as f:
                case_dict = json.load(f)
            input_test_case = case_dict["test_case"]
            pass_option_mut = case_dict["pass_option_mut"]
            protocalchosen = case_dict["protocalchosen"]
            matrix = case_dict["matrix"]
        else:
            raise ValueError(f"Error: {case_dir} does not exist!!!!!!!!!!!!")

        # number of parties
        if protocalchosen == "ABY3":
            partynum = 3
        else:
            partynum = 2

        ### start the reducer
        # create the output folder
        if os.path.exists(log_path_all):
            shutil.rmtree(log_path_all)
        os.makedirs(log_path_all)

        """print the output to the log file"""
        sys.stdout = open(os.path.join(log_path_all, 'test-log.txt'), 'w')
        # print out the pass mutation for record
        print("pass_option_mut:", pass_option_mut)
        print("protocalchosen:", protocalchosen)
        print("matrix:", matrix)
        reducer_instance = reducer(input_test_case, log_path_all, partynum=partynum, protocalchosen=protocalchosen, pass_option_mut=pass_option_mut, matrix=matrix, pattern_window_size=32, inline_call=True, verbose=True)
        reducer_instance.profile_origin()
        reducer_instance.IR_extract()
        if ddmethod == "ddmin":
            reducer_instance.reduce(mod="ddmin")
        elif ddmethod == "onlycomplement":
            reducer_instance.reduce(mod="ddmin", onlycomplement=True)
        elif ddmethod == "CDD":
            reducer_instance.reduce(mod="CDD")
        elif ddmethod == "ProbDD":
            reducer_instance.reduce(mod="ProbDD")
        elif ddmethod == "heur":
            reducer_instance.pattern_predict()
            reducer_instance.reduce(mod="heur")
        elif ddmethod == "heurmin":
            reducer_instance.pattern_predict()
            reducer_instance.reduce(mod="heurmin")
        elif ddmethod == "heurmin_remove":
            reducer_instance.pattern_predict()
            reducer_instance.reduce(mod="heurmin_remove")
        elif ddmethod == "heurmin-avg":
            reducer_instance.pattern_predict()
            reducer_instance.reduce(mod="heurmin-avg")
        elif ddmethod == "ddmin_C":
            reducer_instance.reduce(mod="ddmin_C")
        elif ddmethod == "ddmin_C_OC":
            reducer_instance.reduce(mod="ddmin_C", onlycomplement = True)
        elif ddmethod == "heurmin_C":
            reducer_instance.reduce(mod="heurmin_C")
        else:
            raise ValueError("Invalid ddmethod.")
        sys.stdout = original_stdout
        # except:
        #     sys.stdout = original_stdout
        #     print("Error: ", case_name, ddmethod)
        #     continue
