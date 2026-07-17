import os
import json
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np

"""Define the parameters"""
# define the path of the log files
# log_directory = 'test-log/auto-test-all-SEMI2K'
# optimize_only=False

def analysis_json(log_json):
    for function_test in log_json.keys():
        for pass_option in log_json[function_test].keys():
            total_time = 0
            send_bytes = 0
            recv_bytes = 0
            send_actions = 0
            recv_actions = 0
            for party_num in log_json[function_test][pass_option]["profile"].keys():
                for repeat_num in log_json[function_test][pass_option]["profile"][party_num].keys():
                    total_time += float(log_json[function_test][pass_option]["profile"][party_num][repeat_num]['total_time'][:-1])
                    send_bytes += int(log_json[function_test][pass_option]["profile"][party_num][repeat_num]['send_bytes'])
                    recv_bytes += int(log_json[function_test][pass_option]["profile"][party_num][repeat_num]['recv_bytes'])
                    send_actions += int(log_json[function_test][pass_option]["profile"][party_num][repeat_num]['send_actions'])
                    recv_actions += int(log_json[function_test][pass_option]["profile"][party_num][repeat_num]['recv_actions'])
            sum_info = {
                'total_time': total_time,
                'send_bytes': send_bytes,
                'recv_bytes': recv_bytes,
                'send_actions': send_actions,
                'recv_actions': recv_actions
            }
            log_json[function_test][pass_option]["profile"]["sum_info"] = sum_info

def inf_collect(optimization_dict, pass_closed_list, function_test, baseline_inf, current_inf, optimize_only=True):
    if optimize_only:
        if current_inf['total_time'] < baseline_inf['total_time']:
            # print(f"pass_closed: {pass_closed_list} function_test: {function_test} current_time: {current_inf['total_time']} baseline_time: {baseline_inf['total_time']}")
            optimization_dict['total_time']["|".join(pass_closed_list) + "-" + function_test] = (current_inf['total_time'], baseline_inf['total_time'], current_inf['total_time']/baseline_inf['total_time'])
        if current_inf['send_bytes'] < baseline_inf['send_bytes']:
            # print(f"pass_closed: {pass_closed_list} function_test: {function_test} current_send_bytes: {current_inf['send_bytes']} baseline_send_bytes: {baseline_inf['send_bytes']}")
            optimization_dict['send_bytes']["|".join(pass_closed_list) + "-"  + function_test] = (current_inf['send_bytes'], baseline_inf['send_bytes'], current_inf['send_bytes']/baseline_inf['send_bytes'])
        if current_inf['recv_bytes'] < baseline_inf['recv_bytes']:
            # print(f"pass_closed: {pass_closed_list} function_test: {function_test} current_recv_bytes: {current_inf['recv_bytes']} baseline_recv_bytes: {baseline_inf['recv_bytes']}")
            optimization_dict['recv_bytes']["|".join(pass_closed_list) + "-"  + function_test] = (current_inf['recv_bytes'], baseline_inf['recv_bytes'], current_inf['recv_bytes']/baseline_inf['recv_bytes'])
        if current_inf['send_actions'] < baseline_inf['send_actions']:
            # print(f"pass_closed: {pass_closed_list} function_test: {function_test} current_send_actions: {current_inf['send_actions']} baseline_send_actions: {baseline_inf['send_actions']}")
            optimization_dict['send_actions']["|".join(pass_closed_list) + "-"  + function_test] = (current_inf['send_actions'], baseline_inf['send_actions'], current_inf['send_actions']/baseline_inf['send_actions'])
        if current_inf['recv_actions'] < baseline_inf['recv_actions']:
            # print(f"pass_closed: {pass_closed_list} function_test: {function_test} current_recv_actions: {current_inf['recv_actions']} baseline_recv_actions: {baseline_inf['recv_actions']}")
            optimization_dict['recv_actions']["|".join(pass_closed_list) + "-"  + function_test] = (current_inf['recv_actions'], baseline_inf['recv_actions'], current_inf['recv_actions']/baseline_inf['recv_actions'])
    else:
        if current_inf['total_time'] != baseline_inf['total_time']:
            # print(f"pass_closed: {pass_closed_list} function_test: {function_test} current_time: {current_inf['total_time']} baseline_time: {baseline_inf['total_time']}")
            optimization_dict['total_time']["|".join(pass_closed_list) + "-" + function_test] = (current_inf['total_time'], baseline_inf['total_time'], current_inf['total_time']/baseline_inf['total_time'])
        if current_inf['send_bytes'] != baseline_inf['send_bytes']:
            # print(f"pass_closed: {pass_closed_list} function_test: {function_test} current_send_bytes: {current_inf['send_bytes']} baseline_send_bytes: {baseline_inf['send_bytes']}")
            optimization_dict['send_bytes']["|".join(pass_closed_list) + "-"  + function_test] = (current_inf['send_bytes'], baseline_inf['send_bytes'], current_inf['send_bytes']/baseline_inf['send_bytes'])
        if current_inf['recv_bytes'] != baseline_inf['recv_bytes']:
            # print(f"pass_closed: {pass_closed_list} function_test: {function_test} current_recv_bytes: {current_inf['recv_bytes']} baseline_recv_bytes: {baseline_inf['recv_bytes']}")
            optimization_dict['recv_bytes']["|".join(pass_closed_list) + "-"  + function_test] = (current_inf['recv_bytes'], baseline_inf['recv_bytes'], current_inf['recv_bytes']/baseline_inf['recv_bytes'])
        if current_inf['send_actions'] != baseline_inf['send_actions']:
            # print(f"pass_closed: {pass_closed_list} function_test: {function_test} current_send_actions: {current_inf['send_actions']} baseline_send_actions: {baseline_inf['send_actions']}")
            optimization_dict['send_actions']["|".join(pass_closed_list) + "-"  + function_test] = (current_inf['send_actions'], baseline_inf['send_actions'], current_inf['send_actions']/baseline_inf['send_actions'])
        if current_inf['recv_actions'] != baseline_inf['recv_actions']:
            # print(f"pass_closed: {pass_closed_list} function_test: {function_test} current_recv_actions: {current_inf['recv_actions']} baseline_recv_actions: {baseline_inf['recv_actions']}")
            optimization_dict['recv_actions']["|".join(pass_closed_list) + "-"  + function_test] = (current_inf['recv_actions'], baseline_inf['recv_actions'], current_inf['recv_actions']/baseline_inf['recv_actions'])

def communication_costs_analysis(log_directory, optimize_only=False, get_path=False):
    # set the second test flag
    second_test = False
    log_directory_second = log_directory + "-second"

    optimization_dict = {
        'total_time': {},
        'send_bytes': {},
        'recv_bytes': {},
        'send_actions': {},
        'recv_actions': {}
    }
    for log_sub_directory in glob.glob(os.path.join(log_directory, "*/")):
        # if log_sub_directory.find("preprocessing_test") != -1:
        #     print("preprocessing_test is skipped!!!!!!!!!!!!!!!!!!!!")
        #     continue
        with open(os.path.join(log_sub_directory, 'extract_result.json'), 'r') as file:
            extract_result = json.load(file)

        analysis_json(extract_result)

        for function_test in extract_result.keys():
            baseline_inf = extract_result[function_test]["baseline"]["profile"]["sum_info"]
            for pass_option in extract_result[function_test].keys():
                if pass_option == "baseline":
                    continue
                pass_closed_list = [pass_option]
                current_inf = extract_result[function_test][pass_option]["profile"]["sum_info"]
                inf_collect(optimization_dict, pass_closed_list, function_test, baseline_inf, current_inf, optimize_only=optimize_only)

                # if one pass option has deduplicated not because of order, it should also be recorded for checking its effect
                if pass_option.startswith("able"):
                    for pass_deduplicated in extract_result[function_test][pass_option]["deduplication"]:
                        if pass_deduplicated.split("_")[1] != pass_option.split("_")[1]:
                            inf_collect(optimization_dict, [pass_deduplicated], function_test, baseline_inf, current_inf, optimize_only=optimize_only)

        if get_path:
            for metric in optimization_dict.keys():
                for case in optimization_dict[metric].keys():
                    optimization_dict[metric][case] =  optimization_dict[metric][case] + (log_sub_directory,)
                
        if second_test:
            for second_test_directory in glob.glob(log_sub_directory.replace(log_directory, log_directory_second)[:-1] + "|*"):
                first_pass_option = second_test_directory.split("|")[1]
                with open(os.path.join(second_test_directory, 'extract_result.json'), 'r') as file:
                    extract_result_seoncd = json.load(file)

                analysis_json(extract_result_seoncd)

                for function_test in extract_result.keys():
                    baseline_inf = extract_result[function_test]["baseline"]["profile"]["sum_info"]
                    for pass_option in extract_result[function_test].keys():
                        if pass_option == "baseline":
                            continue
                        pass_closed_list = [first_pass_option, pass_option]
                        current_inf = extract_result[function_test][pass_option]["profile"]["sum_info"]

                        inf_collect(optimization_dict, pass_closed_list, function_test, baseline_inf, current_inf, optimize_only=optimize_only)

    if second_test:
        sorted_send_bytes = sorted(optimization_dict['send_bytes'].items(), key=lambda item: item[1][2])
        for case in sorted_send_bytes:
            if "|" in case[0]:
                result_seoncd = case[1][2]
                pass_options = case[0].split("-")[0].split("|")
                first_test_one = "-".join([pass_options[0], case[0].split("-")[1]])
                first_test_two = "-".join([pass_options[1], case[0].split("-")[1]])
                if first_test_one in optimization_dict['send_bytes']:
                    result_first_one = optimization_dict['send_bytes'][first_test_one][2]
                else:
                    # print(f"case: {case[0]} result: {case[1]} first_test_one: {first_test_one} not found")
                    if first_test_two in optimization_dict['send_bytes']:
                        result_first_two = optimization_dict['send_bytes'][first_test_two][2]
                        if result_seoncd < result_first_two:
                            print(f"case: {case[0]} result: {case[1]} first_test_two: {first_test_two} result: {result_first_two}")
                    continue
                if first_test_two in optimization_dict['send_bytes']:
                    result_first_two = optimization_dict['send_bytes'][first_test_two][2]
                else:
                    # print(f"case: {case[0]} result: {case[1]} first_test_two: {first_test_two} not found")
                    if first_test_one in optimization_dict['send_bytes']:
                        result_first_one = optimization_dict['send_bytes'][first_test_one][2]
                        if result_seoncd < result_first_one:
                            print(f"case: {case[0]} result: {case[1]} first_test_one: {first_test_one} result: {result_first_one}")
                    continue
                if result_seoncd < result_first_one and result_seoncd < result_first_two:
                    print(f"case: {case[0]} result: {case[1]} first_test_one: {first_test_one} result: {result_first_one} first_test_two: {first_test_two} result: {result_first_two}")
    # with open(os.path.join(log_directory, "sorted_send_bytes_all.pkl"), "wb") as fp:
    #     pickle.dump(sorted_send_bytes, fp)

    # sorted_time = sorted(optimization_dict['total_time'].items(), key=lambda item: item[1][2])
    # print(sorted_time[:50])

    sorted_send_actions = sorted(optimization_dict['send_actions'].items(), key=lambda item: item[1][2])
    # print(sorted_send_actions)
    # with open(os.path.join(log_directory, "sorted_send_actions_all.pkl"), "wb") as fp:
    #     pickle.dump(sorted_send_actions, fp)
    sorted_send_bytes = sorted(optimization_dict['send_bytes'].items(), key=lambda item: item[1][2])
    # testcase_set = set()
    # disable_algebraicsimplifier_list = []
    # for case in sorted_send_actions:
    #     testcase = case[0].split("-")[1]
    #     pass_option = case[0].split("-")[0]
    #     if pass_option.startswith("disable"):
    #         testcase_set.add(testcase)
    #     if pass_option == "disable_algebraicsimplifier":
    #         disable_algebraicsimplifier_list.append(case)
    # print(len(testcase_set))
    # for case in sorted_send_bytes:
    #     testcase = case[0].split("-")[1]
    #     pass_option = case[0].split("-")[0]
    #     if pass_option.startswith("disable"):
    #         testcase_set.add(testcase)
    #     if pass_option == "disable_algebraicsimplifier":
    #         disable_algebraicsimplifier_list.append(case)
    # print(len(testcase_set))
    # print(disable_algebraicsimplifier_list)
    # for case in disable_algebraicsimplifier_list:
    #     if "disable_algebraicsimplifier-d2_tweedie_score_weight[0.5 1.  2.  0.5]_power0" in case[0]:
    #         print(case)

    return sorted_send_actions, sorted_send_bytes

def get_single_figure(data_list, figure_name):
    pass_effect_dict = {}
    for case in data_list:
        testcase = case[0].split("-")[1]
        pass_option = case[0].split("-")[0]
        effect = case[1][2]
        if pass_option not in pass_effect_dict:
            pass_effect_dict[pass_option] = [(testcase, effect)]
        else:
            pass_effect_dict[pass_option].append((testcase, effect))

    # 1. Process the Data
    # Initialize lists to store processed data
    pass_names = []
    min_effects = []
    max_effects = []
    effect_ranges = []
    has_effect_below_one = []  # Flag to indicate if pass has any effect <1

    for pass_name, test_cases in pass_effect_dict.items():
        # Extract effect values
        effects = [effect for _, effect in test_cases]
        
        # Calculate min and max effects
        min_effect = min(effects)
        max_effect = max(effects)
        
        # Calculate effect range
        effect_range = max_effect - min_effect
        
        # Determine if any effect <1
        flag = min_effect < 1
        
        # Append to lists
        pass_names.append(pass_name)
        min_effects.append(min_effect)
        max_effects.append(max_effect)
        effect_ranges.append(effect_range)
        has_effect_below_one.append(flag)

    # Convert lists to NumPy arrays for easier manipulation
    pass_names = np.array(pass_names)
    min_effects = np.array(min_effects)
    max_effects = np.array(max_effects)
    effect_ranges = np.array(effect_ranges)
    has_effect_below_one = np.array(has_effect_below_one)

    # 2. Sort the Passes
    # First, separate passes with any effect <1 and those without
    mask = has_effect_below_one

    # Sort passes with effect <1 by effect_range descending
    sorted_indices_with_below_one = np.argsort(effect_ranges[mask])[::-1]

    # Sort passes without any effect <1 by effect_range descending
    sorted_indices_without_below_one = np.argsort(effect_ranges[~mask])[::-1]

    # Combine the sorted indices
    sorted_indices = np.concatenate((np.where(mask)[0][sorted_indices_with_below_one],
                                    np.where(~mask)[0][sorted_indices_without_below_one]))

    # Apply sorted indices to all arrays
    pass_names_sorted = pass_names[sorted_indices]
    min_effects_sorted = min_effects[sorted_indices]
    max_effects_sorted = max_effects[sorted_indices]
    effect_ranges_sorted = effect_ranges[sorted_indices]
    has_effect_below_one_sorted = has_effect_below_one[sorted_indices]

    # 3. Calculate Relative Effects
    # Relative effects: deviation from 1
    # Negative for min_effect <1, zero otherwise
    relative_min = np.where(min_effects_sorted < 1, 1 - min_effects_sorted, 0)
    # Positive for max_effect >1, zero otherwise
    relative_max = np.where(max_effects_sorted > 1, max_effects_sorted - 1, 0)

    # 4. Create the Diverging Bar Chart
    plt.figure(figsize=(14, 10))

    # Positions along the y-axis
    y_positions = np.arange(len(pass_names_sorted))

    # Plot bars for Best Effects (≤1) as negative bars
    bars_min = plt.barh(y_positions, -relative_min, height=0.4, color='steelblue', label='Best Effect (≤1)')

    # Plot bars for Worst Effects (≥1) as positive bars
    bars_max = plt.barh(y_positions, relative_max, height=0.4, color='indianred', label='Worst Effect (≥1)')

    # Add a vertical line at x=0 for the neutral effect
    plt.axvline(x=0, color='grey', linewidth=1)

    # Set y-axis labels with proper formatting
    plt.yticks(y_positions, [pass_name.replace('_', ' ').title() for pass_name in pass_names_sorted], fontsize=10)

    # Labeling
    plt.xlabel('Relative Effect Compared to 1', fontsize=12)
    plt.title('Effect of Each Pass on Test Cases', fontsize=16)

    # # # Set x-axis limits to 0 to +2
    # plt.xlim(-0.5, 1)
    plt.axis([-1, 1, -1, len(pass_names_sorted) + 1])

    # Add legend
    plt.legend(loc='upper right')

    # Add annotations for Best Effects
    for idx, bar in enumerate(bars_min):
        if relative_min[idx] > 0:
            plt.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2,
                    f"{min_effects_sorted[idx]:.2f}",
                    ha='right', va='center', color='white', fontsize=8)

    # Add annotations for Worst Effects
    for idx, bar in enumerate(bars_max):
        if relative_max[idx] > 0:
            plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{max_effects_sorted[idx]:.2f}",
                    ha='left', va='center', color='black', fontsize=8)

    # Optional: Add grid lines for easier reading
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Improve layout
    plt.tight_layout()

    # 5. Save the Plot as an Image
    output_filename = figure_name  # You can change the filename and format
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    # print(f"Plot saved as '{output_filename}' in the current directory.")
    plt.close()

def get_figure(actions_list, bytes_list, protocal, filter):
    if filter == "disable":
        actions_list_filtered = []
        bytes_list_filtered = []
        for case in actions_list:
            if case[0].startswith("disable"):
                actions_list_filtered.append(case)
        for case in bytes_list:
            if case[0].startswith("disable"):
                bytes_list_filtered.append(case)
        get_single_figure(actions_list_filtered, f"figure/pass-effect-disable-{protocal}-actions.png")
        get_single_figure(bytes_list_filtered, f"figure/pass-effect-disable-{protocal}-bytes.png")
    elif filter == "able":
        actions_list_filtered = []
        bytes_list_filtered = []
        for case in actions_list:
            if case[0].startswith("able"):
                actions_list_filtered.append(case)
        for case in bytes_list:
            if case[0].startswith("able"):
                bytes_list_filtered.append(case)
        get_single_figure(actions_list_filtered, f"figure/pass-effect-enable-{protocal}-actions.png")
        get_single_figure(bytes_list_filtered, f"figure/pass-effect-enable-{protocal}-bytes.png")
    elif filter == "algebra":
        get_single_figure(actions_list, f"figure/pass-effect-algebra-{protocal}-actions.png")
        get_single_figure(bytes_list, f"figure/pass-effect-algebra-{protocal}-bytes.png")
    else:
        get_single_figure(actions_list, f"figure/pass-effect-all-{protocal}-actions.png")
        get_single_figure(bytes_list, f"figure/pass-effect-all-{protocal}-bytes.png")
    
    
if __name__ == "__main__":
    optimize_only = False
    # actions_aby3, bytes_aby3 = communication_costs_analysis('test-log/auto-test-all', optimize_only=optimize_only)
    # actions_semi2k, bytes_semi2k = communication_costs_analysis('test-log/auto-test-all-SEMI2K', optimize_only=optimize_only)
    # actions_cheetah, bytes_cheetah = communication_costs_analysis('test-log/auto-test-all-CHEETAH', optimize_only=optimize_only)
    # actions_aby3_algebra, bytes_aby3_algebra = communication_costs_analysis('test-log/auto-test-algebra', optimize_only=optimize_only)
    # actions_cheetah_algebra, bytes_cheetah_algebra = communication_costs_analysis('test-log/auto-test-algebra-CHEETAH', optimize_only=optimize_only)
    # actions_semi2k_algebra, bytes_semi2k_algebra = communication_costs_analysis('test-log/auto-test-algebra-SEMI2K', optimize_only=optimize_only)

    # get_figure(actions_aby3, bytes_aby3, "ABY3", "disable")
    # get_figure(actions_semi2k, bytes_semi2k, "SEMI2K", "disable")
    # get_figure(actions_cheetah, bytes_cheetah, "CHEETAH", "disable")

    # get_figure(actions_aby3, bytes_aby3, "ABY3", "able")
    # get_figure(actions_semi2k, bytes_semi2k, "SEMI2K", "able")
    # get_figure(actions_cheetah, bytes_cheetah, "CHEETAH", "able")

    # get_figure(actions_aby3, bytes_aby3, "ABY3", "all")
    # get_figure(actions_semi2k, bytes_semi2k, "SEMI2K", "all")
    # get_figure(actions_cheetah, bytes_cheetah, "CHEETAH", "all")

    # get_figure(actions_aby3_algebra, bytes_aby3_algebra, "ABY3", "algebra")
    # get_figure(actions_cheetah_algebra, bytes_cheetah_algebra, "CHEETAH", "algebra")
    # get_figure(actions_semi2k_algebra, bytes_semi2k_algebra, "SEMI2K", "algebra")


    # for case in actions_aby3:
    #     if "test_normalizer_l2" in case[0]:
    #         print(case)
    # for case in bytes_aby3:
    #     if "test_normalizer_l2" in case[0]:
    #         print(case)

    # for case in actions_aby3_algebra:
    #     if "test_normalizer_l2" in case[0]:
    #         print(case)
    # for case in bytes_aby3_algebra:
    #     if "test_normalizer_l2" in case[0]:
    #         print(case)
    # print(actions_aby3_algebra)
    # print(bytes_aby3_algebra)

        # output = communication_costs_analysis('test-log/auto-test-all-CHEETAH-r1', optimize_only=optimize_only)

    # actions_semi2k_algebra_more, bytes_semi2k_algebra_more = communication_costs_analysis('test-log/SEMI2K-algebra-more', optimize_only=optimize_only)
    # print(actions_semi2k_algebra_more)
    # print(bytes_semi2k_algebra_more)

    # actions_cheetah_algebra_more, bytes_cheetah_algebra_more = communication_costs_analysis('test-log/CHEETAH-pow-exp', optimize_only=optimize_only)
    # print(actions_cheetah_algebra_more)
    # print(bytes_cheetah_algebra_more)

    # actions_semi2k_algebra_more, bytes_semi2k_algebra_more = communication_costs_analysis('test-log/SEMI2K-ds-reshape-exp', optimize_only=optimize_only)
    # print(actions_semi2k_algebra_more)
    # print(bytes_semi2k_algebra_more)
