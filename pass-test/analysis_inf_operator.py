from extract_inf import split_log_party, extract_log
import json

split_log_party("tmp-exp/svm-semi/able", 2)
extract_log("tmp-exp/svm-semi/able", pphlo_log = False, operator_profile = True)
split_log_party("tmp-exp/svm-semi/disable", 2)
extract_log("tmp-exp/svm-semi/disable", pphlo_log = False, operator_profile = True)

with open("tmp-exp/svm-semi/able/extract_result.json", "r") as f:
    extract_result_able = json.load(f)

with open("tmp-exp/svm-semi/disable/extract_result.json", "r") as f:
    extract_result_disable = json.load(f)

for operator in extract_result_able["test_svm"]["profile"]["repeat_0"]["party_0"]["operator_profile"]["HLO_profile"].keys():
    if operator not in extract_result_disable["test_svm"]["profile"]["repeat_0"]["party_0"]["operator_profile"]["HLO_profile"]:
        print(f"Operator {operator} is not in disable")
    else:
        profile_disable = extract_result_disable["test_svm"]["profile"]["repeat_0"]["party_0"]["operator_profile"]["HLO_profile"][operator]
        profile_able = extract_result_able["test_svm"]["profile"]["repeat_0"]["party_0"]["operator_profile"]["HLO_profile"][operator]
        if profile_disable["send_bytes"] < profile_able["send_bytes"]:
            print("disable:", profile_disable["send_bytes"], "able:", profile_able["send_bytes"])
            print(f"Operator {operator} has lower send_bytes in disable")
            
        elif profile_disable["send_bytes"] > profile_able["send_bytes"]:
            print("disable:", profile_disable["send_bytes"], "able:", profile_able["send_bytes"])
            print(f"Operator {operator} has lower send_bytes in able")
        else:
            print(f"Operator {operator} has same send_bytes")
        print("")