import os
import json
import glob
import pickle

# log_directorys = ['test-log/preprocessing-add', 'test-log/sml-add', 'test-log/sml-more-add']
# log_directorys = ['test-log/precision-aby-add']
# log_directorys = ['test-log/preprocessing-add', 'test-log/sml-add', 'test-log/sml-more-add']
# log_directorys = ['test-log/sml-semi']
# log_directorys = ['test-log/sml-cheetah']
# log_directorys = ['test-log/kmeans-random-cheetah']
log_directorys = ['test-log/sml-cheetah']

sorted_total_send_bytes = []
sorted_send_actions = []

for log_directory in log_directorys:
    with open(os.path.join(log_directory, "sorted_total_send_bytes_all.pkl"), "rb") as fp:
        sorted_total_send_bytes = sorted_total_send_bytes + pickle.load(fp)

    with open(os.path.join(log_directory, "sorted_send_actions_all.pkl"), "rb") as fp:
        sorted_send_actions = sorted_send_actions + pickle.load(fp)

sorted_total_send_bytes = sorted(sorted_total_send_bytes, key=lambda item: item[1][2])
sorted_send_actions = sorted(sorted_send_actions, key=lambda item: item[1][2])
# print([x for x in sorted_total_send_bytes if x[0].find("able_stablesortexpander") != -1])
# print([x for x in sorted_total_send_bytes if x[1][-1] < 1 and x[0].startswith("disable")])
# print([x for x in sorted_total_send_bytes if x[0].find("disable_callinliner_test_quantile") != -1])
# print([x for x in sorted_total_send_bytes if x[0].find("able_stablesortexpander") != -1 and x[0].find("test_knn_distance") != -1 ])
# print([x for x in sorted_total_send_bytes if x[0].find("disable_whileloopsimplifier_test_labelbinarizer_binary") != -1])
# print([x for x in sorted_total_send_bytes if x[0].find("able_stablesortexpander_0") != -1 and x[1][-1] < 1 and not x[0].startswith("disable")])
# print([x for x in sorted_send_actions if x[0].find("able_stablesortexpander_0_test_kbinsdiscretizer_quantile_diverse_n_bins_no_vectorize") != -1])

# print([x for x in sorted_total_send_bytes if x[0].startswith("able") and x[0].find("able_stablesortexpander") == -1 and x[0].find("able_scattersimplifier") == -1])
# print([x for x in sorted_send_actions if x[0].startswith("able") and x[0].find("able_scattersimplifier") != -1 and x[0].find("test_tree") != -1])
# print([x for x in sorted_send_actions if x[0].find("scatterexpander") != -1 and x[0].find("test_tree") != -1])
# print([x for x in sorted_total_send_bytes if x[0].startswith("disable")])
# print([x for x in sorted_send_actions if x[0].find("disable_whileloopsimplifier_test_labelbinarizer_binary") != -1])
# print([x for x in sorted_total_send_bytes if x[0].startswith("able")])
# print([x for x in sorted_total_send_bytes if x[0].find("able_stablesortexpander") != -1])

print([x for x in sorted_send_actions if x[0].find("disable_algebraicsimplifier_proc_Poisson") != -1])