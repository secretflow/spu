import os
import re
import time
import signal
import argparse
import subprocess

"---------------------------------- Parser ----------------------------------"
parser = argparse.ArgumentParser(description="Quick experiment script.")
parser.add_argument("-p", "--protocol", type=str, default="examples/python/ml/flax_gpt2/3pc.json", help="config file to use (examples/python/ml/flax_gpt2/3pc.json default)")



"---------------------------------- Main ----------------------------------"
def quick_exp(config_path: str, log_path: str, err_path: str):
    '''
    Run the quick experiment.
    '''
    
    if not os.path.exists(f"output"):
        os.makedirs(f"output")

    with open(log_path, "w") as f:
        f.write("")
    with open(err_path, "w") as f:
        f.write("")
                
    # console infomation.
    print(f"-----------------------------------")
    print(f"Config file:         {config_path}")
    
    # check if the config file exists.
    if not os.path.exists(config_path):
        raise Exception(f"Config file {config_path} not found.")
        
    # open the subprecess.
    log_f = open(log_path, "w")
    err_f = open(err_path, "w")
    nodectl = subprocess.Popen(
        f"./bazel-bin/examples/python/utils/nodectl -c {config_path} up | tee {log_path}",  
        shell=True,
        stdout=log_f)
    time.sleep(2)
    task = subprocess.Popen(
        f"./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/pumabench -c {config_path} | tee {err_path}",
        shell=True,
        stdout=err_f)
    
    # wait util the process end.
    start_time = time.time()
    while True:
        time.sleep(3)
        print(f"Time:               {time.time() - start_time:<4.2f}s", end="\r")
        if task.poll() is not None:
            if nodectl.poll() is None:
                nodectl.send_signal(signal.SIGINT)
            else:
                print("\nDone.")
                break
                    
# HEADER = "Protocol, sLogSize, rSize, Threads, Params, \
#     sOprfOprf, sOprfTotal, sOprfOnline, sQueryOnline, sQueryComputeHash, sQueryPBBC, \
#     rOprfOffline, rOprfTotal, rQueryCreate, rQueryProcess, \
#     CommOprfR2S, CommOprfS2R, CommQueryR2S, CommQueryS2R, CommTotalR2S, CommTotalS2R, CommAll, \
#     sBundlesPerBin, Matches, sMaxItemsPerBin\n"

# PATTERNS = [
#     "Sender OPRF Offline PRF[\s\S]*?(?:Duration|Durati\*):\s+(\d+\.\d+)",                   # sOprfOprf
#     "Sender OPRF Offline Total[\s\S]*?(?:Duration|Durati\*):\s+(\d+\.\d+)",                 # sOprfTotal    
#     "Sender OPRF Online[\s\S]*?(?:Duration|Durati\*):\s+(\d+\.\d+)",                        # sOprfOnline
#     "Sender Query Online[\s\S]*?(?:Duration|Durati\*):\s+(\d+\.\d+)",                       # sQueryOnline
#     "Sender::ComputePowers[\s\S]*?(?:Duration|Durati\*):\s+(\d+\.\d+)",                     # sQueryComputeHash
#     "Sender Query ProcessBinBundle[\s\S]*?(?:Duration|Durati\*):\s+(\d+\.\d+)",             # sQueryPBBC
#     "Recver OPRF Offline[\s\S]*?(?:Duration|Durati\*):\s+(\d+\.\d+)",                       # rOprfOffline
#     "Recver OPRF Total[\s\S]*?(?:Duration|Durati\*):\s+(\d+\.\d+)",                         # rOprfTotal
#     "Recver Query Create[\s\S]*?(?:Duration|Durati\*):\s+(\d+\.\d+)",                       # rQueryCreate
#     "Recver Result Process[\s\S]*?(?:Duration|Durati\*):\s+(\d+\.\d+)",                     # rQueryProcess
#     "Communication R->S, RequestOPRF:\s+(\d+\.\d+)",                                        # CommOprfR2S
#     "Communication S->R, RequestOPRF:\s+(\d+\.\d+)",                                        # CommOprfS2R
#     "Communication R->S, RequestQuery:\s+(\d+\.\d+)",                                       # CommQueryR2S
#     "Communication S->R, RequestQuery:\s+(\d+\.\d+)",                                       # CommQueryS2R
#     "Communication R->S:\s+(\d+\.\d+)",                                                     # CommTotalR2S
#     "Communication S->R:\s+(\d+\.\d+)",                                                     # CommTotalS2R
#     "Communication all :\s+(\d+\.\d+)",                                                     # CommAll
#     "The largest bundle index holds (\d+) bin bundles",                                     # sBundlesPerBin
#     "Found (\d+) matches",                                                                  # Matches
#     "eq: max_items_per_bin: (\d+)",                                                         # sMaxItemsPerBin
# ]

# def parse_log_to_csv(params):
#     '''
#     Parse the log files to csv file.
    
#     Parameters:
#         params: The parameters to use. For {params}, we will find the log files in ./output/{params}/..
#     '''
#     print(f"----------------------------------")
#     print(f"handle {params}...")
#     log_path = f"output/{params}/"
    
#     # list all the files with extend name ".log" in log_path.
#     log_files = [lf for lf in os.listdir(log_path) if lf.endswith(".log")]
    
#     # check if output dir exists, and open the output file.
#     if not os.path.exists(f"output/{params}"):
#         os.makedirs(f"output/{params}")
#     with open(f"output/{params}/{params}.csv", "w") as output_file:
#         output_file.write(HEADER)
    
#         # parse each log file.
#         visited = set()
#         for lf in log_files:
#             # extract sLogSize, rSize, Threads.
#             X, Y, t, *_ = lf.split("-")
#             t = t[1:]
            
#             # bypass the visited params set.
#             if (X, Y, t) in visited:
#                 continue
#             else:
#                 visited.add((X, Y, t))
#                 print(f"parse {X}-{Y}-t{t}...")
                
#             # parse the log file.
#             string_to_write = f"{PROTOCOL}, {X}, {Y}, {t}, {params}, "
#             log_lines = ""
#             with open(f"{log_path}{X}-{Y}-t{t}-S.log", "r") as sender_file:
#                 log_lines += "".join(sender_file.readlines())
#             with open(f"{log_path}{X}-{Y}-t{t}-R.log", "r") as recver_file:
#                 log_lines += "".join(recver_file.readlines())
            
#             for pattern in PATTERNS:
#                 result = re.search(pattern, log_lines)
#                 if result is not None:
#                     string_to_write += result.group(1) + ", "
#                 else:
#                     raise Exception(f"Pattern \"{pattern}\" not found in {X}-{Y}-t{t}.")
            
#             output_file.write(string_to_write[:-2] + "\n")
                
            

if __name__ == "__main__":
    args = parser.parse_args()
    if args.protocol.lower() == "aby3":
        args.protocol = "3pc"
    config_path = f"examples/python/ml/flax_gpt2/{args.protocol}.json"
    log_path = f"output/{args.protocol}.log"
    err_path = f"output/err.log"
    quick_exp(config_path, log_path, err_path)
    