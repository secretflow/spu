import yaml

with open("EasyLM-repo/scripts/gpu_environment.yml") as file_handle:
    environment_data = yaml.load(file_handle, yaml.SafeLoader)

with open("gpu_environment_requests.txt", "w") as file_handle:
    for dependency in environment_data["dependencies"][-1]["pip"]:
        package_name = dependency
        file_handle.write("{}\n".format(package_name))