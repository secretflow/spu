```shell
#   flax_llama7b
#    |- EasyLM
#    |- open_llama_7b
#    |- open_llama_7b_flax
#    |- ...

cd examples/python/ml/flax_llama7b
pip install 'transformers[flax]'
git clone https://github.com/young-geng/EasyLM.git
mv EasyLM EasyLM-repo
cp -r EasyLM-repo/EasyLM EasyLM

cd EasyLM-repo
git checkout 059f3f4dd831825f39cb8121111905205f4a7241

cd EasyLM/models/llama
sed -i '185c default=False,' ./EasyLM/EasyLM/models/llama/convert_hf_to_easylm.py
python convert_hf_to_easylm.py --checkpoint_dir ../../../../open_llama_7b --output_file ../../../../open_llama_7b_flax --model_size 7b

cd ../../../.. # examples/python/ml/flax_llama7b
touch convert_repyaml_to_reqtxt.py
echo '''
import yaml

with open("EasyLM-repo/scripts/gpu_environment.yml") as file_handle:
    environment_data = yaml.load(file_handle, yaml.SafeLoader)

with open("gpu_environment_requests.txt", "w") as file_handle:
    for dependency in environment_data["dependencies"][-1]["pip"]:
        package_name = dependency
        file_handle.write("{}\n".format(package_name))
''' | cat > convert_repyaml_to_reqtxt.py
pip install -r gpu_environment_requests.txt 
sed -i '29c from EasyLM.models.llama.llama_model import FlaxLLaMAForCausalLM, LLaMAConfigurator' ./flax_llama7b.py

cd ../../.. # spu root dir
bazel run -c opt //examples/python/utils:nodectl -- --config ./examples/python/ml/flax_llama7b/3pc.json up
bazel run -c opt //examples/python/ml/flax_llama7b -- --model_path ./examples/python/ml/flax_llama7b/open_llama_7b_flax --config ./examples/python/ml/flax_llama7b/3pc.json
```