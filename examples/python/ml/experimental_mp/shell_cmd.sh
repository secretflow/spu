model="bert"

if [ "$1" == "vanilla" ]
then
bazel run -c opt //examples/python/ml/experimental_mp:"$model"_bench -- --gelu raw --softmax raw --config examples/python/ml/experimental_mp/3pc.json
elif [ "$1" == "puma" ]
then
bazel run -c opt //examples/python/ml/experimental_mp:"$model"_bench -- --gelu puma --softmax puma --config examples/python/ml/experimental_mp/3pc_nExp.json
elif [ "$1" == "ditto_quan" ]
then
bazel run -c opt //examples/python/ml/experimental_mp:"$model"_bench -- --gelu puma --softmax puma --config examples/python/ml/experimental_mp/3pc_nExp.json
elif [ "$1" == "ditto" ]
then
bazel run -c opt //examples/python/ml/experimental_mp:"$model"_bench -- --gelu quad --softmax puma --config examples/python/ml/experimental_mp/3pc_nExp.json
fi