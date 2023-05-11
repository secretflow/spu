echo "Run private training example..."
for net in network_a network_b network_c network_d; do
    for opt in sgd amsgrad adam; do
        start_ts=$(date +%s)
        echo "Start training "${net}" "${opt}" "${start_ts}""
        bazel run -c opt //examples/python/ml:stax_nn -- --model ${net} --optimizer ${opt} &> ${net}-${opt}.log
        end_ts=$(date +%s)
        echo "Finish training "${net}" "${opt}" "${end_ts}""
        echo "Elapsed time: $[end_ts-start_ts]"
    done
done 