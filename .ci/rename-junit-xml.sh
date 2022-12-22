#!/usr/bin/bash

rm -rf test-results
mkdir -p test-results

# renaming junit xml file to satisfy ci's requirement
for path in $(find bazel-testlogs/ -name "test.xml"); do
    dir_name=$(dirname ${path})
    file_name=$(basename ${path})
    path_md5=$(echo ${path} | md5sum | cut -f1 -d ' ')
    target="test-results/TEST-${path_md5}.xml"
    echo "mv $path to ${target} ..."
    mv ${path} ${target}
done
