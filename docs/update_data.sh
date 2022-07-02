#! /bin/bash

echo "1. Update protocol complexity doc."
bash reference/update_complexity_doc.sh

echo "2. Update pphlo doc."
bash reference/update_pphlo_doc.sh

echo "3. Update runtime config doc."
# runtime_config_md.tmpl is adapted from https://github.com/pseudomuto/protoc-gen-doc/blob/master/examples/templates/grpc-md.tmpl.
docker run --rm -v $(pwd)/reference/:/out \
                -v $(pwd)/../spu:/protos \
                pseudomuto/protoc-gen-doc \
                --doc_opt=/out/runtime_config_md.tmpl,runtime_config.md spu.proto

