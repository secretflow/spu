#! /bin/bash
bazel build --ui_event_filters=-info,-debug,-warning //spu:spu_wheel -c opt
spu_wheel_name=$(<bazel-bin/spu/spu_wheel.name)
spu_wheel_path="bazel-bin/spu/${spu_wheel_name//sf-spu/sf_spu}"
cp $spu_wheel_path ./
cp bazel-bin/spu/spu_wheel.name ./
