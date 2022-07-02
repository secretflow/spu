#! /bin/sh

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

bazel build //spu/dialect:pphlo_op_doc

cp `bazel info workspace`/bazel-bin/spu/dialect/pphlo_op_doc.md $SCRIPTPATH/pphlo_op_doc.md
