#! /bin/bash

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
REPOPATH=`realpath $SCRIPTPATH/../../`

bazel run //spu/mpc/tools:complexity -- --out="$REPOPATH/docs/reference/complexity.json"
python $REPOPATH/docs/reference/gen_complexity_md.py --in="$REPOPATH/docs/reference/complexity.json" --out="$REPOPATH/docs/reference/complexity.md" 

# echo $REPOPATH
