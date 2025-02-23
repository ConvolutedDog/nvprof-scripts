#!/bin/bash

apps_root="./ffma_cudacore"
ncu_path="/usr/local/cuda-11.8/bin/ncu"

##################################################################################
###                                                                            ###
###                                  test.cu                                   ###
###                                                                            ###
##################################################################################

cd "$apps_root" || { echo "Failed to cd to $apps_root"; exit 1; }

make clean && make || { echo "Build failed"; exit 1; }

if [ ! -d "./NsightCollection" ]; then
    mkdir "./NsightCollection"
fi

ncu_cmd="$ncu_path \
         --export ./test_ncu_report_file \
         --force-overwrite --target-processes application-only --devices 0 --set full \
	     ./test 128 128 128"
expect ../expect.expect "$ncu_cmd" || { echo "ncu command failed"; exit 1; }

chmod_cmd="chmod u+rw *.ncu-rep"
expect ../expect.expect "$chmod_cmd" || { echo "chmod failed"; exit 1; }

nsight_save_dir=./NsightCollection
mv -f *.ncu-rep $nsight_save_dir
