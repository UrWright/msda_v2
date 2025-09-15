#!/bin/bash

THIS_PATH=$(realpath $(dirname $0) )

OPS_VENDOR_NAME=""

OPS_SOURCE_PATH="$(realpath ${OPS_SOURCE_PATH:-${THIS_PATH}/../})"
if [[ "${OPS_SOURCE_PATH}" != "" && -f "${OPS_SOURCE_PATH}/CMakePresets.json" ]]; then
	printf "\n[INFO] retrieve vendor name from: %s\n" "${OPS_SOURCE_PATH}/CMakePresets.json"
	VARGS=$(python ${OPS_SOURCE_PATH}/cmake/util/preset_parse.py ${OPS_SOURCE_PATH}/CMakePresets.json)
	for VX in ${VARGS}; do 
		if [[ "${VX}" =~ "-Dvendor_name" ]]; then 
			OPS_VENDOR_NAME=${VX##*=} 
		fi ; 
	done
else
	printf "\n[ERROR] unable to get preset vendor name !!\n"
	printf "%7c check source path '%s' and preset file '%s'\n" ' ' "${OPS_SOURCE_PATH}" "${OPS_SOURCE_PATH}/CMakePresets.json"
fi

OPS_VENDOR_NAME="${OPS_VENDOR_NAME:-customize}"
printf "\n[INFO] operator vendor_name: %s\n" "$OPS_VENDOR_NAME"

printf "\n\n  !!!! Build Begin !!!! \n\n"
rm -rf $THIS_PATH/build_out
mkdir -p $THIS_PATH/build_out
cd $THIS_PATH/build_out && \
	cmake -DVENDOR_NAME=${OPS_VENDOR_NAME} .. && \
	make && \
	printf "\n\n  !!!! Build Done !!!! \n\n"
