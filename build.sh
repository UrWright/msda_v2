#!/bin/bash
script_path=$(realpath $(dirname $0))

HOST_NATIVE_DIR="host_native_tiling"

SOURCE_DIR="${script_path}"
BUILD_DIR="${script_path}/build_out"

mkdir -p ${BUILD_DIR}
rm -rf ${BUILD_DIR}/*

opts=$(python3 $script_path/cmake/util/preset_parse.py $script_path/CMakePresets.json)

ENABLE_CROSS="-DENABLE_CROSS_COMPILE=True"
ENABLE_BINARY="-DENABLE_BINARY_PACKAGE=True"
ENABLE_LIBRARY="-DASCEND_PACK_SHARED_LIBRARY=True"
cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')

target=package
if [ "$1"x != ""x ]; then target=$1; fi
if [[ $opts =~ $ENABLE_LIBRARY ]]; then target=install; fi

if [[ $opts =~ $ENABLE_CROSS ]] && [[ $opts =~ $ENABLE_BINARY ]]
then
  if [ "$cmake_version" \< "3.19.0" ] ; then
    cmake -S ${SOURCE_DIR} -B "$BUILD_DIR" $opts -DENABLE_CROSS_COMPILE=0
  else
    cmake -S ${SOURCE_DIR} -B "$BUILD_DIR" --preset=default -DENABLE_CROSS_COMPILE=0
  fi
  cmake --build "$BUILD_DIR" --target cust_optiling
  mkdir $BUILD_DIR/$HOST_NATIVE_DIR
  cp $(find $BUILD_DIR -name "libcust_opmaster_rt2.0.so") $BUILD_DIR/$HOST_NATIVE_DIR
  cp -r $BUILD_DIR/$HOST_NATIVE_DIR .
  rm -rf $BUILD_DIR/*
  mv $HOST_NATIVE_DIR $BUILD_DIR
  host_native_tiling_lib=$(realpath $(find $BUILD_DIR -type f -name "libcust_opmaster_rt2.0.so"))
  if [ "$cmake_version" \< "3.19.0" ] ; then
    cmake -S ${SOURCE_DIR} -B "$BUILD_DIR" $opts -DHOST_NATIVE_TILING_LIB=$host_native_tiling_lib
  else
    cmake -S ${SOURCE_DIR} -B "$BUILD_DIR" --preset=default -DHOST_NATIVE_TILING_LIB=$host_native_tiling_lib
  fi
  cmake --build "$BUILD_DIR" --target binary -j$(nproc)
  cmake --build "$BUILD_DIR" --target $target -j$(nproc)
else
  if [ "$cmake_version" \< "3.19.0" ] ; then
		cmake -S ${SOURCE_DIR} -B "$BUILD_DIR" $opts
  else
		cmake -S ${SOURCE_DIR} -B "$BUILD_DIR" --preset=default
  fi
  cmake --build "$BUILD_DIR" --target binary -j$(nproc)
  cmake --build "$BUILD_DIR" --target $target -j$(nproc)
fi


# for debug
# cd build_out
# make
# cpack
# verbose append -v
# mkdir -p temp && tar xfvz custom_opp_euleros_aarch64.run -C temp && ./temp/install.sh

