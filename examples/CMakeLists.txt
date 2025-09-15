# CMake lowest version requirement
cmake_minimum_required(VERSION 3.16.0)

project(simu_one_layer)

add_compile_options(-std=c++11)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

if(NOT DEFINED VENDOR_NAME)
    set(VENDOR_NAME "customize" CACHE STRING "Vendor Name" FORCE)
endif()

message(STATUS "vendor name: ${VENDOR_NAME}")

set(ASCEND_TK_PATH "/usr/local/Ascend/ascend-toolkit/latest")
message(STATUS "ascend toolkit: ${ASCEND_TK_PATH}")

set(CUST_VENDOR_PATH "${ASCEND_TK_PATH}/opp/vendors/${VENDOR_NAME}")
set(CUST_OPAPI_PATH "${CUST_VENDOR_PATH}/op_api") 
set(CUST_OPTILING_PATH "${CUST_VENDOR_PATH}/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64")

include_directories(
    ${ASCEND_TK_PATH}/runtime/include
    ${ASCEND_TK_PATH}/atc/include
    ${CUST_OPAPI_PATH}/include
)

link_directories(
    ${CUST_OPAPI_PATH}/lib
    ${CUST_OPTILING_PATH}
    ${ASCEND_TK_PATH}/aarch64-linux/lib64
)

add_executable(${PROJECT_NAME}
	${PROJECT_NAME}.cpp
)

target_link_libraries(${PROJECT_NAME}
    ascendcl
    cust_opapi
    cust_opmaster_rt2.0
    graph
    nnopbase
    acl_op_compiler
    stdc++
)

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
    INSTALL_RPATH "${CUST_OPTILING_PATH}"
)

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
    INSTALL_RPATH "${CUST_OPAPI_PATH}/lib"
)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

