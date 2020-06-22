
include_directories(${PROJECT_SOURCE_DIR}/common_tool/include)



add_executable(mulprocess mulprocess.cpp)

cuda_add_executable(mps_test mps_test.cu)

target_link_libraries(mps_test common_tool)