cmake_minimum_required(VERSION 3.1)
project(test)

find_package(OpenCV REQUIRED) # 找到需求的库或者组件

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

include_directories( ${OpenCV_INCLUDE_DIRS} ) # 添加一个用于包含的文件去构建的文件夹
add_executable( test main.cpp ) # 添加参与编译的项目完整包
target_link_libraries( test ${OpenCV_LIBS}  openvino::runtime) # 链接的库