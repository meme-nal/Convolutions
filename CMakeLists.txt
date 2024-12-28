cmake_minimum_required(VERSION 3.15)
project(conv)
add_executable(${PROJECT_NAME} main.cpp conv.cpp conv.hpp)

set_target_properties(${PROJECT_NAME} PROPERTIES
CXX_STANDARD 17
CXX_STANDARD_REQUIRED ON)