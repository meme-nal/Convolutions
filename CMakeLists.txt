cmake_minimum_required(VERSION 3.15)
project(conv)

set(SOURCES
  main.cpp
  conv.cpp
  lin.cpp
  net.cpp)

set(HEADERS
  conv.hpp
  lin.hpp
  net.hpp)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

find_package(nlohmann_json 3.2.0 REQUIRED)

set_target_properties(${PROJECT_NAME} PROPERTIES
CXX_STANDARD 17
CXX_STANDARD_REQUIRED ON)

target_link_libraries (${PROJECT_NAME} PUBLIC 
nlohmann_json::nlohmann_json)