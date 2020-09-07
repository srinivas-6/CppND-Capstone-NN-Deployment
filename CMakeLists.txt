cmake_minimum_required(VERSION 3.7)
project(openCVObjectDetect)
# cmake needs this line
add_definitions(-std=c++14)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++14 -pthread")

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI

SET("OpenCV_DIR" "/usr/local/include")

find_package(OpenCV 4.1 REQUIRED)
include_directories(include)
include_directories(images)
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
    # Add OpenCV headers location to your include paths
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Declare the executable target built from your sources
add_executable(yolo src/main.cpp src/input.cpp src/utils.cpp)

# Link your application with OpenCV libraries
target_link_libraries(yolo ${OpenCV_LIBS})
