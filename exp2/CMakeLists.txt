cmake_minimum_required(VERSION 3.10)
project(QuatPathDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

if(TARGET GLEW::GLEW)
    set(GLEW_TARGET GLEW::GLEW)
else()
    set(GLEW_TARGET ${GLEW_LIBRARIES})
    include_directories(${GLEW_INCLUDE_DIRS})
endif()

add_executable(quat_path_viewer src/main.cpp)
target_link_libraries(quat_path_viewer PRIVATE ${GLEW_TARGET} glfw OpenGL::GL)
