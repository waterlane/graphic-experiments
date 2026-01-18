cmake_minimum_required(VERSION 3.10)
project(ObjPreview LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Handle older GLEW packages that do not provide imported targets
if(TARGET GLEW::GLEW)
    set(GLEW_TARGET GLEW::GLEW)
else()
    set(GLEW_TARGET ${GLEW_LIBRARIES})
    include_directories(${GLEW_INCLUDE_DIRS})
endif()

add_executable(obj_viewer src/main.cpp)
target_link_libraries(obj_viewer PRIVATE ${GLEW_TARGET} glfw OpenGL::GL)
