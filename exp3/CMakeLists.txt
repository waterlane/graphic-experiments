cmake_minimum_required(VERSION 3.10)
project(RayTracingRoom LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenGL REQUIRED)

# 尝试找到 GLUT；如果使用 freeglut，请确保已安装相应开发包
find_package(GLUT REQUIRED)

add_executable(ray_tracing_room src/main.cpp)
target_link_libraries(ray_tracing_room PRIVATE OpenGL::GL GLUT::GLUT)
