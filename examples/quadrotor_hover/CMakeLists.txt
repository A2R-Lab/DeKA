cmake_minimum_required(VERSION 3.15...3.31)

project(
  quadrotor_hover
  LANGUAGES CXX
)

find_package(deka CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(autodiff REQUIRED)

message("deka_DIR: ${deka_DIR}")
message("deka_INCLUDE_DIR: ${deka_INCLUDE_DIR}")

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} deka::deka Eigen3::Eigen autodiff::autodiff)

add_custom_target(
  run
  COMMAND quadrotor_hover
  DEPENDS quadrotor_hover
  WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
