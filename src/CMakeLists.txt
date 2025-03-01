set(
  HEADER_LIST
  "${deka_SOURCE_DIR}/include/deka/solvers.hpp"
  "${deka_SOURCE_DIR}/include/deka/util.hpp"
)

set(
  IMPL_LIST
  "util.cpp"
  "solvers.cpp"
)

add_library(${PROJECT_NAME} ${IMPL_LIST} ${HEADER_LIST})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
