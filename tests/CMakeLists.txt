FetchContent_Declare(
  catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.0
)
FetchContent_MakeAvailable(catch)

add_executable(testlib test.cpp)

target_compile_features(testlib PRIVATE cxx_std_20)

target_link_libraries(testlib PRIVATE ${PROJECT_NAME} Catch2::Catch2WithMain Eigen3::Eigen)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(testlib)
