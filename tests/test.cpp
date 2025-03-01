#include <deka/solvers.hpp>
#include <deka/util.hpp>

#define CATCH_CONFIG_MAIN

#include <catch2/catch_test_macros.hpp>

TEST_CASE("test_hello_world") {
  std::string value = deka::util::test();
  REQUIRE(value == std::string("Hello world!"));
}

TEST_CASE("test_rk") {
  Eigen::MatrixXd A(3, 3);

  A << 2, 3, 5,
       3, -1, 2,
       5, 2, 6;

  Eigen::VectorXd x0 {{0, 0, 0}};
  Eigen::VectorXd b {{10, 5, 12}};

  deka::solvers::RecursiveKaczmarz rk(1000);

  auto x = rk.iterate(A, b, x0);
  auto res = A * x;
  bool approx = res.isApprox(b);

  REQUIRE(approx == true);
}
