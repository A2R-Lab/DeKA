#include "quadrotor.hpp"

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

using namespace autodiff;

VectorXreal quad_dynamics(
  const VectorXreal& x,
  const VectorXreal& u,
  const real& mass,
  const real& g
) {
  
}

VectorXreal quad_dynamics_rk4(
  const VectorXreal& x,
  const VectorXreal& u,
  const real& mass,
  const real& g
) {
  auto f1 = quad_dynamics(x, u, mass, g);
  auto f2 = quad_dynamics(x, u, mass, g);
  auto f3 = quad_dynamics(x, u, mass, g);
  auto f4 = quad_dynamics(x, u, mass, g);
}
