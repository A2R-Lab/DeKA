#pragma once

#include <random>
#include <string>
#include <type_traits>

namespace deka {
namespace util {

template <typename T>
int choice(std::vector<T> pmf) {
  static_assert(std::is_floating_point<T>::value, "");

  if (pmf.size() == 0) return 0;

  std::random_device rd;
  std::mt19937 gen(rd());

  std::discrete_distribution<int> dist(pmf.begin(), pmf.end());
  
  return dist(gen);
}

std::string test();

}
}
