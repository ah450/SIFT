#pragma once
#include <sift.hpp>
#include <utility>
#include <cstddef>




namespace sift {
namespace internal{

typedef std::pair<std::size_t, std::size_t> point;

vector<point> find_local_extremas(const Mat &lower_dog,
                                  const Mat &current_dog, const Mat &upper_dog);


}
}