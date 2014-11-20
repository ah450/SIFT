#pragma once
#include <sift.hpp>
#include <utility>
#include <cstddef>
#include <cmath>



namespace sift {
namespace internal{

typedef std::pair<std::size_t, std::size_t> point;

vector<point> find_local_extremas(const Mat &lower_dog,
                                  const Mat &current_dog, const Mat &upper_dog);



inline auto compute_octave_sigma(const int ocatave_number) -> decltype(GAUSSIAN_PYR_SIGMA0) {
    return std::sqrt(std::pow(2, ocatave_number) * GAUSSIAN_PYR_SIGMA0 * 
        std::pow(GAUSSIAN_PYR_K, 2));
}

}
}