#pragma once
#include <opencv2/core/core.hpp>
#include <ctype.h>
#include <cmath>

namespace sift {
    constexpr auto IMAGE_DATA_TYPE = CV_8U;
    typedef uint8_t image_t ;

    constexpr auto GAUSSIAN_PYR_KERNEL_SIZE = 3;
    // sigma of the gaussian blur kernel
    constexpr auto GAUSSIAN_PYR_SIGMA0      = std::sqrt(2);
    // size of each octave is s+3
    constexpr auto GAUSSIAN_PYR_S           = 2;
    // factor between sigma of each blurred image
    constexpr auto GAUSSIAN_PYR_K           = std::pow(2.0, 1.0/GAUSSIAN_PYR_S);
    // size of each octave (number of blurred images)
    constexpr auto GAUSSIAN_PYR_OCTAVE_SIZE = GAUSSIAN_PYR_S + 3;
}
