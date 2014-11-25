#pragma once
#include <opencv2/core/core.hpp>
#include <ctype.h>
#include <cmath>

namespace sift {
    const auto IMAGE_DATA_TYPE = CV_64F;
    typedef double image_t;

    const auto GAUSSIAN_PYR_KERNEL_SIZE = 3;
    // sigma of the gaussian blur kernel
    const auto GAUSSIAN_PYR_SIGMA0      = std::sqrt(2);

    // size of each octave is s+3
    const auto GAUSSIAN_PYR_S           = 2;
    // factor between sigma of each blurred image
    const auto GAUSSIAN_PYR_K           = std::pow(2.0, 1.0/GAUSSIAN_PYR_S);

    // size of each octave (number of blurred images)
    const auto GAUSSIAN_PYR_OCTAVE_SIZE = GAUSSIAN_PYR_S + 3;
}
