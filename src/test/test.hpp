#pragma once
#include "catch.hpp"
#include "sift.hpp"
#include "internal.hpp"

extern sift::image_t img128x128_data[128][128];
extern sift::image_t img8x8_data[8][8];
extern sift::image_t img4x4_data[4][4];
extern sift::image_t lower_9x9_extrema[9][9];
extern sift::image_t current_9x9_extrema[9][9];
extern sift::image_t upper_9x9_extrema[9][9];
extern std::array<sift::internal::point, 7> extremas;

inline cv::Mat clone_test_data(int width, int height, void* data) {
  return cv::Mat(width, height, sift::IMAGE_DATA_TYPE, data).clone();
}
