#include "test.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace std;
using namespace sift;

TEST_CASE("Create Gaussian Pyramid from an image and nOctaves", "[buildGaussianPyramid]") {

    Mat image(8, 8, sift::IMAGE_DATA_TYPE, img8x8_data);
    vector<vector<Mat>> pyr;
    int nOctaves = 2;

    buildGaussianPyramid(image, pyr, nOctaves);

    SECTION("images in octave 1 should be the original size (8x8)") {
        for (auto img: pyr[0]) {
            REQUIRE(img.cols == 8);
            REQUIRE(img.rows == 8);
        }

        SECTION("images in octave 2 should be half the size (4x4)") {
            for (auto img: pyr[1]) {
                REQUIRE(img.cols == 4);
                REQUIRE(img.rows == 4);
            }

        }
    }
}
