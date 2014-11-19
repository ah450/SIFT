#include "test.hpp"
#include <vector>

TEST_CASE("Difference of Gaussian", "[DoG]") {
    using std::vector;
    using cv::Mat;
    Mat image(8, 8, sift::IMAGE_DATA_TYPE, img8x8_data);
    vector<vector<cv::Mat>> pyr;
    constexpr int nOctaves = 2;
    sift::buildGaussianPyramid(image, pyr, nOctaves);
    vector<vector<Mat>> dogs = sift::buildDogPyr(pyr);
    SECTION("Size of DoG") {
        REQUIRE(pyr.size() == dogs.size());
        for (auto dog_itr = dogs.begin(), pyr_itr = pyr.begin(); 
                dog_itr != dogs.end(); dog_itr++, pyr_itr++) {
            REQUIRE(dog_itr->size() == pyr_itr->size() - 1);
        }
    }
}