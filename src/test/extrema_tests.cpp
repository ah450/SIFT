#include "test.hpp"


TEST_CASE("getScaleSpaceExtrema", "[extrema]") {
    cv::Mat image(128, 128, sift::IMAGE_DATA_TYPE, img128x128_data);
    std::vector<std::vector<cv::Mat>> gaus_pyr;
    sift::buildGaussianPyramid(image, gaus_pyr, 6);
    std::vector<std::vector<cv::Mat>> dogs = sift::buildDogPyr(gaus_pyr);

    SECTION("No Throw") {
        std::vector<cv::KeyPoint> keypoints;

        REQUIRE_NOTHROW(sift::getScaleSpaceExtrema(dogs, keypoints));
        INFO("Num keypoints " << keypoints.size());
        REQUIRE_FALSE(keypoints.empty());
    }
    
}