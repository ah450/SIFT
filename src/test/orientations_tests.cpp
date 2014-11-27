#include "test.hpp"


TEST_CASE("computeOrientationHist", "[hist]") {
    cv::Mat image = clone_test_data(128, 128, img128x128_data);
    std::vector<std::vector<cv::Mat>> gaus_pyr;
    sift::buildGaussianPyramid(image, gaus_pyr, 6);
    std::vector<std::vector<cv::Mat>> dogs = sift::buildDogPyr(gaus_pyr);
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Mat> images;
    for (std::size_t i = 0; i < gaus_pyr.size(); i++) {
        images.emplace_back(gaus_pyr[i][0]);
    }
    sift::getScaleSpaceExtrema(dogs, keypoints);
    std::vector<std::vector<double>> histograms;
    SECTION("Orientation Size"){
        REQUIRE_NOTHROW(histograms = sift::computeOrientationHist(images, keypoints));
        REQUIRE(histograms.size() == keypoints.size());
        for (auto &hist : histograms ) {
            REQUIRE(hist.size() == 36);
        }


    }
}
