#include "test.hpp"


TEST_CASE("computeOrientationHist", "[hist]") {
    cv::Mat image(128, 128, sift::IMAGE_DATA_TYPE, img128x128_data);
    std::vector<std::vector<cv::Mat>> gaus_pyr;
    sift::buildGaussianPyramid(image, gaus_pyr, 6);
    std::vector<std::vector<cv::Mat>> dogs = sift::buildDogPyr(gaus_pyr);
    std::vector<cv::KeyPoint> keypoints;
    sift::getScaleSpaceExtrema(dogs, keypoints);
    std::vector<std::vector<double>> histograms;
    SECTION("Orientation Size"){
        REQUIRE_NOTHROW(histograms = sift::computeOrientationHist(dogs, keypoints));
        REQUIRE(histograms.size() == dogs.size());
        for (auto &hist : histograms ) {
            REQUIRE(hist.size() == 36);
        }


    }
}