#include "sift.hpp"
#include <opencv2/highgui/highgui.hpp>
int nOctaves;

namespace sift {

void findSiftInterestPoint(Mat& image, vector<KeyPoint>& kps) {
    vector<vector<Mat>> gaus_pyr;
    buildGaussianPyramid(image, gaus_pyr, nOctaves);
    vector<vector<Mat>> dog = buildDogPyr(gaus_pyr);
    getScaleSpaceExtrema(dog, kps);
    cleanPoints(image, dog, kps);
    vector<Mat> images;
    images.reserve(nOctaves);
    for(auto & octave : gaus_pyr) {
        images.emplace_back(octave.front());
    }
    vector<descriptor_t> descriptors = computeOrientationHist(images, kps);
    cv::Mat kp_image;
    cv::Mat image_char;
    image.convertTo(image_char, CV_8U);
    cv::drawKeypoints(image_char, kps, kp_image);
    cv::namedWindow("KeyPoints", cv::WINDOW_AUTOSIZE);
    cv::imshow("KeyPoints", kp_image);
    cv::waitKey(0);
}

}