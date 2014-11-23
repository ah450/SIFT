#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "constants.hpp"

#include <vector>


namespace sift {
using std::vector;
using cv::Mat;
using cv::KeyPoint;

Mat downSample(Mat& image);

void buildGaussianPyramid(Mat& image, vector<vector<Mat>>& pyr, int nOctaves);

vector<vector<Mat>> buildDogPyr(vector<vector<Mat>> gauss_pyr);

void getScaleSpaceExtrema(vector<vector<Mat>>& dog_pyr,
                          vector<KeyPoint>& keypoints);

// clean points based on contrast and principal curvature ratio
void cleanPoints(Mat& image, int curv_thr);

// calculate the gradient vector of the features
vector<double> computeOrientationHist(const Mat& image);

void findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints);
}
