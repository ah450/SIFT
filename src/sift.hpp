#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "constants.hpp"

#include <vector>
#include <array>

namespace sift {
using std::vector;
using std::array;
using cv::Mat;
using cv::KeyPoint;

Mat downSample(Mat& image);

void buildGaussianPyramid(Mat& image, vector<vector<Mat>>& pyr, int nOctaves);

vector<vector<Mat>> buildDogPyr(vector<vector<Mat>> gauss_pyr);

void getScaleSpaceExtrema(vector<vector<Mat>>& dog_pyr,
                          vector<KeyPoint>& keypoints);

// clean points based on contrast and principal curvature ratio
void cleanPoints(Mat& image, const vector<vector<Mat>>& dog_pyr, vector<KeyPoint>& keypoints);

// calculate the gradient vector of the features
typedef array<array<double, 8>, 16> descriptor_t;

vector<descriptor_t> computeOrientationHist(const vector<Mat> &images, vector<KeyPoint> &kps);

void findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints);
}
