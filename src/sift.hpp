#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>


namespace sift {
using std::vector;
using cv::Mat;
using cv::Keypoint;

void findSiftInterestPoint(Mat& image, vector<KeyPoint>& keypoints);

void buildGaussianPyramid(Mat& image, vector<vector<Mat>>& pyr,
                            int nOctaves);

void cleanPoints(Mat& image, int curv_thr ); //based on contrast
                                            //and principal curvature ratio

Mat downSample(Mat& image);

vector<vector<Mat>> buildDogPyr(vector<vector<Mat>> gauss_pyr);

vector<double> computeOrientationHist(const Mat& image);
// Calculates the gradient vector of the feature

void getScaleSpaceExtrema(vector<vector<Mat>>& dog_pyr,
                            vector<KeyPoint>& keypoints);
    
}
