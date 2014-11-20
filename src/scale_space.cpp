#include "sift.hpp"
#include "internal.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <utility>
#include <iterator>
#include <algorithm>



namespace sift {

void buildGaussianPyramid(Mat& image, vector<vector<Mat>>& pyr, int nOctaves) {
    Mat image_ds = image.clone();
    pyr.resize(nOctaves);
    for (int o = 0; o < nOctaves; o++) {
        auto sigma = GAUSSIAN_PYR_SIGMA0;
        for (auto s = 0; s < GAUSSIAN_PYR_OCTAVE_SIZE; s++) {
            Mat image_ds_blur(image_ds.rows, image_ds.cols, image_ds.type());
            cv::GaussianBlur(
                    image_ds, image_ds_blur,
                    cv::Size(GAUSSIAN_PYR_KERNEL_SIZE, GAUSSIAN_PYR_KERNEL_SIZE),
                    sigma);
            pyr[o].push_back(image_ds_blur);

            sigma *= GAUSSIAN_PYR_K;
        }

        image_ds = downSample(pyr[o][GAUSSIAN_PYR_OCTAVE_SIZE-3]);
    }
}

/**
 * @brief Builds a difference of gaussian for each octave
 * 
 * @param gauss_pyr 2D vector representing pyramids of all octaves.
 * @remark order of dogs is identical to gauss_pyr
 * @return 2D vector of DoGs
 */
vector<vector<Mat>> buildDogPyr(vector<vector<Mat>> gauss_pyr) {
    vector<vector<Mat>> dogs(gauss_pyr.size());
    auto dog_iterator  = std::begin(dogs);
    for (const auto &octave : gauss_pyr) {
        auto start = std::begin(octave);
        auto end = --std::end(octave);
        std::transform(start, end, start + 1, 
            std::back_inserter(*dog_iterator),
            [](const Mat & lower, const Mat &upper) {
                return upper - lower;  
            } );
        dog_iterator++;
    }
    return dogs;
}



/**
 * @brief Detects keypoints from DoG
 * @details Finds extremas in a 3x3x3 window
 *
 * @param dog_pyr DoGs for each octave
 * @param keypoints vector to be populated
 */
void getScaleSpaceExtrema(vector<vector<Mat>> &dog_pyr,
vector<KeyPoint> &keypoints) {
    for (std::size_t octave = 0; octave < dog_pyr.size(); octave++) {
        auto &octave_dog = dog_pyr[octave];
        auto octave_sigma = internal::compute_octave_sigma(octave);
        for (std::size_t image = 1; image < octave_dog.size() - 1; image++) {
            vector<internal::point> points = internal::find_local_extremas(
                octave_dog[image - 1], octave_dog[image], octave_dog[image + 1]);
            // Now we have x, y coordinates of extremas, we need to create keypoints
            /***********************************************************************************************
             *                            References                                                       *
             *- http://answers.opencv.org/question/7337/keypoint-size/                                     *
             *- http://www.aishack.in/tutorials/sift-scale-invariant-feature-transform-keypoints/          *
             *- http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html  *
             ***********************************************************************************************/
            for (const auto &p : points) {
                // FIXME: need to compute response value
                // found abs(interpolated_DoG_value), what does that mean?
                // FIXME: Use subpixel values 
                keypoints.emplace_back(p.first, p.second, octave_sigma, -1, 0, octave);
            }
        }
    }
}



}


