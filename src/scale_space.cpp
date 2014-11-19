#include "sift.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
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
            std::back_inserter(*dog_iterator), [](const Mat & lower, const Mat &upper) {
                return upper - lower;  
            } );
        dog_iterator++;
    }
    return dogs;
}

}


