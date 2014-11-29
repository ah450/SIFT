#include "sift.hpp"
#include "internal.hpp"
#include <opencv2/opencv.hpp>
#include <tbb/tbb.h>
#include <cmath>
#include <cassert>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <future>

namespace sift {

static vector<Mat> compute_dx(const vector<Mat> &images) {
    std::vector<Mat> dx;
    dx.reserve(images.size());
    for (auto &image : images) {
        Mat out;
        cv::Sobel(image, out, sift::IMAGE_DATA_TYPE, 1, 0, 3);
        dx.emplace_back(out);
    }
    return dx;
}


static vector<Mat> compute_dy(const vector<Mat> &images) {
    std::vector<Mat> dy;
    dy.reserve(images.size());
    for (auto &image : images) {
        Mat out;
        cv::Sobel(image, out, sift::IMAGE_DATA_TYPE, 0, 1, 3);
        dy.emplace_back(out);
    }
    return dy;
}

static vector<Mat> compute_mag(vector<Mat> &dx, vector<Mat> &dy) {
    vector<Mat> mags;
    mags.reserve(dx.size());
    for (std::size_t i = 0; i < dx.size(); i++) {
        mags.emplace_back(cv::Mat(dx[i].rows, dx[i].cols, sift::IMAGE_DATA_TYPE));
        std::transform(dx[i].begin<image_t>(), dx[i].end<image_t>(), dy[i].begin<image_t>(),
                mags.back().begin<image_t>(), std::hypot < image_t, image_t > );

    }
    return mags;
}

static vector<Mat> compute_thetas(vector<Mat> &dx, vector<Mat> &dy) {
    vector<Mat> thetas;
    thetas.reserve(dx.size());
    for (std::size_t i = 0; i < dx.size(); i++) {
        // angles always as doubles
        thetas.emplace_back(cv::Mat(dx[i].rows, dx[i].cols, CV_64F));
        std::transform(dx[i].begin<image_t>(), dx[i].end<image_t>(), dy[i].begin<image_t>(),
                thetas.back().begin<double>(),
                [](const image_t &dx, const image_t &dy) -> double {
                    double angle = 0.0;
                    if (dx == 0) {
                        if (std::signbit(dy)) {
                            angle = M_PI / 2.0;
                        } else {
                            angle = 3 / 2 * M_PI;
                        }
                        angle *= 180.0 / M_PI;
                    } else {
                        angle = std::tan(dy / dx) * (180 / M_PI);
                        // shift from [-90, 90] to [0, 180]
                        angle = std::max(-90.0, std::min(angle, 90.0)) + 90.0;
                        // Map to [0, 359]
                        if (std::signbit(dy / dx)) {
                            // negative gradient means going down
                            angle = angle + 180;
                        }
                        angle = std::max(0.0, std::min(angle, 359.0));

                    }
                    return angle;
                });

    }
    return thetas;
}

class LazyGauss {
    const Mat * orig;
    std::vector<bool> is_computed;
    std::vector<double> values;
    std::function<double (const cv::Mat &, int, int)> gauss_func;
public:
    LazyGauss(): orig(nullptr){} // required for, containers.
    LazyGauss(int kernel_size, double scale, const Mat * orig): orig(orig), is_computed(orig->rows * orig->cols, false),
    values(orig->rows * orig->cols, 0){
        cv::Mat gauss = cv::getGaussianKernel(kernel_size, scale) * cv::getGaussianKernel(kernel_size, scale).t();
        gauss_func = [gauss](const cv::Mat &orig, int row, int column) {
            internal::Neighbourhood area(orig.rows, orig.cols, row, column, gauss.rows/2);
            double value = 0.0;
            for (int i = area.row_start; i < area.row_end; i++) {
                for(int j = area.col_start; j < area.col_end; j++) {
                    value += gauss.at<double>(i  - area.row_start, j - area.col_start) *
                            orig.at<double>(i, j);
                }
            }
            return value;
        };
    }

    double operator()(int row, int column) {
        if (is_computed[row * orig->cols + column]) {
            return values[row * orig->cols + column];
        }else{
            is_computed[row * orig->cols + column] = true;
            return values[row * orig->cols + column] = gauss_func(*orig, row, column);
        }
    }

    template <typename ReturnType>
    ReturnType at(int row, int column) {
        return (*this)(row, column);
    }
};



vector<descriptor_t> computeOrientationHist(const vector<Mat> &images,
            vector<KeyPoint> &kps) {
    auto future = std::async(std::launch::async, [&kps]() -> void {
        std::sort(kps.begin(), kps.end(), [](const KeyPoint &lhs, const KeyPoint &rhs) {
           return lhs.size < rhs.size;
        });
    });
    using std::vector;
    vector<descriptor_t> descriptors(kps.size());
    vector<Mat> mags, thetas;
    {
        vector<Mat> dx = compute_dx(images);
        vector<Mat> dy = compute_dy(images);
        mags = compute_mag(dx, dy);
        thetas = compute_thetas(dx, dy);
    }
    future.get();
    auto compute_func = [&](const tbb::blocked_range<std::size_t> &range) {
        std::vector<std::unordered_map<double, LazyGauss>> cache(mags.size());

        for (std::size_t index = range.begin(); index != range.end(); index++) {
            KeyPoint &kp = kps[index];
            auto &mag = mags[kp.octave];
            auto &theta = thetas[kp.octave];
            kp.angle = theta.at<double>(kp.pt.x, kp.pt.y);
            // Create Gaussian window based on keypoint scale
            descriptor_t desc({0});
            internal::Neighbourhood area(mag.rows, mag.cols, kp.pt.x, kp.pt.y, 8);
            std::size_t hist_i = 0;
            if( ! cache[kp.octave].count(kp.size)) {
              cache[kp.octave][kp.size] = LazyGauss(9, kp.size, &mag);
            }
            LazyGauss & weightedMagnitudes = cache[kp.octave][kp.size];
            for (int sub_window_row = area.row_start + 4; sub_window_row < area.row_end; sub_window_row += 4) {
                for (int sub_window_col = area.col_start + 4; sub_window_col < area.col_end; sub_window_col += 4) {

                    internal::Neighbourhood sub_area(mag.rows, mag.cols, sub_window_row, sub_window_col, 2);
                    // Compute histogram for each 4*4 window
                    assert(hist_i < 16);
                    histogram_t &hist = desc[hist_i++];

                    for (int hist_row = sub_area.row_start; hist_row < sub_area.row_end; hist_row++) {
                        for (int hist_col = sub_area.col_start; hist_col < sub_area.col_end; hist_col++) {
                            hist[theta.at<double>(hist_row, hist_col) / 45.0] += weightedMagnitudes.at<double>(hist_row, hist_col);
                        }
                    }

                }
            }
            descriptors[index] = desc;
        }
    };

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, kps.size()), compute_func);
    return descriptors;
}

}