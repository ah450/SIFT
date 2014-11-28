#include "sift.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>

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
    for (std::size_t i = 0; i < mags.size(); i++) {
        mags.emplace_back(cv::Mat(dx[i].rows, dx[i].cols, sift::IMAGE_DATA_TYPE));
        std::transform(dx[i].begin<image_t>(), dx[i].end<image_t>(), dy[i].begin<image_t>(),
                mags.back().begin<image_t>(), std::hypot < image_t, image_t > );

    }
    return mags;
}

static vector<Mat> compute_thetas(vector<Mat> &dx, vector<Mat> &dy) {
    vector<Mat> thetas;
    thetas.reserve(dx.size());
    for (std::size_t i = 0; i < thetas.size(); i++) {
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

vector<descriptor_t> computeOrientationHist(const vector<Mat> &images,
            vector<KeyPoint> &kps) {
    using std::vector;
    vector<descriptor_t> histogram(kps.size());
    vector<Mat> mag, thetas;
    {
        vector<Mat> dx = compute_dx(images);
        vector<Mat> dy = compute_dy(images);
        mag = compute_mag(dx, dy);
        thetas = compute_thetas(dx, dy);
    }


    return histogram;
}

}