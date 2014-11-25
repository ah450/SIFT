#include "sift.hpp"
#include "internal.hpp"
#include <opencv2/opencv.hpp>


namespace sift {

template <class Func_t>
class LazyMat {
    typedef typename Func_t::result_type value_t;
    Func_t f;
    std::vector<bool> is_computed;
    std::vector<value_t> values;

public:
    const int rows, cols;
    LazyMat(Func_t f, int rows, int cols): f(f), is_computed(rows *cols),
        values(rows *cols), rows(rows), cols(cols) {}
    template <typename ReturnValue>
    ReturnValue at(int x, int y) {
        int index  = x * cols + y;
        if (is_computed[index]) {
            return static_cast<ReturnValue>(values[index]);
        } else {
            is_computed[index] = true;
            return static_cast<ReturnValue>(values[index] = f(x, y));
        }
    }

};

inline double delta(const Mat &image, int x1, int y1, int x2, int y2) {
    return image.template at<sift::image_t>(x2,
                                            y2) - image.template at<sift::image_t>(x1, y1);
}

double delta_y(const Mat &image, int x, int y) {
    int above = y < image.rows - 1 ? y + 1 : y;
    int below = y > 0 ? 0 : 1;
    return delta(image, x, above, x, below);
}

double delta_x(const Mat &image, int x, int y) {
    int after = x < image.cols - 1 ? x + 1 : x;
    int before = x > 0 ? 0 : 1;
    return delta(image, after, y, before, y);
}


struct Neighbourhood {
    int kernel_size, row_start, row_end, col_start, col_end;
    Neighbourhood(int num_rows, int num_columns, int row, int col,
                  double scale, double min = 5.0) {
        kernel_size = std::max(3 * scale, min);
        if (kernel_size % 2  == 0) {
            kernel_size += 1;
        }
        int kernel_half = kernel_size / 2;
        row_start = row > kernel_half ? row - kernel_half : 0;
        row_end = row + kernel_half < num_rows - 1 ? row + kernel_half + 1 : num_rows;
        col_start = col > kernel_half ? col - kernel_half : 0;
        col_end = col + kernel_half < num_columns - 1 ? col + kernel_half + 1 :
                  num_columns;
    }
};

typedef std::function<double (int, int)> eval_func_t;

/**
 * @brief creates a function that applies a smoothing filter.
 * @return function wrapper
 */
template <class MagnitudesMat>
eval_func_t create_smoothing_func(MagnitudesMat &grad_magnitudes, KeyPoint &kp,
                                  int kernel_size) {

    Mat gauss_x = cv::getGaussianKernel(kernel_size, kp.size);
    Mat gauss_y = cv::getGaussianKernel(kernel_size, kp.size);
    // Create 2D gauss
    Mat gauss = gauss_x * gauss_y.t();
    auto scale = kp.size;
    // Function to apply gaussian depends on gra
    auto apply_gauss = [&grad_magnitudes, gauss, scale](int row, int column) {
        Neighbourhood area(grad_magnitudes.rows, grad_magnitudes.cols, row, column,
                           scale);
        double value = 0.0;
        for (int i = area.row_start; i < area.row_end; i++) {
            for (int j = area.col_start; j < area.col_end; j++ ) {
                value += grad_magnitudes.template at<double>(i, j) *
                         gauss.at<double>(i - area.row_start, j - area.col_start);
            }
        }
        return value;
    };
    return apply_gauss;
}

/**
 * @brief Creates a function that calculates the gradient magnitude at a point.
 * @return function wrapper.
 */
template <class DxMat, class DyMat>
eval_func_t create_magnitude_func(DxMat &dx_mat, DyMat &dy_mat) {
    using namespace std::placeholders;
    auto get_mag = [&dx_mat, &dy_mat](int row, int column) {
        double dx = dx_mat.template at<double>(row, column);
        double dy = dy_mat.template at<double>(row, column);
        return std::hypot(dx, dy);
    };
    return get_mag;
}

/**
 * @brief Creates a function that calculates the slope angle from gradient values.
 * @remark angle is computed in degrees between [0, 359.0]
 * @return function wrapper
 */
template <class DxMat, class DyMat>
eval_func_t create_angle_func(DxMat &dx_mat, DyMat &dy_mat) {
    auto get_angle = [&dx_mat, &dy_mat](int row, int column) {
        double dx = dx_mat.template at<double>(row, column);
        double dy = dy_mat.template at<double>(row, column);
        // Convert value to degrees
        double angle;
        if (dx == 0.0) {
            // angle either pi/2 or 3pi/2
            if (std::signbit(dy)) {
                angle = M_PI / 2.0;
            } else {
                angle = 3 / 2 * M_PI;
            }
            angle *=  (180.0 / M_PI);
        } else {
            angle =  std::atan(dy / dx) * (180.0 / M_PI);
            // ensure range [0, 180]
            angle = std::max(-90.0, std::min(angle, 90.0)) + 90;
            // map to [0, 359]
            if (std::signbit(dy / dx)) {
                // negative gradient means going down
                angle = angle + 180;
            }
            angle =  std::max(0.0, std::min(angle, 359.0));
        }
        return angle;

    };
    return get_angle;
}



vector<vector<double>> computeOrientationHist(const vector<Mat> &images,
vector<KeyPoint> &kps) {
    vector<vector<double>> histograms;
    histograms.reserve(kps.size());
    using namespace std::placeholders;
    vector<LazyMat<eval_func_t>> dx, dy, grads, angles;


    for (auto &image : images) {
        auto xf = std::bind(delta_x, std::ref(image), _1, _2);
        auto yf = std::bind(delta_y, std::ref(image), _1, _2);
        dx.emplace_back(xf, image.rows, image.cols);
        dy.emplace_back(yf, image.rows, image.cols);
    }

    for (std::size_t i = 0; i < dx.size(); i++) {
        grads.emplace_back(create_magnitude_func(dx[i], dy[i]), dx[i].rows, dx[i].cols);
        angles.emplace_back(create_angle_func(dx[i], dy[i]), dx[i].rows, dx[i].cols);
    }




    // Calculate histograms of each keypoint
    for (auto &kp : kps) {
        LazyMat<eval_func_t> &grad_magnitudes = grads[kp.octave];
        LazyMat<eval_func_t> &angles_mat = angles[kp.octave];
        std::vector<double> kp_histogram(36,
                                         0.0); // Histogram for current keypoint to be populated
        // Number of samples taken into account depend on scale of keypoint
        // NeighbourhoodArea is a helper class for finding the row and column boundries
        Neighbourhood area_info(grad_magnitudes.rows,
                                grad_magnitudes.cols, kp.pt.x, kp.pt.y, kp.size);
        // Need to smooth magnitudes with special filter based on scale
        auto smoothing_func = create_smoothing_func(grad_magnitudes, kp,
                              area_info.kernel_size);
        // Lazily evaluate as usual.
        LazyMat<eval_func_t> smoothed_magnitudes(smoothing_func, grad_magnitudes.rows,
                grad_magnitudes.cols);
        // In our area of interest, calculate histogram.
        for (int i = area_info.row_start; i < area_info.row_end; i++) {
            for (int j = area_info.col_start; j < area_info.col_end; j++) {
                // Important to note that angle is [0, 359]
                double angle = angles_mat.template at<double>(i, j);
                // Bucket index
                int index = static_cast<int>(angle / 10);
                // Add weighted magnitude value.
                double magnitude = smoothed_magnitudes.at<double>(i, j);
                kp_histogram[index] += magnitude;
            }
        }
        kp.angle = angles_mat.template at<double>(kp.pt.x, kp.pt.y);
        kp.response = smoothed_magnitudes.at<double>(kp.pt.x, kp.pt.y);
        histograms.emplace_back(kp_histogram);

    }

    return histograms;
}
}