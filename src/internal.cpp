#include <internal.hpp>



inline void max_min(int row_start, int row_end, int col_start,
                    int col_end, const cv::Mat &mat,
                    sift::image_t &max, sift::image_t &min) {

    for (int row = row_start; row < row_end; row++) {
        for (int col = col_start; col < col_end; col++) {
            sift::image_t current = mat.at<sift::image_t>(row, col);
            if (current > max) {
                max = current;
            } else if (current < min) {
                min = current;
            }
        }
    }
}


inline bool is_max_or_min(int row, int col,
                          const cv::Mat &lower,
                          const cv::Mat &current, const cv::Mat &upper) {
    sift::internal::Neighbourhood area(current.rows, current.cols, row, col, 1);
    sift::image_t center = current.at<sift::image_t>(row, col);
    sift::image_t max(center), min(center);    
    // check in current
    max_min(area.row_start, area.row_end, area.col_start, area.col_end, current,
        max, min);
    // check in upper
    max_min(area.row_start, area.row_end, area.col_start, area.col_end, upper,
        max, min);
    // cehck in lower
    max_min(area.row_start, area.row_end, area.col_start, area.col_end, lower,
        max, min);
    return (max == center) | (min == center);
}


namespace sift {
namespace internal {

vector<point> find_local_extremas(const Mat &lower_dog,
                                  const Mat &current_dog, const Mat &upper_dog) {
    vector<point> points;
    for (int row = 0 ; row < current_dog.rows; row++) {
        for (int col = 0; col < current_dog.cols; col++) {
            bool is_extrema = is_max_or_min(row, col, lower_dog, current_dog, upper_dog);
            if (is_extrema) {
                points.emplace_back(row, col);
            }
        }
    }
    return points;
}

double compute_keypoint_curvature(const vector<vector<Mat>>& dog_pyr, const KeyPoint kp) {
    image_t xx_d[3]    = {1.0, -2.0, 1.0};
    image_t xy_d[3][3] =
       {{1.0,  0.0, -1.0},
        {0.0,  0.0,  0.0},
        {-1.0, 0.0,  1.0}};

    Mat xx(3, 1, IMAGE_DATA_TYPE, xx_d);
    Mat yy(1, 3, IMAGE_DATA_TYPE, xx_d);
    Mat xy(3, 3, IMAGE_DATA_TYPE, xy_d);

    Mat dog = dog_pyr[kp.octave][(int)kp.angle];
    Mat Dxx = dog(cv::Range(kp.pt.x-1, kp.pt.x+2), cv::Range(kp.pt.y,   kp.pt.y+1)).mul(xx);
    Mat Dyy = dog(cv::Range(kp.pt.x,   kp.pt.x+1), cv::Range(kp.pt.y-1, kp.pt.y+2)).mul(yy);
    Mat Dxy = dog(cv::Range(kp.pt.x-1, kp.pt.x+2), cv::Range(kp.pt.y-1, kp.pt.y+2)).mul(xy);

    double Dxx_sum = std::accumulate(Dxx.begin<image_t>(), Dxx.end<image_t>(), 0.0);
    double Dyy_sum = std::accumulate(Dyy.begin<image_t>(), Dyy.end<image_t>(), 0.0);
    double Dxy_sum = std::accumulate(Dxy.begin<image_t>(), Dxy.end<image_t>(), 0.0);

    // Compute the trace and the determinant of the Hessian.
    double Tr_H = Dxx_sum + Dyy_sum;
    double Det_H = Dxx_sum*Dyy_sum - std::pow(Dxy_sum, 2);
    if (Det_H < EPSILON) return -1;

    // Compute the ratio of the principal curvatures.
    return std::pow(Tr_H, 2)/Det_H;
}

}
}
