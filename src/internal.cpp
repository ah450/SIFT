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
    // find window size, ends are one past.
    int row_start = row > 0 ? row - 1 : row;
    int row_end = row + 1 < current.rows ? row + 2 : row + 1;
    int col_end = col + 1 < current.cols ? col + 2 : col + 1;
    int col_start = col > 0 ? col - 1 : col;
    sift::image_t center = current.at<sift::image_t>(row, col);
    sift::image_t max(center), min(center);    
    // check in current
    max_min(row_start, row_end, col_start, col_end, current, max, min);
    // check in upper
    max_min(row_start, row_end, col_start, col_end, upper, max, min);
    // cehck in lower
    max_min(row_start, row_end, col_start, col_end, lower, max, min);
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

}
}