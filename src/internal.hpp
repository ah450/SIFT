#pragma once
#include <sift.hpp>
#include <utility>
#include <cstddef>
#include <cmath>



namespace sift {
namespace internal{

typedef std::pair<int, int> point;

vector<point> find_local_extremas(const Mat &lower_dog,
                                  const Mat &current_dog, const Mat &upper_dog);



inline auto compute_octave_sigma(const int ocatave_number) -> decltype(GAUSSIAN_PYR_SIGMA0) {
    return std::sqrt(std::pow(2, ocatave_number) * GAUSSIAN_PYR_SIGMA0 * 
        std::pow(GAUSSIAN_PYR_K, 2));
}


/**
 * @brief Represents Rectangular area.
 * @detail Either a rectangle whose size is a parameter or one for 
 * a gaussian kernel based on scale value with a minimum of 5x5.
 */
struct Neighbourhood {
    int kernel_size, row_start, row_end, col_start, col_end;

    /**
     * @brief Constructs a Neighbourhood for a gaussian kernel
     * @param num_rows Total number of rows in parent region
     * @param num_columns Total number of columns in parent region
     * @param row center row value
     * @param col center col value
     * @param scale gauss sigma
     * @param min minimum kernel size, defaults to 5.
     */
    Neighbourhood(int num_rows, int num_columns, int row, int col,
                  double scale, double min = 5.0) {
        kernel_size = std::max(3 * scale, min);
        if (kernel_size % 2  == 0) {
            kernel_size += 1;
        }
        int kernel_half = kernel_size / 2;
        calc_rect(num_rows, num_columns, row, col, kernel_half);
    }

    /**
     * @brief Calculates a size * size rectangle  around row, col.
     */
    Neighbourhood(int num_rows, int num_columns, int row, int col, int half_size) {
        calc_rect(num_rows, num_columns, row, col, half_size);
    }

private:
    void calc_rect(int num_rows, int num_columns, int row, int col, int half_size){
        row_start = row > half_size ? row - half_size : 0;
        row_end = row + half_size < num_rows - 1 ? row + half_size + 1 : num_rows;
        col_start = col > half_size ? col - half_size : 0;
        col_end = col + half_size < num_columns - 1 ? col + half_size + 1 :
                  num_columns;
    }

};

}
}