#include "sift.hpp" 


/**
 * @brief Downsamples an image by a factor of 2
 * @details Skips odd rows/columns
 * 
 * @param image input to downsaple
 * @return downsampled image
 */
cv::Mat sift::downSample(Mat &image) {
    Mat out(image.rows/2, image.cols /2, IMAGE_DATA_TYPE);

    for (int row = 0; row < image.rows; row+=2 ) {
        if (row % 2 == 0) {
            for (int column = 0; column < image.cols; column+=2) {
                if (column % 2 == 0)
                    out.at<sift::image_t>(row/2, column/2) = image.at<sift::image_t>(row, column);
            }
        }
    }

    return out;
}