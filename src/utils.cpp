#include "sift.hpp" 

cv::Mat sift::downSample(Mat &image) {
    Mat out(image.rows/2, image.cols /2, IMAGE_DATA_TYPE);
    auto inserter = out.begin<image_t>();
    for (auto itr = image.begin<image_t>(); itr != image.end<image_t>(); itr++++, inserter++)
    {
        *inserter = *itr;
    }
    return out;
}