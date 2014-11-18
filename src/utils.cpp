#include "sift.hpp" 
#include <iterator>

cv::Mat sift::downSample(Mat &image) {
    Mat out(image.rows/2, image.cols /2, IMAGE_DATA_TYPE);
    auto reader = image.begin<image_t>();
    for (auto itr = out.begin<image_t>(); itr != out.end<image_t>(); itr++ )
    {        
        *itr = *reader;
        std::advance(reader, 2);
    }
    return out;
}