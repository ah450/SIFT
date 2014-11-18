#include "test.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <iostream>
#include <iterator>


TEST_CASE("Down sampling", "[utilities]") {
    cv::Mat m(8, 8, sift::IMAGE_DATA_TYPE, img8x8_data);
    cv::Mat downsampled;
    SECTION("NO THROW") {
        
        REQUIRE_NOTHROW(downsampled = sift::downSample(m));
    }

    SECTION("Matrix size") {
        downsampled = sift::downSample(m);
        REQUIRE( downsampled.rows == m.rows/2 );
        REQUIRE( downsampled.cols == m.cols/2 );
    }

    SECTION("Actual Data check") {
        downsampled = sift::downSample(m);
        auto downsampled_fixture = cv::Mat(4, 4, 
            sift::IMAGE_DATA_TYPE, image4x4_data);
        REQUIRE( std::equal(downsampled_fixture.begin<sift::image_t>(),
            downsampled_fixture.end<sift::image_t>(), 
            downsampled.begin<sift::image_t>()) );
        
    }
}