#include "test.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <iterator>


TEST_CASE("Down sampling", "[utilities]") {
    cv::Mat m(8, 8, sift::IMAGE_DATA_TYPE, img8x8_data);
    cv::Mat downsampled;
    SECTION("Exception safety") {
        
        REQUIRE_NOTHROW( downsampled = sift::downSample(m) );
    }

    SECTION("Matrix size") {
        downsampled = sift::downSample(m);
        REQUIRE( downsampled.rows == m.rows/2 );
        REQUIRE( downsampled.cols == m.cols/2 );
    }

    SECTION("Actual Data check") {
        downsampled = sift::downSample(m);
        auto downsampled_fixture = cv::Mat(4, 4, 
            sift::IMAGE_DATA_TYPE, img4x4_data);
        REQUIRE( std::equal(downsampled_fixture.begin<sift::image_t>(),
            downsampled_fixture.end<sift::image_t>(), 
            downsampled.begin<sift::image_t>()) );
        
    }
}


TEST_CASE("find_local_extremas", "[extrema]") {

    std::vector<sift::internal::point> points;
    cv::Mat lower(9, 9, sift::IMAGE_DATA_TYPE, lower_9x9_extrema), 
        current(9, 9, sift::IMAGE_DATA_TYPE, current_9x9_extrema),
        upper(9, 9, sift::IMAGE_DATA_TYPE, upper_9x9_extrema);

    REQUIRE_NOTHROW( points = sift::internal::find_local_extremas(lower, current,
        upper) );

    SECTION("Size"){
        REQUIRE(points.size() == extremas.size());
    }

    SECTION("Verify extremas") {
        for(auto& p : extremas) {
            REQUIRE(std::find(points.begin(), points.end(), p) != points.end());
        }
    }

}