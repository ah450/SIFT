#include "sift.hpp"
#include <boost/program_options.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>

    
namespace po = boost::program_options;
int main(int argc, char const *argv[]) {
    po::options_description desc("options");
    desc.add_options()
        ("help,h", "produce this help message")
        ("input,i", po::value<std::vector<std::string>>()->composing(),
        "input image or folder to be searched recursively")
    ;
    po::positional_options_description pos_desc;
    pos_desc.add("input", -1); // all positional arguments will be input

    po::variables_map vm;
    po::store(po::command_line_parser(argc,
        argv).options(desc).positional(pos_desc).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return EXIT_SUCCESS;
    }

    if (vm.count("input")){
        for (const auto& image_path : vm["input"].as<std::vector<std::string>>()) {
            // first load image
            cv::Mat image_orig_format = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
            cv::Mat image; // proper format
            image_orig_format.convertTo(image, sift::IMAGE_DATA_TYPE);
            if (!image.data){
                std::cerr << "Could not load image " + image_path << std::endl;
                return EXIT_FAILURE;
            }
            const std::string orig_name = image_path + " Original";
            cv::namedWindow(orig_name, cv::WINDOW_AUTOSIZE);
            cv::imshow(orig_name, image);
            std::cout << "Building Gauss Pyramid" << std::endl;
            std::vector<std::vector<cv::Mat>> gaus_pyr;
            sift::buildGaussianPyramid(image, gaus_pyr, 2);
            std::cout << "Building DoG pyramid" << std::endl;
            std::vector<std::vector<cv::Mat>>  dog_pyr = sift::buildDogPyr(gaus_pyr);
            std::vector<cv::KeyPoint> keypoints;
            // Get keypoints
             std::cout << "Computing Keypoints" << std::endl;
            sift::getScaleSpaceExtrema(dog_pyr, keypoints);
            // Assign orientation, ignore histogram for now
            std::vector<cv::Mat> images;
            for (std::size_t i = 0; i < gaus_pyr.size(); i++) {
                images.emplace_back(gaus_pyr[i][0]);
            }
            sift::computeOrientationHist(images, keypoints);
            // Draw keypoints
            cv::Mat kp_image;
            cv::drawKeypoints(image, keypoints, kp_image);
            const std::string kp_name = image_path + " Keypoints";
            cv::namedWindow(kp_name, cv::WINDOW_AUTOSIZE);
            cv::imshow(kp_name, kp_image);
            // Clean them
            sift::cleanPoints(image, dog_pyr, keypoints);
            std::cout << "After cleaning" << std::endl;
            cv::Mat kp_clean_image;
            cv::drawKeypoints(image, keypoints, kp_clean_image);
            const std::string clean_kp_name = image_path + " Clean Keypoints";
            cv::namedWindow(clean_kp_name, cv::WINDOW_AUTOSIZE);
            cv::imshow(clean_kp_name, kp_clean_image);
            cv::waitKey(0);
        }
    }else {
        std::cout << "No input given\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
