#include "sift.hpp"
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>

extern int nOctaves;
    
namespace po = boost::program_options;

void trace(const std::string &image_path, cv::Mat &image) {
    const std::string orig_name = image_path + " Original";
    namedWindow(orig_name, cv::WINDOW_AUTOSIZE);
    imshow(orig_name, image);
    std::cout << "Building Gauss Pyramid" << std::endl;
    std::vector<std::vector<cv::Mat>> gaus_pyr;
    sift::buildGaussianPyramid(image, gaus_pyr, nOctaves);
    for(std::size_t i = 0; i < gaus_pyr.size(); i++) {
                for(std::size_t j = 0; j < gaus_pyr[i].size(); j++){
                    const std::string name = std::string("Gauss[") +
                        boost::lexical_cast<std::string>(i) + "][" +
                        boost::lexical_cast<std::string>(j) + "]";
                    namedWindow(name, cv::WINDOW_AUTOSIZE);
                    imshow(name ,gaus_pyr[i][j]);
                }
            }

    std::cout << "Building DoG pyramid" << std::endl;
    std::vector<std::vector<cv::Mat>>  dog_pyr = sift::buildDogPyr(gaus_pyr);
    for(std::size_t i = 0; i < dog_pyr.size(); i++) {
                for(std::size_t j = 0; j < dog_pyr[i].size(); j++){
                    const std::string name = std::string("DoG[") +
                        boost::lexical_cast<std::string>(i) + "][" +
                        boost::lexical_cast<std::string>(j) + "]";
                    namedWindow(name, cv::WINDOW_AUTOSIZE);
                    imshow(name ,dog_pyr[i][j]);
                }
            }
    std::vector<cv::KeyPoint> keypoints;
    // Get keypoints
    std::cout << "Computing Keypoints" << std::endl;
    sift::getScaleSpaceExtrema(dog_pyr, keypoints);
    // Assign orientation, ignore histogram for now
    std::vector<cv::Mat> images;
    for (std::size_t i = 0; i < gaus_pyr.size(); i++) {
                images.emplace_back(gaus_pyr[i][0]);
            }
    // Clean them
    std::cout << "Cleaning points" << std::endl;
    sift::cleanPoints(image, dog_pyr, keypoints);
    std::cout << "Assigning orientations" << std::endl;
    sift::computeOrientationHist(images, keypoints);
    // Draw keypoints
    cv::Mat kp_image;
    cv::Mat image_char;
    image.convertTo(image_char, CV_8U);
    drawKeypoints(image_char, keypoints, kp_image);
    const std::string kp_name = image_path + " Keypoints";
    namedWindow(kp_name, cv::WINDOW_AUTOSIZE);
    imshow(kp_name, kp_image);

    std::cout << "Done" << std::endl;
    cv::waitKey(0);
}

int main(int argc, char const *argv[]) {
    po::options_description desc("options");
    desc.add_options()
        ("help,h", "produce this help message")
        ("input,i", po::value<std::vector<std::string>>()->composing(),
        "input image")
        ("trace,t", po::bool_switch(), "Show trace of each step")
        ("nOctaves,n", po::value<int>(&nOctaves)->default_value(2), "Number of octaves")
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

            if(vm["trace"].as<bool>()) {
                trace(image_path, image);
            }else {
                std::vector<cv::KeyPoint> kps;
                sift::findSiftInterestPoint(image, kps);
            }

        }
    }else {
        std::cout << "No input given\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
