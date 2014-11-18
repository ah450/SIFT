#include <boost/program_options.hpp>
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

    }else {
        std::cout << "No input given\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
