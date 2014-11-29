
list(APPEND SRCS utils.cpp)
list(APPEND SRCS scale_space.cpp)
list(APPEND SRCS internal.cpp)
list(APPEND SRCS orientations.cpp)
list(APPEND SRCS find_interest_points.cpp)
list(APPEND TEST_SRCS test/test.cpp)
list(APPEND TEST_SRCS test/utils_tests.cpp)
list(APPEND TEST_SRCS test/buildGaussianPyramid_test.cpp)
list(APPEND TEST_SRCS test/dog_tests.cpp)
list(APPEND TEST_SRCS test/extrema_tests.cpp)
list(APPEND TEST_SRCS test/orientations_tests.cpp)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake/")
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(TBB REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIRS})
include_directories(".")

add_executable(${EXECUTABLE_NAME} ${SRCS} main.cpp)
target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${TBB_LIBRARIES})

add_executable(${TEST_EXEC_NAME} ${SRCS}  ${TEST_SRCS} )
target_link_libraries(${TEST_EXEC_NAME} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${TBB_LIBRARIES})
