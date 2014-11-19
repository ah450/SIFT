
list(APPEND SRCS utils.cpp)
list(APPEND SRCS scale_space.cpp)
list(APPEND TEST_SRCS test/test.cpp)
list(APPEND TEST_SRCS test/utils_tests.cpp)
list(APPEND TEST_SRCS test/buildGaussianPyramid_test.cpp)
list(APPEND TEST_SRCS test/dog_tests.cpp)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})


find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

if ( NOT OpenCV_FOUND )
    message(FATAL_ERROR "Package OpenCV required but not found!")
endif()

if ( NOT Boost_FOUND )
    message(FATAL_ERROR "Package Boost required but not found!")
endif()

include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(".")

add_executable(${EXECUTABLE_NAME} ${SRCS} main.cpp)
target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

add_executable(${TEST_EXEC_NAME} ${SRCS}  ${TEST_SRCS} )
target_link_libraries(${TEST_EXEC_NAME} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
