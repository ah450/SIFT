
list(APPEND SRCS )
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})


find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(${EXECUTABLE_NAME} ${SRCS} main.cpp)
target_link_libraries(${EXECUTABLE_NAME} ${OPENCV_LIBRARIES} ${Boost_LIBRARIES})

add_executable(${TEST_EXEC_NAME} ${SRCS} test.cpp)
target_link_libraries(${TEST_EXEC_NAME} ${OPENCV_LIBRARIES} ${Boost_LIBRARIES})
