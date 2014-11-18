cmake_minimum_required(VERSION 2.8)
project(Sift)
set(EXECUTABLE_NAME "sift")
set(TEST_EXEC_NAME "sift-tests")
if(CMAKE_COMPILER_IS_GNUCXX)
    ADD_DEFINITIONS(
        -std=c++11
        -Wall
        -Wextra
        -Werror
        )
    # Yes discard default values
    set(CMAKE_CXX_FLAGS_RELEASE "-flto -O3 -fmerge-all-constants -fgcse-sm -fgcse-las")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto -s")
endif()
add_subdirectory(src)
