cmake_minimum_required(VERSION 2.8)
project(Sift)
set(EXECUTABLE_NAME "sift")
set(TEST_EXEC_NAME "sift-tests")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
    ADD_DEFINITIONS(
        -std=c++11
        -Wall
        -Wextra
        -Werror
        )
    # Yes discard default values
    set(CMAKE_CXX_FLAGS_RELEASE "-flto -O3 -march=native -fmerge-all-constants -fgcse-sm -fgcse-las")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto -s")
elseif ( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" )
    add_definitions(
        /Wall
        /WX        
        )
    set(CMAKE_CXX_FLAGS_RELEASE "/Ox /Oi /arch:SSE2")
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_definitions(-DWE_HAVE_CONST_MATH)    
endif()
add_subdirectory(src)
