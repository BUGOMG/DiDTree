cmake_minimum_required(VERSION 3.21)
#设置C编译器  
set(CMAKE_C_FLAGS "-g -Wall  -I D:\\code\\mingw64\\include -L D:\\code\\mingw64\\lib")# set(CMAKE_CXX_COMPILER "g++")#设置C++编译器  

set(CMAKE_CXX_FLAGS "-g -Wall  -I D:\\code\\mingw64\\include -L D:\\code\\mingw64\\lib")

PROJECT(DiDTree CXX)  


SET(BOOST_ROOT "D:\\code\\boost")
SET(boost_log_FOUND TRUE)
SET(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_RELEASE_LIBS OFF)
find_package(Boost 1.77 REQUIRED
            COMPONENTS log)

SET(ARMADILLO_INCLUDE_DIR "D:\\code\\armadillo\\include")
SET(BOOST_INCLUDE_DIR "D:\\code\\boost\\include\\boost-1_77")
SET(DiD_INCLUDE_DIR "include")

SET(Boost_LIBRARY_DIRS "D:\\code\\boost\\lib")


INCLUDE_DIRECTORIES(${ARMADILLO_INCLUDE_DIR} ${DiD_INCLUDE_DIR})
# SET(SRC_MAIN src/main.cpp)
# SET(SRC_LIST src/dataset.cpp)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
AUX_SOURCE_DIRECTORY(src SRC_LIST)

ADD_EXECUTABLE(DiDTree ${SRC_LIST})

if(Boost_FOUND)
    message("find boost library")
    include_directories(${BOOST_INCLUDE_DIR})
    TARGET_LINK_LIBRARIES(DiDTree pthread Boost::log ${Boost_LIBRARY_DIRS})
endif()