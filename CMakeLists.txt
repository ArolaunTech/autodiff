cmake_minimum_required(VERSION 3.25)

project(autodiff LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE adsrc
	"test/*.h"
	"test/*.cpp"
)

add_library(ad INTERFACE)
target_include_directories(ad INTERFACE .)

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag(-Wall temp)

if(temp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

check_cxx_compiler_flag(-O3 temp2)

if(temp2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

add_executable(run ${adsrc})
target_link_libraries(run PRIVATE ad)