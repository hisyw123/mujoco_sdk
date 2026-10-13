cmake_minimum_required(VERSION 3.18)
project(gym_articulated_actuator_net LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Fallbacks for this machine; override with -DCMAKE_PREFIX_PATH="<raisim>;<libtorch>" elsewhere.
list(APPEND CMAKE_PREFIX_PATH /root/lib/raisimLib/raisim/linux /root/lib/libtorch)

find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)

add_executable(augmented_sim src/augmented_sim.cpp)
target_link_libraries(augmented_sim PRIVATE raisim::raisim Eigen3::Eigen ${TORCH_LIBRARIES} pthread)
target_compile_options(augmented_sim PRIVATE ${TORCH_CXX_FLAGS})
