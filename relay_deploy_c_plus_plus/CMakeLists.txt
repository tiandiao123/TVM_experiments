# set minimum cmake version
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# project name and language
project(recipe-03 LANGUAGES CXX)

# require C++11
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)


# main code
add_executable(relay_build_module_test relay_build_module_test.cc)

# example library
include_directories(SYSTEM /root/std_tvm/3rdparty/dlpack/include/)
include_directories(SYSTEM /root/std_tvm/3rdparty/dmlc-core/include/)
include_directories(SYSTEM /root/std_tvm/include)


link_directories(/root/std_tvm/build)


# we will use the network to fetch Google Test sources
# make it possible to disable unit tests when not on network
option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")

if(ENABLE_UNIT_TESTS)
  include(fetch_git_repo.cmake)

  fetch_git_repo(
    googletest
    ${CMAKE_BINARY_DIR}/_deps
    https://github.com/google/googletest.git
    release-1.8.0
  )

  # Prevent GoogleTest from overriding our compiler/linker options
  # when building with Visual Studio
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  # Prevent GoogleTest from using PThreads
  set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)

  # adds the targers: gtest, gtest_main, gmock, gmock_main
  add_subdirectory(
    ${googletest_SOURCE_DIR}
    ${googletest_BINARY_DIR}
    )

  # Silence std::tr1 warning on MSVC
  if(MSVC)
    foreach(_tgt gtest gtest_main gmock gmock_main)
      target_compile_definitions(${_tgt}
        PRIVATE
          "_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING"
        )
    endforeach()
  endif()



  target_link_libraries(relay_build_module_test
    PRIVATE
      gtest_main
      "/root/std_tvm/build/libtvm.so"
    )
  

  enable_testing()

  add_test(
    NAME google_test
    COMMAND $<TARGET_FILE:relay_build_module_test>
    )
endif()
