cmake_minimum_required(VERSION 3.24)

# GPU architectures: by default the GPU in this machine ("native"; this must be
# set before project(), which otherwise picks the compiler's default, and code
# for an older architecture would be JIT-compiled at every start, slowly).
# For distribution builds pass e.g. -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120".
# Compute capability 7.5 (Turing) is the oldest supported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
endif()

project(gpuecm2 VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- dependencies -------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
# GMP (host only). On Windows use e.g. vcpkg's gmp and point GMP_ROOT at it.
find_path(GMP_INCLUDE_DIR gmp.h HINTS ${GMP_ROOT} PATH_SUFFIXES include)
find_library(GMP_LIBRARY NAMES gmp libgmp mpir HINTS ${GMP_ROOT} PATH_SUFFIXES lib)
if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
  message(FATAL_ERROR "GMP not found; set GMP_ROOT")
endif()
add_library(gmp_lib UNKNOWN IMPORTED)
set_target_properties(gmp_lib PROPERTIES IMPORTED_LOCATION "${GMP_LIBRARY}"
                                         INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}")

# CGBN (header-only, vendored)
add_library(cgbn INTERFACE)
target_include_directories(cgbn SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cgbn/include)

# --- core library ---------------------------------------------------------------
add_library(gpuecm2core STATIC
  src/host/version.cpp
  src/host/expr.cpp
  src/host/resume.cpp
  src/host/curve.cpp
  src/host/params.cpp
  src/host/roots_plan.cpp
  src/gpu/rns.cu
  src/gpu/ntt.cu
  src/gpu/polymul.cu
  src/gpu/limbops.cu
  src/gpu/polyalg.cu
  src/gpu/roots.cu
  src/gpu/prod.cu
  src/api/stage2.cu
)
target_include_directories(gpuecm2core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(gpuecm2core PUBLIC gmp_lib cgbn CUDA::cudart_static)
# Development aid: compile the CGBN kernels for one modulus size only (e.g. 1024)
# to cut build times; the default builds all sizes (512 .. 4608 bits, see src/gpu/cgbn_sizes.h).
set(GPUECM2_ONLY_BITS "" CACHE STRING "Compile CGBN kernels for this BITS only (development)")
if(GPUECM2_ONLY_BITS)
  target_compile_definitions(gpuecm2core PUBLIC GPUECM2_ONLY_BITS=${GPUECM2_ONLY_BITS})
endif()
target_compile_definitions(gpuecm2core PUBLIC GPUECM2_VERSION="${PROJECT_VERSION}")
set_target_properties(gpuecm2core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                                             POSITION_INDEPENDENT_CODE ON)

# --- executable -------------------------------------------------------------------
add_executable(gpuecm2 src/cli/main.cpp)
target_link_libraries(gpuecm2 PRIVATE gpuecm2core)

# --- C interface (src/api/gpuecm2.h): libgpuecm2 ----------------------------------
# A shared library exporting only the gpuecm2_* functions (static on Windows).
if(WIN32)
  add_library(gpuecm2lib STATIC src/api/gpuecm2.cpp)
else()
  add_library(gpuecm2lib SHARED src/api/gpuecm2.cpp)
  target_link_options(gpuecm2lib PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/api/gpuecm2.map")
  set_target_properties(gpuecm2lib PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/api/gpuecm2.map
                                              VERSION ${PROJECT_VERSION} SOVERSION 0)
endif()
set_target_properties(gpuecm2lib PROPERTIES OUTPUT_NAME gpuecm2 PUBLIC_HEADER src/api/gpuecm2.h)
target_link_libraries(gpuecm2lib PRIVATE gpuecm2core PUBLIC gmp_lib)
target_include_directories(gpuecm2lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

include(GNUInstallDirs)
install(TARGETS gpuecm2 gpuecm2lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# --- tests ------------------------------------------------------------------------
enable_testing()
add_executable(test_smoke tests/test_smoke.cu)
target_link_libraries(test_smoke PRIVATE gpuecm2core)
add_test(NAME smoke COMMAND test_smoke)
add_executable(test_host tests/test_host.cpp)
target_link_libraries(test_host PRIVATE gpuecm2core)
add_test(NAME host COMMAND test_host)
add_executable(test_poly tests/test_poly.cu)
target_link_libraries(test_poly PRIVATE gpuecm2core)
add_test(NAME poly COMMAND test_poly --quick)
add_executable(test_polyalg tests/test_polyalg.cu)
target_link_libraries(test_polyalg PRIVATE gpuecm2core)
add_test(NAME polyalg COMMAND test_polyalg --quick)
add_executable(test_roots tests/test_roots.cu)
target_link_libraries(test_roots PRIVATE gpuecm2core)
add_test(NAME roots COMMAND test_roots)
add_executable(test_stage2 tests/test_stage2.cu)
target_link_libraries(test_stage2 PRIVATE gpuecm2core)
add_test(NAME stage2 COMMAND test_stage2)
add_executable(test_capi tests/test_capi.c)
target_link_libraries(test_capi PRIVATE gpuecm2lib)
add_test(NAME capi COMMAND test_capi)
add_executable(capi_resume tests/capi_resume.c)
target_link_libraries(capi_resume PRIVATE gpuecm2lib)
# known stage 2 hits (tests/data/stage2_hits.txt), via the program and via the C interface
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME hits COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_hits.py $<TARGET_FILE:gpuecm2>)
  add_test(NAME hits_capi COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_hits.py
                                  $<TARGET_FILE:capi_resume>)
  add_test(NAME input COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/input_tests.py
                              $<TARGET_FILE:gpuecm2>)
endif()

# --- benchmarks (bench/) -----------------------------------------------------------
add_executable(bench_roots bench/bench_roots.cu)
target_link_libraries(bench_roots PRIVATE gpuecm2core)
add_executable(bench_stage2 bench/bench_stage2.cu)
target_link_libraries(bench_stage2 PRIVATE gpuecm2core)
target_compile_definitions(bench_stage2 PRIVATE GPUECM2_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
add_executable(bench_conv bench/bench_conv.cu)
target_link_libraries(bench_conv PRIVATE gpuecm2core)
add_executable(bench_poly bench/bench_poly.cu)
target_link_libraries(bench_poly PRIVATE gpuecm2core)
