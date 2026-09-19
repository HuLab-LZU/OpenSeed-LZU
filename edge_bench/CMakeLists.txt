cmake_minimum_required(VERSION 3.18)
project(mnn_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# MSVC: MNN adds /Zi (PDB debug info) for Release. With parallel builds, multiple
# cl.exe processes can contend for the same .pdb and fail with C1041; /FS forces
# synchronous PDB writes and avoids the race.
if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /FS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FS")
endif()

# MNN source can be downloaded via FetchContent. Use a mirror if needed.
set(MNN_TAG "3.6.1" CACHE STRING "MNN git tag to fetch")
set(MNN_GIT_REPO
    "https://gh-proxy.org/https://github.com/alibaba/MNN.git"
    CACHE STRING "MNN git repository URL")

# Build MNN CV (image codecs) for MNN::CV::imread.
set(MNN_BUILD_OPENCV ON CACHE BOOL "Build MNN OpenCV API" FORCE)
set(MNN_IMGCODECS ON CACHE BOOL "Build MNN image codecs" FORCE)
set(MNN_SUPPORT_BF16 ON CACHE BOOL "Enable MNN BF16 support" FORCE)
set(MNN_SSE_USE_FP16_INSTEAD ON CACHE BOOL "Use FP16 instead of FP32 for SSE" FORCE)
set(MNN_SEP_BUILD OFF CACHE BOOL "Build MNN backends as separate shared libraries" FORCE)

# Avoid building MNN demos / tests / training for the benchmark.
set(MNN_BUILD_TRAIN OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_CONVERTER OFF CACHE BOOL "" FORCE)

# ---------------------------------------------------------------------------
# Automatic backend selection.
# ---------------------------------------------------------------------------

# Metal -> Apple / macOS
if(NOT DEFINED MNN_METAL)
    if(APPLE)
        set(MNN_METAL ON CACHE BOOL "Enable MNN Metal backend" FORCE)
    else()
        set(MNN_METAL OFF CACHE BOOL "Enable MNN Metal backend" FORCE)
    endif()
endif()

# CUDA -> Linux/Windows x86 machines that have the CUDA toolkit
if(NOT DEFINED MNN_CUDA)
    if((CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Windows") AND NOT APPLE)
        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            set(MNN_CUDA ON CACHE BOOL "Enable MNN CUDA backend" FORCE)
            set(MNN_CUDA_BF16 ON CACHE BOOL "Enable MNN CUDA BF16 support" FORCE)
            # When CUDA is enabled, only compile for the local GPU architecture by default.
            # This avoids CUDA >= 13 failing on old compute_60/61/... targets.
            set(MNN_CUDA_NATIVE_ARCH ON CACHE BOOL "Only emit gencode for local GPU arch" FORCE)
        else()
            set(MNN_CUDA OFF CACHE BOOL "Enable MNN CUDA backend" FORCE)
        endif()
    else()
        set(MNN_CUDA OFF CACHE BOOL "Enable MNN CUDA backend" FORCE)
    endif()
endif()

# OpenCL -> Linux / Android / Darwin with an OpenCL implementation found
if(NOT DEFINED MNN_OPENCL)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        find_package(OpenCL)
        if(OpenCL_FOUND)
            set(MNN_OPENCL ON CACHE BOOL "Enable MNN OpenCL backend" FORCE)
        else()
            set(MNN_OPENCL OFF CACHE BOOL "Enable MNN OpenCL backend" FORCE)
        endif()
    else()
        set(MNN_OPENCL OFF CACHE BOOL "Enable MNN OpenCL backend" FORCE)
    endif()
endif()

# Vulkan is not required by the benchmark; keep it off unless explicitly enabled.
if(NOT DEFINED MNN_VULKAN)
    set(MNN_VULKAN OFF CACHE BOOL "Enable MNN Vulkan backend" FORCE)
endif()

# nlohmann/json for JSON parsing/saving
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(json)

# MNN 3.6.1 detects fp16 in the CUDA Softmax execution by inspecting the tensor type
# instead of the backend precision. The CUDA backend keeps float tensors declared as
# fp32 and only halves the buffer, so the fp32 kernel ends up reading an fp16 buffer
# and every Softmax output becomes NaN - which wipes out all attention models
# (ViT / Swin / MobileViT / MobileNetV4-hybrid) whenever precision is Low.
# patches/mnn_cuda_softmax_fp16.patch fixes it; it is applied idempotently after fetch.
FetchContent_Declare(
    MNN
    GIT_REPOSITORY ${MNN_GIT_REPO}
    GIT_TAG ${MNN_TAG}
    GIT_PROGRESS TRUE
    PATCH_COMMAND ${CMAKE_COMMAND}
        -DPATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/patches/mnn_cuda_softmax_fp16.patch
        -DWORKDIR=<SOURCE_DIR>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply_patch.cmake
)
FetchContent_MakeAvailable(MNN)

message(STATUS "MNN backends: Metal=${MNN_METAL} CUDA=${MNN_CUDA} OpenCL=${MNN_OPENCL} Vulkan=${MNN_VULKAN}")

add_executable(mnn_bench src/mnn_bench.cc)
# MNN FetchContent does not always propagate its public include dirs.
target_include_directories(mnn_bench PRIVATE ${MNN_INCLUDE_DIRS} ${mnn_SOURCE_DIR}/tools/cv/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include)  # vendored indicators progress-bar headers
target_compile_definitions(mnn_bench PRIVATE MNN_USE_OPENCV)
target_link_libraries(mnn_bench PRIVATE MNN nlohmann_json::nlohmann_json)
# In non-separated builds (MNN_SEP_BUILD=OFF), MNN_Express / MNNOpenCV are
# OBJECT libraries already folded into MNN (dll or static). Linking them again
# makes the executable duplicate those symbols (LNK2005 on MSVC).
if(MNN_SEP_BUILD)
    target_link_libraries(mnn_bench PRIVATE ${MNN_LIBS})
endif()

# On Windows, MNN.dll is built into _deps/mnn-build/<Config>/. Copy it next to
# mnn_bench.exe so the process can load it without PATH tweaks.
if(WIN32 AND MNN_BUILD_SHARED_LIBS)
    add_custom_command(TARGET mnn_bench POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:MNN>" "$<TARGET_FILE_DIR:mnn_bench>")
endif()
