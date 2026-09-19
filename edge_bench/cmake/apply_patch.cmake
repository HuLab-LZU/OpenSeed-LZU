# Apply a git patch to a fetched source tree, skipping it when it is already applied.
#
#   cmake -DPATCH_FILE=<file.patch> -DWORKDIR=<source dir> -P apply_patch.cmake
#
# Used as the FetchContent PATCH_COMMAND for MNN so that local fixes survive a fresh
# _deps checkout.  A failure only warns: mnn_bench defaults to a precision that is
# correct even without the patch, so a modified MNN must not break the build.

if(NOT DEFINED PATCH_FILE OR NOT DEFINED WORKDIR)
    message(FATAL_ERROR "apply_patch.cmake requires -DPATCH_FILE=<patch> -DWORKDIR=<dir>")
endif()
if(NOT EXISTS "${PATCH_FILE}")
    message(FATAL_ERROR "patch file not found: ${PATCH_FILE}")
endif()

execute_process(COMMAND git apply --reverse --check "${PATCH_FILE}"
                WORKING_DIRECTORY "${WORKDIR}"
                OUTPUT_QUIET ERROR_QUIET
                RESULT_VARIABLE already_applied)

if(already_applied EQUAL 0)
    message(STATUS "patch already applied, skipping: ${PATCH_FILE}")
    return()
endif()

execute_process(COMMAND git apply "${PATCH_FILE}"
                WORKING_DIRECTORY "${WORKDIR}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err
                RESULT_VARIABLE rc)

if(rc EQUAL 0)
    message(STATUS "applied ${PATCH_FILE} to ${WORKDIR}")
else()
    message(WARNING
        "could not apply ${PATCH_FILE} to ${WORKDIR}.\n"
        "  stdout: ${out}\n  stderr: ${err}\n"
        "  MNN's CUDA fp16 Softmax stays broken: --precision low returns NaN for\n"
        "  attention models. mnn_bench defaults to --precision normal, which is fine.")
endif()
