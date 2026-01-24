cmake_minimum_required(VERSION 3.20)
project(slicot VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTING "Build tests" OFF)
option(SLICOT_VENDOR_OPENBLAS "Vendor OpenBLAS for wheel distribution" OFF)

# BLAS/LAPACK
if(SLICOT_VENDOR_OPENBLAS)
    include(${CMAKE_SOURCE_DIR}/cmake/BuildOpenBLAS.cmake)
    set(BLAS_LIBRARIES OpenBLAS::OpenBLAS)
    set(LAPACK_LIBRARIES OpenBLAS::OpenBLAS)
    set(SLC_FC_LOWER_US ON)
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)

    include(CheckFunctionExists)
    set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    check_function_exists(dgemm_ HAVE_DGEMM_UNDERSCORE)
    if(HAVE_DGEMM_UNDERSCORE)
        set(SLC_FC_LOWER_US ON)
    else()
        check_function_exists(dgemm HAVE_DGEMM_LOWER)
        if(HAVE_DGEMM_LOWER)
            set(SLC_FC_LOWER ON)
        else()
            set(SLC_FC_LOWER_US ON)
        endif()
    endif()
endif()

configure_file(
    ${PROJECT_SOURCE_DIR}/include/slicot_config.h.in
    ${PROJECT_BINARY_DIR}/include/slicot_config.h
)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_subdirectory(src)

# scikit-build-core integration
if(DEFINED SKBUILD)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

    execute_process(
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/generate_docstrings.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    file(GLOB WRAPPER_SOURCES ${CMAKE_SOURCE_DIR}/python/wrappers/py_*.c)

    python_add_library(_slicot MODULE
        ${CMAKE_SOURCE_DIR}/python/slicot_module.c
        ${WRAPPER_SOURCES}
        WITH_SOABI
    )

    target_include_directories(_slicot PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}/python
        ${Python_NumPy_INCLUDE_DIRS}
    )

    target_link_libraries(_slicot PRIVATE slicot)

    install(TARGETS _slicot LIBRARY DESTINATION slicot)
    install(FILES
        ${CMAKE_SOURCE_DIR}/python/slicot/__init__.py
        ${CMAKE_SOURCE_DIR}/python/slicot/py.typed
        DESTINATION slicot
        OPTIONAL
    )
endif()
