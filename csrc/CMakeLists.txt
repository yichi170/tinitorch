add_library(tinitorch_core STATIC
    ttcore/dtype.cpp
    ttcore/tensor_impl.cpp
    ttcore/dispatch.cpp
    ttcore/kernels/cpu_elementwise.cpp
)

# Required for linking static library into shared library on Linux
set_target_properties(tinitorch_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(tinitorch_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

nanobind_add_module(_C binding.cpp)
target_link_libraries(_C PRIVATE tinitorch_core)

set_target_properties(_C PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/tinitorch"
)
