add_library(tinitorch_core STATIC
    ttcore/dtype.cpp
    ttcore/ops.cpp
    ttcore/tensor_impl.cpp
)

target_include_directories(tinitorch_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

nanobind_add_module(_C binding.cpp)
target_link_libraries(_C PRIVATE tinitorch_core)

set_target_properties(_C PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/tinitorch"
)
