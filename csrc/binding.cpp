#include <nanobind/nanobind.h>
#include "ttcore/ops.h"

namespace nb = nanobind;

NB_MODULE(_C, m) {
    m.def("add", &add);
}
