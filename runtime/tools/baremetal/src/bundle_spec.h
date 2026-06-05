#ifndef iree_baremetal_BUNDLE_SPEC_H_
#define iree_baremetal_BUNDLE_SPEC_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_baremetal_ro_span_t {
  const uint8_t* data;
  iree_host_size_t data_length;
} iree_baremetal_ro_span_t;

typedef struct iree_baremetal_rw_span_t {
  uint8_t* data;
  iree_host_size_t data_length;
} iree_baremetal_rw_span_t;

typedef enum iree_baremetal_input_kind_e {
  IREE_BAREMETAL_INPUT_KIND_TENSOR = 0,
  IREE_BAREMETAL_INPUT_KIND_I32 = 1,
  IREE_BAREMETAL_INPUT_KIND_I64 = 2,
  IREE_BAREMETAL_INPUT_KIND_F32 = 3,
  IREE_BAREMETAL_INPUT_KIND_F64 = 4,
} iree_baremetal_input_kind_t;

typedef struct iree_baremetal_tensor_input_t {
  iree_hal_element_type_t element_type;
  iree_hal_encoding_type_t encoding_type;
  iree_host_size_t shape_rank;
  const iree_hal_dim_t* shape_dims;
  iree_baremetal_ro_span_t buffer;
} iree_baremetal_tensor_input_t;

typedef struct iree_baremetal_input_t {
  iree_baremetal_input_kind_t kind;
  union {
    iree_baremetal_tensor_input_t tensor;
    int32_t i32;
    int64_t i64;
    float f32;
    double f64;
  } value;
} iree_baremetal_input_t;

typedef struct iree_baremetal_output_t {
  iree_hal_element_type_t element_type;
  iree_hal_encoding_type_t encoding_type;
  iree_host_size_t shape_rank;
  const iree_hal_dim_t* shape_dims;
  iree_host_size_t element_count;
  iree_host_size_t element_size;
  const char* element_type_label;
  iree_baremetal_rw_span_t result_buffer;
} iree_baremetal_output_t;

typedef struct iree_baremetal_bundle_spec_t {
  iree_string_view_t entry_function;
  iree_baremetal_ro_span_t module_data;
  iree_host_size_t input_count;
  const iree_baremetal_input_t* inputs;
  iree_host_size_t output_count;
  const iree_baremetal_output_t* outputs;
  bool print_outputs;
  iree_host_size_t print_limit;
  int tooling_argc;
  char** tooling_argv;
} iree_baremetal_bundle_spec_t;

iree_status_t iree_baremetal_run(const iree_baremetal_bundle_spec_t* spec);
const iree_baremetal_bundle_spec_t* iree_baremetal_acquire_spec(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // iree_baremetal_BUNDLE_SPEC_H_
