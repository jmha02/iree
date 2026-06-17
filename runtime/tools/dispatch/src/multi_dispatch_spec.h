#ifndef IREE_MULTI_DISPATCH_SPEC_H_
#define IREE_MULTI_DISPATCH_SPEC_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/hal/api.h"
#include "iree/hal/local/executable_library.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct multi_dispatch_buffer_desc_t {
  void* data;
  size_t length;
} multi_dispatch_buffer_desc_t;

typedef struct multi_dispatch_output_desc_t {
  uint32_t element_type;
  const char* element_type_label;
  size_t element_size;
  size_t element_count;
  size_t shape_rank;
  const int64_t* shape_dims;
  multi_dispatch_buffer_desc_t buffer;
} multi_dispatch_output_desc_t;

typedef struct multi_dispatch_desc_t {
  const char* label;
  uint32_t export_ordinal;
  uint32_t workgroup_size[3];
  uint32_t workgroup_count[3];
  uint32_t max_concurrency;
  const uint32_t* constants;
  size_t constant_count;
  void* const* binding_ptrs;
  const size_t* binding_lengths;
  size_t binding_count;
  size_t output_count;
  const multi_dispatch_output_desc_t* outputs;
  bool print_outputs;
  size_t print_limit;
} multi_dispatch_desc_t;

typedef struct multi_dispatch_bundle_spec_t {
  const iree_hal_executable_library_header_t** (*query_fn)(
      iree_hal_executable_library_version_t max_version,
      const iree_hal_executable_environment_v0_t* environment);
  size_t dispatch_count;
  const multi_dispatch_desc_t* dispatches;
} multi_dispatch_bundle_spec_t;

const multi_dispatch_bundle_spec_t* multi_dispatch_acquire_spec(void);

#ifdef __cplusplus
}
#endif

#endif  // IREE_MULTI_DISPATCH_SPEC_H_
