#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "multi_dispatch_spec.h"
#include "iree/hal/riscv_cycle_measure.h"

extern void __attribute__((noreturn)) tohost_exit(uintptr_t code);

static float bf16_to_f32(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } conv = {.u32 = (uint32_t)value << 16};
  return conv.f32;
}

static void print_decimal(float value) {
  if (value != value) {
    printf("nan");
    return;
  }
  if (value == (float)1.0 / 0.0) {
    printf("inf");
    return;
  }
  if (value == (float)-1.0 / 0.0) {
    printf("-inf");
    return;
  }
  printf("%0.6f", value);
}

static void print_shape(const multi_dispatch_output_desc_t* output) {
  if (!output || output->shape_rank == 0 || !output->shape_dims) {
    printf("[]");
    return;
  }
  printf("[");
  for (size_t i = 0; i < output->shape_rank; ++i) {
    if (i) printf("x");
    printf("%lld", (long long)output->shape_dims[i]);
  }
  printf("]");
}

static size_t output_element_count(const multi_dispatch_output_desc_t* output) {
  if (!output) return 0;
  if (output->element_count) return output->element_count;
  if (!output->element_size) return 0;
  return output->buffer.length / output->element_size;
}

static uint64_t hash_output_bytes(const multi_dispatch_output_desc_t* output) {
  if (!output || !output->buffer.data || output->buffer.length == 0) return 0;
  const uint8_t* data = (const uint8_t*)output->buffer.data;
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < output->buffer.length; ++i) {
    hash ^= (uint64_t)data[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

static void dump_output(const multi_dispatch_output_desc_t* output,
                        size_t limit) {
  if (!output || !output->buffer.data || output->buffer.length == 0) {
    printf("  (no data)\n");
    return;
  }
  size_t count = output_element_count(output);
  if (count == 0) {
    printf("  (empty tensor)\n");
    return;
  }
  if (limit == 0 || limit > count) limit = count;

  switch (output->element_type) {
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16: {
      const uint16_t* values = (const uint16_t*)output->buffer.data;
      for (size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] ", (unsigned long long)i);
        print_decimal(bf16_to_f32(values[i]));
        printf("\n");
      }
      break;
    }
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32: {
      const float* values = (const float*)output->buffer.data;
      for (size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] ", (unsigned long long)i);
        print_decimal(values[i]);
        printf("\n");
      }
      break;
    }
    case IREE_HAL_ELEMENT_TYPE_SINT_32: {
      const int32_t* values = (const int32_t*)output->buffer.data;
      for (size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] %" PRId32 "\n", (unsigned long long)i, values[i]);
      }
      break;
    }
    case IREE_HAL_ELEMENT_TYPE_UINT_32: {
      const uint32_t* values = (const uint32_t*)output->buffer.data;
      for (size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] %" PRIu32 "\n", (unsigned long long)i, values[i]);
      }
      break;
    }
    case IREE_HAL_ELEMENT_TYPE_SINT_16: {
      const int16_t* values = (const int16_t*)output->buffer.data;
      for (size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] %d\n", (unsigned long long)i, (int)values[i]);
      }
      break;
    }
    case IREE_HAL_ELEMENT_TYPE_UINT_16: {
      const uint16_t* values = (const uint16_t*)output->buffer.data;
      for (size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] %u\n", (unsigned long long)i, (unsigned)values[i]);
      }
      break;
    }
    case IREE_HAL_ELEMENT_TYPE_SINT_8: {
      const int8_t* values = (const int8_t*)output->buffer.data;
      for (size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] %d\n", (unsigned long long)i, (int)values[i]);
      }
      break;
    }
    case IREE_HAL_ELEMENT_TYPE_UINT_8: {
      const uint8_t* values = (const uint8_t*)output->buffer.data;
      for (size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] %u\n", (unsigned long long)i, (unsigned)values[i]);
      }
      break;
    }
    default:
      printf("  (printing for element type %u not implemented)\n",
             (unsigned)output->element_type);
      break;
  }
}

int main(void) {
  const multi_dispatch_bundle_spec_t* spec = multi_dispatch_acquire_spec();

#if defined(IREE_HAL_RISCV_MEASURE_CYCLES)
  iree_hal_cycle_runtime_start();
#endif

  iree_hal_executable_environment_v0_t environment;
  memset(&environment, 0, sizeof(environment));

  union {
    const iree_hal_executable_library_header_t** header;
    const iree_hal_executable_library_v0_t* v0;
  } lib;
  lib.header = spec->query_fn(IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
                               &environment);
  if (!lib.header || !*lib.header || !lib.v0) {
    printf("[IREE][error] executable library query failed\n");
    return 1;
  }

#if defined(IREE_HAL_RISCV_MEASURE_CYCLES)
  iree_hal_cycle_mark_vm_invoke();
#endif

  for (size_t di = 0; di < spec->dispatch_count; ++di) {
    const multi_dispatch_desc_t* d = &spec->dispatches[di];

    if (d->export_ordinal >= lib.v0->exports.count) {
      printf("[IREE][error] dispatch[%llu] export ordinal %u out of range (count=%u)\n",
             (unsigned long long)di, d->export_ordinal, lib.v0->exports.count);
      return 1;
    }
    const iree_hal_executable_dispatch_v0_t entry =
        lib.v0->exports.ptrs[d->export_ordinal];

    const iree_hal_executable_dispatch_state_v0_t dispatch_state = {
        .workgroup_size_x = d->workgroup_size[0],
        .workgroup_size_y = d->workgroup_size[1],
        .workgroup_size_z = (uint16_t)d->workgroup_size[2],
        .constant_count = (uint16_t)d->constant_count,
        .workgroup_count_x = d->workgroup_count[0],
        .workgroup_count_y = d->workgroup_count[1],
        .workgroup_count_z = (uint16_t)d->workgroup_count[2],
        .max_concurrency = (uint8_t)d->max_concurrency,
        .binding_count = (uint8_t)d->binding_count,
        .constants = d->constants,
        .binding_ptrs = d->binding_ptrs,
        .binding_lengths = d->binding_lengths,
    };

    iree_hal_executable_workgroup_state_v0_t workgroup_state = {0};

#if defined(IREE_HAL_RISCV_MEASURE_CYCLES)
    size_t cycle_slot = iree_hal_cycle_dispatch_begin();
    iree_hal_cycle_mark_elf_enter();
#endif

    for (uint32_t z = 0; z < dispatch_state.workgroup_count_z; ++z) {
      workgroup_state.workgroup_id_z = z;
      for (uint32_t y = 0; y < dispatch_state.workgroup_count_y; ++y) {
        workgroup_state.workgroup_id_y = y;
        for (uint32_t x = 0; x < dispatch_state.workgroup_count_x; ++x) {
          workgroup_state.workgroup_id_x = x;
          entry(&environment, &dispatch_state, &workgroup_state);
        }
      }
    }

#if defined(IREE_HAL_RISCV_MEASURE_CYCLES)
    iree_hal_cycle_mark_elf_exit();
    iree_hal_cycle_dispatch_end(cycle_slot);
#endif

    if (d->print_outputs && d->outputs) {
      for (size_t i = 0; i < d->output_count; ++i) {
        const multi_dispatch_output_desc_t* out = &d->outputs[i];
        printf("dispatch[%llu] output[%llu] %s ",
               (unsigned long long)di, (unsigned long long)i,
               out->element_type_label ? out->element_type_label : "");
        print_shape(out);
        printf(" hash=0x%016llx\n",
               (unsigned long long)hash_output_bytes(out));
        if (d->print_limit > 0) {
          dump_output(out, d->print_limit);
        }
      }
    }
  }

#if defined(IREE_HAL_RISCV_MEASURE_CYCLES)
  iree_hal_cycle_mark_vm_return();
  iree_hal_cycle_runtime_end();
  iree_hal_cycle_dump();
#endif

  tohost_exit(0);
}
