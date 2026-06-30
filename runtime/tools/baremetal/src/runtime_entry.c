#include "bundle_spec.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/buffer_view.h"
#include "iree/hal/buffer_view_util.h"
#include "iree/hal/device_group.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/embedded_elf_loader.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

typedef struct {
  bool initialized;
  iree_vm_instance_t* instance;
  iree_async_proactor_pool_t* proactor_pool;
  iree_hal_device_t* device;
  iree_vm_context_t* context;
  iree_vm_function_t entry_function;
} iree_baremetal_runtime_state_t;

static iree_baremetal_runtime_state_t g_runtime_state = {0};

#if IREE_BAREMETAL_ACCUMULATE_DISPATCH_CYCLES
extern void iree_baremetal_reset_dispatch_cycles(void);
extern uint64_t iree_baremetal_read_dispatch_cycle_sum(void);
extern uint64_t iree_baremetal_read_dispatch_count(void);
#endif  // IREE_BAREMETAL_ACCUMULATE_DISPATCH_CYCLES

static inline uint64_t iree_baremetal_read_cycle(void) {
#if defined(__riscv)
  uint64_t value = 0;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
#else
  return 0;
#endif
}

static void iree_baremetal_marker(const char* label) {
  printf("[IREE][marker] %s\n", label);
  fflush(stdout);
}

static iree_status_t iree_baremetal_check_status(const char* label,
                                                 iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "[baremetal] %s failed (code=%d)\n", label,
            (int)iree_status_code(status));
    iree_status_fprint(stderr, status);
    fputc('\n', stderr);
  }
  return status;
}

static iree_status_t iree_baremetal_create_device(
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);

  iree_hal_executable_loader_t* loader = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_embedded_elf_loader_create(
      /*plugin_manager=*/NULL, host_allocator, &loader));

  iree_string_view_t identifier = iree_make_cstring_view("local-sync");

  iree_hal_allocator_t* device_allocator = NULL;
  iree_status_t status = iree_hal_allocator_create_heap(
      identifier, host_allocator, host_allocator, &device_allocator);

  if (iree_status_is_ok(status)) {
    iree_async_proactor_pool_options_t pool_options =
        iree_async_proactor_pool_options_default();
    status = iree_async_proactor_pool_create(
        /*node_count=*/1, /*node_ids=*/NULL, pool_options, host_allocator,
        &g_runtime_state.proactor_pool);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = g_runtime_state.proactor_pool;
    status = iree_hal_sync_device_create(
        identifier, &params, &create_params, /*loader_count=*/1, &loader,
        device_allocator, host_allocator, out_device);
  }

  iree_hal_allocator_release(device_allocator);
  iree_hal_executable_loader_release(loader);
  return status;
}

static float iree_baremetal_bf16_to_f32(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } conv = {.u32 = (uint32_t)value << 16};
  return conv.f32;
}

static float iree_baremetal_f16_to_f32(uint16_t value) {
  uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
  uint32_t exp = ((uint32_t)value >> 10) & 0x1Fu;
  uint32_t mant = (uint32_t)value & 0x03FFu;
  uint32_t bits = 0;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x03FFu;
      bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  union {
    uint32_t u32;
    float f32;
  } conv = {.u32 = bits};
  return conv.f32;
}

static void iree_baremetal_print_decimal(float value) {
  if (isnan(value)) {
    printf("nan");
    return;
  }
  if (isinf(value)) {
    printf("%sinf", signbit(value) ? "-" : "");
    return;
  }
  double magnitude = (double)value;
  if (magnitude == 0.0) {
    printf("0.000000");
    return;
  }
  if (magnitude < 0.0) {
    printf("-");
    magnitude = -magnitude;
  }
  double int_part = floor(magnitude);
  printf("%llu", (unsigned long long)int_part);
  printf(".");
  double frac = magnitude - int_part;
  for (int i = 0; i < 6; ++i) {
    frac *= 10.0;
    int digit = (int)frac;
    printf("%d", digit);
    frac -= digit;
  }
}

static iree_host_size_t iree_baremetal_output_element_count(
    const iree_baremetal_output_t* output) {
  if (!output) return 0;
  if (output->element_count) return output->element_count;
  if (!output->element_size) return 0;
  return output->result_buffer.data_length / output->element_size;
}

static void iree_baremetal_print_shape(const iree_baremetal_output_t* output) {
  if (!output || output->shape_rank == 0 || !output->shape_dims) {
    printf("[]");
    return;
  }
  printf("[");
  for (iree_host_size_t i = 0; i < output->shape_rank; ++i) {
    if (i != 0) printf("x");
    printf("%lld", (long long)output->shape_dims[i]);
  }
  printf("]");
}

static void iree_baremetal_print_tensor_values(
    const iree_baremetal_output_t* output, iree_host_size_t limit) {
  if (!output || !output->result_buffer.data ||
      output->result_buffer.data_length == 0) {
    printf("  (no data)\n");
    return;
  }
  const iree_host_size_t element_count =
      iree_baremetal_output_element_count(output);
  if (element_count == 0) {
    printf("  (empty tensor)\n");
    return;
  }
  if (limit == 0 || limit > element_count) limit = element_count;

  switch (output->element_type) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16: {
      const uint16_t* values = (const uint16_t*)output->result_buffer.data;
      for (iree_host_size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] ", (unsigned long long)i);
        iree_baremetal_print_decimal(iree_baremetal_f16_to_f32(values[i]));
        printf("\n");
      }
      break;
    }
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16: {
      const uint16_t* values = (const uint16_t*)output->result_buffer.data;
      for (iree_host_size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] ", (unsigned long long)i);
        iree_baremetal_print_decimal(iree_baremetal_bf16_to_f32(values[i]));
        printf("\n");
      }
      break;
    }
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32: {
      const float* values = (const float*)output->result_buffer.data;
      for (iree_host_size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] ", (unsigned long long)i);
        iree_baremetal_print_decimal(values[i]);
        printf("\n");
      }
      break;
    }
    case IREE_HAL_ELEMENT_TYPE_SINT_32: {
      const int32_t* values = (const int32_t*)output->result_buffer.data;
      for (iree_host_size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] %d\n", (unsigned long long)i, values[i]);
      }
      break;
    }
    case IREE_HAL_ELEMENT_TYPE_UINT_32: {
      const uint32_t* values = (const uint32_t*)output->result_buffer.data;
      for (iree_host_size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] %u\n", (unsigned long long)i, values[i]);
      }
      break;
    }
    default: {
      const uint8_t* bytes = output->result_buffer.data;
      const iree_host_size_t stride = output->element_size ? output->element_size : 1;
      for (iree_host_size_t i = 0; i < limit; ++i) {
        printf("  [%5llu] 0x", (unsigned long long)i);
        for (iree_host_size_t b = 0; b < stride; ++b) {
          printf("%02X", bytes[i * stride + b]);
        }
        printf("\n");
      }
      break;
    }
  }
  if (limit < element_count) {
    printf("  ... (%llu values truncated)\n",
           (unsigned long long)(element_count - limit));
  }
}

static bool iree_baremetal_is_f32_vector(const iree_baremetal_output_t* output,
                                         iree_host_size_t element_count) {
  return output && output->element_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
         iree_baremetal_output_element_count(output) == element_count &&
         output->result_buffer.data &&
         output->result_buffer.data_length >= element_count * sizeof(float);
}

static bool iree_baremetal_is_f32_tensor_input(
    const iree_baremetal_input_t* input, iree_host_size_t element_count) {
  return input && input->kind == IREE_BAREMETAL_INPUT_KIND_TENSOR &&
         input->value.tensor.element_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
         input->value.tensor.buffer.data &&
         input->value.tensor.buffer.data_length >=
             element_count * sizeof(float);
}

static float iree_baremetal_abs_f32(float value) {
  return value < 0.0f ? -value : value;
}

static double iree_baremetal_l1_f32(const float* values,
                                    iree_host_size_t count) {
  double sum = 0.0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    sum += (double)iree_baremetal_abs_f32(values[i]);
  }
  return sum;
}

typedef struct {
  double l1;
  iree_host_size_t nan_count;
  iree_host_size_t inf_count;
} iree_baremetal_finite_l1_t;

static iree_baremetal_finite_l1_t iree_baremetal_finite_l1_f32(
    const float* values, iree_host_size_t count) {
  iree_baremetal_finite_l1_t stats = {0};
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (isnan(values[i])) {
      ++stats.nan_count;
    } else if (isinf(values[i])) {
      ++stats.inf_count;
    } else {
      stats.l1 += (double)iree_baremetal_abs_f32(values[i]);
    }
  }
  return stats;
}

static iree_baremetal_finite_l1_t iree_baremetal_finite_delta_l1_f32(
    const float* before, const float* after, iree_host_size_t count) {
  iree_baremetal_finite_l1_t stats = {0};
  for (iree_host_size_t i = 0; i < count; ++i) {
    const float delta = after[i] - before[i];
    if (isnan(delta)) {
      ++stats.nan_count;
    } else if (isinf(delta)) {
      ++stats.inf_count;
    } else {
      stats.l1 += (double)iree_baremetal_abs_f32(delta);
    }
  }
  return stats;
}

static double iree_baremetal_delta_l1_f32(const float* before,
                                          const float* after,
                                          iree_host_size_t count) {
  double sum = 0.0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    sum += (double)iree_baremetal_abs_f32(after[i] - before[i]);
  }
  return sum;
}

static iree_host_size_t iree_baremetal_argmax_f32(const float* values,
                                                  iree_host_size_t count,
                                                  float* out_max) {
  iree_host_size_t index = 0;
  float max_value = values[0];
  for (iree_host_size_t i = 1; i < count; ++i) {
    if (values[i] > max_value) {
      max_value = values[i];
      index = i;
    }
  }
  if (out_max) *out_max = max_value;
  return index;
}

static iree_host_size_t iree_baremetal_changed_count_f32(
    const float* before, const float* after, iree_host_size_t count) {
  iree_host_size_t changed = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (after[i] != before[i]) ++changed;
  }
  return changed;
}

static const char* iree_baremetal_pafkip_output_name(
    const iree_baremetal_bundle_spec_t* spec, iree_host_size_t index) {
  if (spec && spec->output_count == 7 &&
      iree_baremetal_is_f32_vector(&spec->outputs[0], 1000) &&
      iree_baremetal_is_f32_vector(&spec->outputs[1], 1) &&
      iree_baremetal_is_f32_vector(&spec->outputs[2], 1) &&
      iree_baremetal_is_f32_vector(&spec->outputs[3], 53120) &&
      iree_baremetal_is_f32_vector(&spec->outputs[4], 53120) &&
      iree_baremetal_is_f32_vector(&spec->outputs[5], 53120) &&
      iree_baremetal_is_f32_vector(&spec->outputs[6], 53120)) {
    static const char* kNames[] = {
        "final_logits",       "energy",          "loss",
        "new_main_bn_params", "new_ema_bn_params",
        "new_sgd_velocity",   "flat_bn_grads",
    };
    return kNames[index];
  }
  return "unnamed";
}

static void iree_baremetal_print_pafkip_summary(
    const iree_baremetal_bundle_spec_t* spec) {
  if (!spec || spec->input_count < 8 || spec->output_count < 7) return;

  const iree_host_size_t bn_count = 53120;
  if (!iree_baremetal_is_f32_vector(&spec->outputs[0], 1000) ||
      !iree_baremetal_is_f32_vector(&spec->outputs[1], 1) ||
      !iree_baremetal_is_f32_vector(&spec->outputs[2], 1) ||
      !iree_baremetal_is_f32_vector(&spec->outputs[3], bn_count) ||
      !iree_baremetal_is_f32_vector(&spec->outputs[4], bn_count) ||
      !iree_baremetal_is_f32_vector(&spec->outputs[5], bn_count) ||
      !iree_baremetal_is_f32_vector(&spec->outputs[6], bn_count) ||
      !iree_baremetal_is_f32_tensor_input(&spec->inputs[1], bn_count) ||
      !iree_baremetal_is_f32_tensor_input(&spec->inputs[2], bn_count) ||
      !iree_baremetal_is_f32_tensor_input(&spec->inputs[4], bn_count)) {
    return;
  }

  const float* logits = (const float*)spec->outputs[0].result_buffer.data;
  const float energy = ((const float*)spec->outputs[1].result_buffer.data)[0];
  const float loss = ((const float*)spec->outputs[2].result_buffer.data)[0];
  const float* new_main =
      (const float*)spec->outputs[3].result_buffer.data;
  const float* new_ema = (const float*)spec->outputs[4].result_buffer.data;
  const float* new_velocity =
      (const float*)spec->outputs[5].result_buffer.data;
  const float* grads = (const float*)spec->outputs[6].result_buffer.data;
  const float* old_main =
      (const float*)spec->inputs[1].value.tensor.buffer.data;
  const float* old_ema =
      (const float*)spec->inputs[2].value.tensor.buffer.data;

  float max_logit = 0.0f;
  iree_host_size_t pred =
      iree_baremetal_argmax_f32(logits, 1000, &max_logit);
  iree_baremetal_finite_l1_t grad_stats =
      iree_baremetal_finite_l1_f32(grads, bn_count);
  iree_baremetal_finite_l1_t velocity_stats =
      iree_baremetal_finite_l1_f32(new_velocity, bn_count);
  iree_baremetal_finite_l1_t sgd_stats =
      iree_baremetal_finite_delta_l1_f32(old_main, new_main, bn_count);
  iree_baremetal_finite_l1_t ema_stats =
      iree_baremetal_finite_delta_l1_f32(old_ema, new_ema, bn_count);
  iree_host_size_t changed_main =
      iree_baremetal_changed_count_f32(old_main, new_main, bn_count);
  iree_host_size_t changed_ema =
      iree_baremetal_changed_count_f32(old_ema, new_ema, bn_count);

  printf("[PAFKIP] step=1\n");
  printf("[PAFKIP] stage1 transform+resnet pred=%llu max_logit=",
         (unsigned long long)pred);
  iree_baremetal_print_decimal(max_logit);
  printf(" energy=");
  iree_baremetal_print_decimal(energy);
  printf("\n");

  printf("[PAFKIP] stage2 tta_loss=");
  iree_baremetal_print_decimal(loss);
  printf(" bn_grad_l1=");
  iree_baremetal_print_decimal((float)grad_stats.l1);
  printf("\n");

  printf("[PAFKIP] stage3 sgd_bn delta_l1=");
  iree_baremetal_print_decimal((float)sgd_stats.l1);
  printf(" velocity_l1=");
  iree_baremetal_print_decimal((float)velocity_stats.l1);
  printf(" changed=%llu/%llu", (unsigned long long)changed_main,
         (unsigned long long)bn_count);
  printf(" param0=");
  iree_baremetal_print_decimal(old_main[0]);
  printf(" -> ");
  iree_baremetal_print_decimal(new_main[0]);
  printf("\n");

  printf("[PAFKIP] stage4 ema delta_l1=");
  iree_baremetal_print_decimal((float)ema_stats.l1);
  printf(" changed=%llu/%llu", (unsigned long long)changed_ema,
         (unsigned long long)bn_count);
  printf(" param0=");
  iree_baremetal_print_decimal(old_ema[0]);
  printf(" -> ");
  iree_baremetal_print_decimal(new_ema[0]);
  printf("\n");

  printf("[PAFKIP] stage5 kip_final_pred=%llu updated=1\n",
         (unsigned long long)pred);
}

static void iree_baremetal_print_outputs(
    const iree_baremetal_bundle_spec_t* spec) {
  if (!spec || !spec->print_outputs || !spec->outputs ||
      spec->output_count == 0) {
    return;
  }
  iree_baremetal_print_pafkip_summary(spec);
  for (iree_host_size_t i = 0; i < spec->output_count; ++i) {
    const iree_baremetal_output_t* output = &spec->outputs[i];
    const char* type_label =
        output->element_type_label ? output->element_type_label : "unknown";
    const char* output_name = iree_baremetal_pafkip_output_name(spec, i);
    printf("[IREE][output %llu %s] shape=", (unsigned long long)i,
           output_name);
    iree_baremetal_print_shape(output);
    printf(" dtype=%s elements=%llu\n", type_label,
           (unsigned long long)iree_baremetal_output_element_count(output));
    iree_baremetal_print_tensor_values(output, spec->print_limit);
    printf("\n");
  }
  fflush(stdout);
}

static iree_status_t iree_baremetal_create_buffer_from_span(
    iree_hal_device_t* device, const iree_baremetal_tensor_input_t* input,
    iree_hal_buffer_view_t** out_view) {
  if (!input || !out_view) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid tensor input");
  }

  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device);
  return iree_hal_buffer_view_allocate_buffer_copy(
      device, allocator, input->shape_rank, input->shape_dims,
      input->element_type, input->encoding_type,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(input->buffer.data, input->buffer.data_length),
      out_view);
}

static iree_status_t iree_baremetal_initialize(
    const iree_baremetal_bundle_spec_t* spec,
    iree_allocator_t host_allocator) {
  iree_baremetal_marker("initialize.begin");
  IREE_RETURN_IF_ERROR(iree_baremetal_check_status(
      "vm_instance_create",
      iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT, host_allocator,
                              &g_runtime_state.instance)));
  IREE_RETURN_IF_ERROR(iree_baremetal_check_status(
      "hal_module_register_all_types",
      iree_hal_module_register_all_types(g_runtime_state.instance)));
  IREE_RETURN_IF_ERROR(iree_baremetal_check_status(
      "create_device",
      iree_baremetal_create_device(host_allocator, &g_runtime_state.device)));

  iree_async_frontier_tracker_t* frontier_tracker = NULL;
  IREE_RETURN_IF_ERROR(iree_baremetal_check_status(
      "frontier_tracker_create",
      iree_async_frontier_tracker_create(
          iree_async_frontier_tracker_options_default(), host_allocator,
          &frontier_tracker)));

  iree_hal_device_group_t* device_group = NULL;
  iree_status_t status = iree_hal_device_group_create_from_device(
      g_runtime_state.device, frontier_tracker, host_allocator, &device_group);
  iree_async_frontier_tracker_release(frontier_tracker);
  IREE_RETURN_IF_ERROR(
      iree_baremetal_check_status("device_group_create", status));

  iree_vm_module_t* hal_module = NULL;
  status = iree_hal_module_create(
      g_runtime_state.instance, iree_hal_module_device_policy_default(),
      device_group, IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
      iree_hal_module_debug_sink_stdio(stderr), host_allocator, &hal_module);
  iree_hal_device_group_release(device_group);
  IREE_RETURN_IF_ERROR(iree_baremetal_check_status("hal_module_create", status));

  iree_const_byte_span_t module_data = iree_make_const_byte_span(
      spec->module_data.data, spec->module_data.data_length);

  iree_vm_module_t* bytecode_module = NULL;
  status = iree_vm_bytecode_module_create(
      g_runtime_state.instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE, module_data,
      iree_allocator_null(), host_allocator, &bytecode_module);
  if (!iree_status_is_ok(status)) {
    iree_vm_module_release(hal_module);
    return iree_baremetal_check_status("vm_bytecode_module_create", status);
  }

  iree_vm_module_t* modules[] = {hal_module, bytecode_module};
  status = iree_vm_context_create_with_modules(
      g_runtime_state.instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules),
      modules, host_allocator, &g_runtime_state.context);
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);
  IREE_RETURN_IF_ERROR(
      iree_baremetal_check_status("vm_context_create_with_modules", status));

  IREE_RETURN_IF_ERROR(iree_baremetal_check_status(
      "vm_context_resolve_function",
      iree_vm_context_resolve_function(g_runtime_state.context,
                                       spec->entry_function,
                                       &g_runtime_state.entry_function)));
  g_runtime_state.initialized = true;
  iree_baremetal_marker("initialize.end");
  return iree_ok_status();
}

iree_status_t iree_baremetal_run(const iree_baremetal_bundle_spec_t* spec) {
  iree_baremetal_marker("run.begin");
  if (!spec) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "spec is null");
  }

  iree_allocator_t host_allocator = iree_allocator_system();
  if (!g_runtime_state.initialized) {
    IREE_RETURN_IF_ERROR(iree_baremetal_initialize(spec, host_allocator));
  }

  iree_vm_list_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                           spec->input_count, host_allocator,
                                           &inputs));

  iree_baremetal_marker("inputs.begin");
  for (iree_host_size_t i = 0; i < spec->input_count; ++i) {
    const iree_baremetal_input_t* input = &spec->inputs[i];
    switch (input->kind) {
      case IREE_BAREMETAL_INPUT_KIND_TENSOR: {
        iree_hal_buffer_view_t* buffer_view = NULL;
        IREE_RETURN_IF_ERROR(iree_baremetal_create_buffer_from_span(
            g_runtime_state.device, &input->value.tensor, &buffer_view));
        iree_vm_ref_t buffer_ref = iree_hal_buffer_view_move_ref(buffer_view);
        IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &buffer_ref));
        break;
      }
      case IREE_BAREMETAL_INPUT_KIND_I32: {
        iree_vm_value_t value = iree_vm_value_make_i32(input->value.i32);
        IREE_RETURN_IF_ERROR(iree_vm_list_push_value(inputs, &value));
        break;
      }
      case IREE_BAREMETAL_INPUT_KIND_I64: {
        iree_vm_value_t value = iree_vm_value_make_i64(input->value.i64);
        IREE_RETURN_IF_ERROR(iree_vm_list_push_value(inputs, &value));
        break;
      }
      case IREE_BAREMETAL_INPUT_KIND_F32: {
        iree_vm_value_t value = iree_vm_value_make_f32(input->value.f32);
        IREE_RETURN_IF_ERROR(iree_vm_list_push_value(inputs, &value));
        break;
      }
      case IREE_BAREMETAL_INPUT_KIND_F64: {
        iree_vm_value_t value = iree_vm_value_make_f64(input->value.f64);
        IREE_RETURN_IF_ERROR(iree_vm_list_push_value(inputs, &value));
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported input kind %d", (int)input->kind);
    }
  }
  iree_baremetal_marker("inputs.end");

  iree_vm_list_t* outputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                           spec->output_count, host_allocator,
                                           &outputs));

  iree_baremetal_marker("invoke.begin");
#if IREE_BAREMETAL_ACCUMULATE_DISPATCH_CYCLES
  iree_baremetal_reset_dispatch_cycles();
#endif  // IREE_BAREMETAL_ACCUMULATE_DISPATCH_CYCLES
  const uint64_t invoke_cycle_begin = iree_baremetal_read_cycle();
  iree_status_t status =
      iree_vm_invoke(g_runtime_state.context, g_runtime_state.entry_function,
                     IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL, inputs,
                     outputs, host_allocator);
  const uint64_t invoke_cycle_end = iree_baremetal_read_cycle();
  iree_baremetal_marker("invoke.end");
  printf("[PAFKIP][timing] invoke_cycles=%" PRIu64 "\n",
         invoke_cycle_end - invoke_cycle_begin);
#if IREE_BAREMETAL_ACCUMULATE_DISPATCH_CYCLES
  printf("[PAFKIP][timing] kernel_cycles_sum=%" PRIu64
         " kernel_dispatches=%" PRIu64 "\n",
         iree_baremetal_read_dispatch_cycle_sum(),
         iree_baremetal_read_dispatch_count());
#endif  // IREE_BAREMETAL_ACCUMULATE_DISPATCH_CYCLES
  fflush(stdout);
  status = iree_baremetal_check_status("vm_invoke", status);

  if (iree_status_is_ok(status)) {
    iree_baremetal_marker("outputs.begin");
    for (iree_host_size_t i = 0; i < spec->output_count; ++i) {
      const iree_baremetal_output_t* output = &spec->outputs[i];
      if (!output->result_buffer.data || output->result_buffer.data_length == 0) {
        continue;
      }
      iree_hal_buffer_view_t* buffer_view =
          iree_vm_list_get_buffer_view_assign(outputs, i);
      if (!buffer_view) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "missing output buffer %" PRIhsz, i);
        break;
      }
      status = iree_hal_device_transfer_d2h(
          g_runtime_state.device, iree_hal_buffer_view_buffer(buffer_view), 0,
          output->result_buffer.data, output->result_buffer.data_length,
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
      status = iree_baremetal_check_status("device_transfer_d2h", status);
      if (!iree_status_is_ok(status)) break;
    }
    iree_baremetal_marker("outputs.end");
    if (iree_status_is_ok(status)) {
      iree_baremetal_print_outputs(spec);
    }
  }

  iree_vm_list_release(outputs);
  iree_vm_list_release(inputs);
  iree_baremetal_marker("run.end");
  return status;
}

int main(void) {
  iree_baremetal_marker("main.begin");
  __asm__ volatile("li t0, 0x1E600\n"
                   "csrs mstatus, t0\n"
                   ::: "t0");

  const iree_baremetal_bundle_spec_t* spec = iree_baremetal_acquire_spec();
  const iree_status_t status = iree_baremetal_run(spec);
  iree_baremetal_marker("main.after_run");
  int exit_code = (int)iree_status_code(status);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }
  __asm__ volatile("fence" ::: "memory");
  return exit_code;
}
