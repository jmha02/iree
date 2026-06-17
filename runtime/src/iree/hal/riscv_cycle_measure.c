#include "iree/hal/riscv_cycle_measure.h"

#if defined(IREE_HAL_RISCV_MEASURE_CYCLES)

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define IREE_HAL_CYCLE_MAX_DISPATCHES 64

typedef struct {
  uint64_t command_begin;
  uint64_t command_end;
  uint64_t elf_begin;
  uint64_t elf_end;
} iree_hal_cycle_dispatch_measure_t;

typedef struct {
  uint64_t runtime_start;
  uint64_t before_vm_invoke;
  uint64_t after_vm_invoke;
  uint64_t runtime_end;
  uint64_t queue_submit_total;
  uint64_t queue_submit_begin;
  size_t queue_submit_count;
  size_t dispatch_count;
  size_t dispatch_stack_depth;
  size_t dispatch_stack[IREE_HAL_CYCLE_MAX_DISPATCHES];
  iree_hal_cycle_dispatch_measure_t
      dispatches[IREE_HAL_CYCLE_MAX_DISPATCHES];
} iree_hal_cycle_measure_state_t;

static iree_hal_cycle_measure_state_t g_cycle_state;

static inline uint64_t iree_hal_cycle_read(void) {
  uint64_t value = 0;
  __asm__ __volatile__("rdcycle %0" : "=r"(value));
  return value;
}

void iree_hal_cycle_runtime_start(void) {
  memset(&g_cycle_state, 0, sizeof(g_cycle_state));
  g_cycle_state.runtime_start = iree_hal_cycle_read();
}

void iree_hal_cycle_mark_vm_invoke(void) {
  if (g_cycle_state.before_vm_invoke == 0) {
    g_cycle_state.before_vm_invoke = iree_hal_cycle_read();
  }
}

void iree_hal_cycle_mark_vm_return(void) {
  g_cycle_state.after_vm_invoke = iree_hal_cycle_read();
}

size_t iree_hal_cycle_dispatch_begin(void) {
  if (g_cycle_state.dispatch_count >= IREE_HAL_CYCLE_MAX_DISPATCHES) {
    return SIZE_MAX;
  }
  size_t slot = g_cycle_state.dispatch_count++;
  g_cycle_state.dispatch_stack[g_cycle_state.dispatch_stack_depth++] = slot;
  g_cycle_state.dispatches[slot].command_begin = iree_hal_cycle_read();
  return slot;
}

static void iree_hal_cycle_pop_slot(size_t slot) {
  if (slot == SIZE_MAX || g_cycle_state.dispatch_stack_depth == 0) {
    return;
  }
  size_t* stack = g_cycle_state.dispatch_stack;
  size_t depth = g_cycle_state.dispatch_stack_depth;
  if (stack[depth - 1] == slot) {
    g_cycle_state.dispatch_stack_depth = depth - 1;
    return;
  }
  for (size_t i = 0; i < depth; ++i) {
    if (stack[i] == slot) {
      for (size_t j = i + 1; j < depth; ++j) {
        stack[j - 1] = stack[j];
      }
      g_cycle_state.dispatch_stack_depth = depth - 1;
      return;
    }
  }
}

void iree_hal_cycle_dispatch_end(size_t slot) {
  if (slot == SIZE_MAX || slot >= g_cycle_state.dispatch_count) {
    return;
  }
  g_cycle_state.dispatches[slot].command_end = iree_hal_cycle_read();
  iree_hal_cycle_pop_slot(slot);
}

void iree_hal_cycle_mark_elf_enter(void) {
  if (g_cycle_state.dispatch_stack_depth == 0) {
    return;
  }
  size_t slot =
      g_cycle_state.dispatch_stack[g_cycle_state.dispatch_stack_depth - 1];
  if (slot >= g_cycle_state.dispatch_count || slot == SIZE_MAX) {
    return;
  }
  iree_hal_cycle_dispatch_measure_t* measure =
      &g_cycle_state.dispatches[slot];
  if (measure->elf_begin == 0) {
    __asm__ __volatile__("lui t6, 0xd10" ::: "t6");  // trace marker: dispatch enter
    measure->elf_begin = iree_hal_cycle_read();
  }
}

void iree_hal_cycle_mark_elf_exit(void) {
  if (g_cycle_state.dispatch_stack_depth == 0) {
    return;
  }
  size_t slot =
      g_cycle_state.dispatch_stack[g_cycle_state.dispatch_stack_depth - 1];
  if (slot >= g_cycle_state.dispatch_count || slot == SIZE_MAX) {
    return;
  }
  __asm__ __volatile__("lui t6, 0xd11" ::: "t6");  // trace marker: dispatch exit
  g_cycle_state.dispatches[slot].elf_end = iree_hal_cycle_read();
}

void iree_hal_cycle_runtime_end(void) {
  g_cycle_state.runtime_end = iree_hal_cycle_read();
}

void iree_hal_cycle_mark_queue_submit_begin(void) {
  g_cycle_state.queue_submit_begin = iree_hal_cycle_read();
}

void iree_hal_cycle_mark_queue_submit_end(void) {
  if (g_cycle_state.queue_submit_begin == 0) {
    return;
  }
  uint64_t end = iree_hal_cycle_read();
  if (end >= g_cycle_state.queue_submit_begin) {
    g_cycle_state.queue_submit_total +=
        end - g_cycle_state.queue_submit_begin;
  }
  g_cycle_state.queue_submit_begin = 0;
  ++g_cycle_state.queue_submit_count;
}

static uint64_t iree_hal_cycle_delta(uint64_t start, uint64_t end) {
  return (start == 0 || end == 0 || end < start) ? 0 : (end - start);
}

void iree_hal_cycle_dump(void) {
  if (g_cycle_state.runtime_start == 0 ||
      g_cycle_state.runtime_end <= g_cycle_state.runtime_start) {
    return;
  }

  uint64_t prologue = iree_hal_cycle_delta(
      g_cycle_state.runtime_start, g_cycle_state.before_vm_invoke);
  printf("prologue: %" PRIu64 "\n", prologue);

  for (size_t i = 0; i < g_cycle_state.dispatch_count; ++i) {
    const iree_hal_cycle_dispatch_measure_t* measure =
        &g_cycle_state.dispatches[i];
    if (measure->elf_begin == 0 || measure->elf_end == 0) {
      continue;
    }
    uint64_t kernel_cycles =
        iree_hal_cycle_delta(measure->elf_begin, measure->elf_end);
    printf("dispatch%" PRIu64 ": cb=%" PRIu64 ", elf=%" PRIu64
           ", kernel=%" PRIu64 "\n",
           (uint64_t)i, measure->command_begin, measure->elf_begin,
           kernel_cycles);
  }
  fflush(stdout);
}

#endif  // IREE_HAL_RISCV_MEASURE_CYCLES
