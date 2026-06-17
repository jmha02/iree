#ifndef IREE_HAL_RISCV_CYCLE_MEASURE_H_
#define IREE_HAL_RISCV_CYCLE_MEASURE_H_

#include <stddef.h>

#if defined(IREE_HAL_RISCV_MEASURE_CYCLES)

size_t iree_hal_cycle_dispatch_begin(void);
void iree_hal_cycle_dispatch_end(size_t slot);
void iree_hal_cycle_mark_elf_enter(void);
void iree_hal_cycle_mark_elf_exit(void);

void iree_hal_cycle_mark_queue_submit_begin(void);
void iree_hal_cycle_mark_queue_submit_end(void);

void iree_hal_cycle_runtime_start(void);
void iree_hal_cycle_mark_vm_invoke(void);
void iree_hal_cycle_mark_vm_return(void);
void iree_hal_cycle_runtime_end(void);
void iree_hal_cycle_dump(void);

#else  // !IREE_HAL_RISCV_MEASURE_CYCLES

static inline size_t iree_hal_cycle_dispatch_begin(void) { return 0; }
static inline void iree_hal_cycle_dispatch_end(size_t slot) { (void)slot; }
static inline void iree_hal_cycle_mark_elf_enter(void) {}
static inline void iree_hal_cycle_mark_elf_exit(void) {}

static inline void iree_hal_cycle_mark_queue_submit_begin(void) {}
static inline void iree_hal_cycle_mark_queue_submit_end(void) {}

static inline void iree_hal_cycle_runtime_start(void) {}
static inline void iree_hal_cycle_mark_vm_invoke(void) {}
static inline void iree_hal_cycle_mark_vm_return(void) {}
static inline void iree_hal_cycle_runtime_end(void) {}
static inline void iree_hal_cycle_dump(void) {}

#endif  // IREE_HAL_RISCV_MEASURE_CYCLES

#endif  // IREE_HAL_RISCV_CYCLE_MEASURE_H_
