// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/thread.h"

#if defined(IREE_PLATFORM_GENERIC)

struct iree_thread_t {
  int unused;
};

iree_status_t iree_thread_create(iree_thread_entry_t entry, void* entry_arg,
                                 iree_thread_create_params_t params,
                                 iree_allocator_t allocator,
                                 iree_thread_t** out_thread) {
  (void)entry;
  (void)entry_arg;
  (void)params;
  (void)allocator;
  *out_thread = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "threads are unavailable on Generic");
}

void iree_thread_retain(iree_thread_t* thread) { (void)thread; }

void iree_thread_release(iree_thread_t* thread) { (void)thread; }

uintptr_t iree_thread_id(iree_thread_t* thread) {
  (void)thread;
  return 0;
}

iree_thread_override_t* iree_thread_priority_class_override_begin(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class) {
  (void)thread;
  (void)priority_class;
  return NULL;
}

void iree_thread_override_end(iree_thread_override_t* override_token) {
  (void)override_token;
}

void iree_thread_request_affinity(iree_thread_t* thread,
                                  iree_thread_affinity_t affinity) {
  (void)thread;
  (void)affinity;
}

void iree_thread_resume(iree_thread_t* thread) { (void)thread; }

void iree_thread_join(iree_thread_t* thread) { (void)thread; }

void iree_thread_yield(void) {}

#endif  // IREE_PLATFORM_GENERIC
