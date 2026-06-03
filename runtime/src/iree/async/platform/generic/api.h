// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_PLATFORM_GENERIC_API_H_
#define IREE_ASYNC_PLATFORM_GENERIC_API_H_

#include "iree/async/proactor.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a minimal proactor for bare-metal Generic targets. It supports the
// notification operations required by the local-sync HAL driver and reports
// unavailable for host OS I/O features such as files, sockets, and signals.
iree_status_t iree_async_proactor_create_generic(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_GENERIC_API_H_
