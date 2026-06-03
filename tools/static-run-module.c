// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/driver_registry.h"
#include "iree/hal/drivers/local_sync/registration/driver_module.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/run_module.h"
#include "iree/vm/api.h"

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      "static-run-module",
      "Runs a function within a compiled IREE module using a statically-linked\n"
      "local-sync HAL driver. Modules can be provided by file path\n"
      "(`--module=file.vmfb`) or read from stdin (`--module=-`) and the\n"
      "function to execute matches the original name provided to the compiler\n"
      "(`--function=foo` for `func.func @foo`).\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t host_allocator = iree_allocator_system();

  iree_vm_instance_t* instance = NULL;
  iree_status_t status =
      iree_tooling_create_instance(host_allocator, &instance);

  if (iree_status_is_ok(status)) {
    status = iree_hal_local_sync_driver_module_register(
        iree_hal_driver_registry_default());
  }

  int exit_code = EXIT_SUCCESS;
  if (iree_status_is_ok(status)) {
    status = iree_tooling_run_module_with_data(
        instance, IREE_SV("local-sync"), iree_const_byte_span_empty(),
        host_allocator, &exit_code);
  }

  iree_vm_instance_release(instance);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
