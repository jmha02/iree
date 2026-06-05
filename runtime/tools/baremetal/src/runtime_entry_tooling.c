#include "bundle_spec.h"

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/driver_registry.h"
#include "iree/hal/drivers/local_sync/registration/driver_module.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/run_module.h"
#include "iree/vm/api.h"

iree_status_t iree_baremetal_run(const iree_baremetal_bundle_spec_t* spec) {
  if (!spec) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "spec is null");
  }

  iree_flags_set_usage(
      "baremetal-static-run-module",
      "Runs an embedded IREE VMFB with embedded baremetal inputs.\n");
  int argc = spec->tooling_argc;
  char** argv = spec->tooling_argv;
  iree_status_t status = iree_flags_parse(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc,
                                          &argv);
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "[baremetal-tooling] flag parse failed\n");
    iree_status_fprint(stderr, status);
    fputc('\n', stderr);
    iree_flags_dump(IREE_FLAG_DUMP_MODE_DEFAULT, stderr);
    return status;
  }

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_vm_instance_t* instance = NULL;
  status = iree_tooling_create_instance(host_allocator, &instance);

  if (iree_status_is_ok(status)) {
    status = iree_hal_local_sync_driver_module_register(
        iree_hal_driver_registry_default());
  }

  int exit_code = 0;
  if (iree_status_is_ok(status)) {
    status = iree_tooling_run_module_with_data(
        instance, IREE_SV("local-sync"),
        iree_make_const_byte_span(spec->module_data.data,
                                  spec->module_data.data_length),
        host_allocator, &exit_code);
  }

  iree_vm_instance_release(instance);

  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "[baremetal-tooling] run_module failed (code=%d)\n",
            (int)iree_status_code(status));
    iree_status_fprint(stderr, status);
    fputc('\n', stderr);
    return status;
  }
  if (exit_code != 0) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "run_module returned exit code %d", exit_code);
  }
  return iree_ok_status();
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  const iree_baremetal_bundle_spec_t* spec = iree_baremetal_acquire_spec();
  iree_status_t status = iree_baremetal_run(spec);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return 1;
  }
  return 0;
}
