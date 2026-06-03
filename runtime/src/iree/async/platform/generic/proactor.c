// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/generic/api.h"

#include <stddef.h>

#include "iree/async/notification.h"
#include "iree/async/region.h"
#include "iree/base/internal/atomics.h"

typedef struct iree_async_generic_proactor_t {
  iree_async_proactor_t base;
} iree_async_generic_proactor_t;

typedef struct iree_async_generic_buffer_registration_t {
  iree_async_buffer_registration_entry_t entry;
  iree_async_region_t region;
} iree_async_generic_buffer_registration_t;

static iree_async_generic_proactor_t* iree_async_generic_proactor_cast(
    iree_async_proactor_t* base_proactor) {
  return (iree_async_generic_proactor_t*)base_proactor;
}

static iree_status_t iree_async_generic_unavailable(const char* operation) {
  (void)operation;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Generic proactor does not support host async I/O");
}

static void iree_async_generic_proactor_destroy(
    iree_async_proactor_t* base_proactor) {
  iree_async_generic_proactor_t* proactor =
      iree_async_generic_proactor_cast(base_proactor);
  iree_allocator_t allocator = base_proactor->allocator;
  iree_allocator_free(allocator, proactor);
}

static iree_async_proactor_capabilities_t
iree_async_generic_proactor_query_capabilities(
    iree_async_proactor_t* base_proactor) {
  (void)base_proactor;
  return IREE_ASYNC_PROACTOR_CAPABILITY_NONE;
}

static iree_status_t iree_async_generic_proactor_submit(
    iree_async_proactor_t* base_proactor,
    iree_async_operation_list_t operations) {
  (void)base_proactor;
  (void)operations;
  return iree_async_generic_unavailable("submit");
}

static iree_status_t iree_async_generic_proactor_poll(
    iree_async_proactor_t* base_proactor, iree_timeout_t timeout,
    iree_host_size_t* out_completed_count) {
  iree_host_size_t completed_count =
      iree_async_proactor_run_progress(base_proactor);
  if (out_completed_count) *out_completed_count = completed_count;
  if (completed_count > 0) return iree_ok_status();
  return iree_timeout_is_immediate(timeout)
             ? iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED)
             : iree_ok_status();
}

static void iree_async_generic_proactor_wake(
    iree_async_proactor_t* base_proactor) {
  (void)base_proactor;
}

static iree_status_t iree_async_generic_proactor_cancel(
    iree_async_proactor_t* base_proactor,
    iree_async_operation_t* operation) {
  (void)base_proactor;
  (void)operation;
  return iree_status_from_code(IREE_STATUS_NOT_FOUND);
}

static iree_status_t iree_async_generic_proactor_create_socket(
    iree_async_proactor_t* base_proactor, iree_async_socket_type_t type,
    iree_async_socket_options_t options, iree_async_socket_t** out_socket) {
  (void)base_proactor;
  (void)type;
  (void)options;
  *out_socket = NULL;
  return iree_async_generic_unavailable("create_socket");
}

static iree_status_t iree_async_generic_proactor_import_socket(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t primitive,
    iree_async_socket_type_t type, iree_async_socket_flags_t flags,
    iree_async_socket_t** out_socket) {
  (void)base_proactor;
  (void)primitive;
  (void)type;
  (void)flags;
  *out_socket = NULL;
  return iree_async_generic_unavailable("import_socket");
}

static void iree_async_generic_proactor_destroy_socket(
    iree_async_proactor_t* base_proactor, iree_async_socket_t* socket) {
  (void)base_proactor;
  (void)socket;
}

static iree_status_t iree_async_generic_proactor_import_file(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t primitive,
    iree_async_file_t** out_file) {
  (void)base_proactor;
  (void)primitive;
  *out_file = NULL;
  return iree_async_generic_unavailable("import_file");
}

static void iree_async_generic_proactor_destroy_file(
    iree_async_proactor_t* base_proactor, iree_async_file_t* file) {
  (void)base_proactor;
  (void)file;
}

static iree_status_t iree_async_generic_proactor_create_event(
    iree_async_proactor_t* base_proactor, iree_async_event_t** out_event) {
  (void)base_proactor;
  *out_event = NULL;
  return iree_async_generic_unavailable("create_event");
}

static void iree_async_generic_proactor_destroy_event(
    iree_async_proactor_t* base_proactor, iree_async_event_t* event) {
  (void)base_proactor;
  (void)event;
}

static iree_status_t iree_async_generic_proactor_register_event_source(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t handle,
    iree_async_event_source_callback_t callback,
    iree_async_event_source_t** out_event_source) {
  (void)base_proactor;
  (void)handle;
  (void)callback;
  *out_event_source = NULL;
  return iree_async_generic_unavailable("register_event_source");
}

static void iree_async_generic_proactor_unregister_event_source(
    iree_async_proactor_t* base_proactor,
    iree_async_event_source_t* event_source) {
  (void)base_proactor;
  (void)event_source;
}

static iree_status_t iree_async_generic_proactor_create_notification(
    iree_async_proactor_t* base_proactor, iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification) {
  *out_notification = NULL;
  iree_async_notification_t* notification = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(base_proactor->allocator,
                                             sizeof(*notification),
                                             (void**)&notification));
  memset(notification, 0, sizeof(*notification));
  iree_atomic_ref_count_init(&notification->ref_count);
  notification->proactor = base_proactor;
  notification->epoch_ptr = &notification->epoch;
  notification->flags = flags;
  notification->mode = IREE_ASYNC_NOTIFICATION_MODE_FUTEX;
  *out_notification = notification;
  return iree_ok_status();
}

static iree_status_t iree_async_generic_proactor_create_notification_shared(
    iree_async_proactor_t* base_proactor,
    const iree_async_notification_shared_options_t* options,
    iree_async_notification_t** out_notification) {
  *out_notification = NULL;
  iree_async_notification_t* notification = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(base_proactor->allocator,
                                             sizeof(*notification),
                                             (void**)&notification));
  memset(notification, 0, sizeof(*notification));
  iree_atomic_ref_count_init(&notification->ref_count);
  notification->proactor = base_proactor;
  notification->epoch_ptr = options->epoch_address;
  notification->flags = IREE_ASYNC_NOTIFICATION_FLAG_SHARED;
  notification->mode = IREE_ASYNC_NOTIFICATION_MODE_FUTEX;
  *out_notification = notification;
  return iree_ok_status();
}

static void iree_async_generic_proactor_destroy_notification(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification) {
  iree_allocator_free(base_proactor->allocator, notification);
}

static void iree_async_generic_proactor_notification_signal(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification, int32_t wake_count) {
  (void)base_proactor;
  (void)notification;
  (void)wake_count;
}

static bool iree_async_generic_proactor_notification_wait(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification, uint32_t wait_token,
    iree_timeout_t timeout) {
  (void)base_proactor;
  if (iree_atomic_load(notification->epoch_ptr, iree_memory_order_acquire) !=
      (int32_t)wait_token) {
    return true;
  }
  if (iree_timeout_is_immediate(timeout)) return false;
  return iree_atomic_load(notification->epoch_ptr, iree_memory_order_acquire) !=
         (int32_t)wait_token;
}

static iree_status_t iree_async_generic_proactor_register_relay(
    iree_async_proactor_t* base_proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay) {
  (void)base_proactor;
  (void)source;
  (void)sink;
  (void)flags;
  (void)error_callback;
  *out_relay = NULL;
  return iree_async_generic_unavailable("register_relay");
}

static void iree_async_generic_proactor_unregister_relay(
    iree_async_proactor_t* base_proactor, iree_async_relay_t* relay) {
  (void)base_proactor;
  (void)relay;
}

static void iree_async_generic_buffer_registration_destroy(
    iree_async_region_t* region) {
  iree_async_generic_buffer_registration_t* registration =
      (iree_async_generic_buffer_registration_t*)((char*)region -
                                                  offsetof(
                                                      iree_async_generic_buffer_registration_t,
                                                      region));
  iree_allocator_free(region->proactor->allocator, registration);
}

static void iree_async_generic_buffer_registration_cleanup(void* entry_ptr,
                                                           void* proactor_ptr) {
  (void)proactor_ptr;
  iree_async_generic_buffer_registration_t* registration =
      (iree_async_generic_buffer_registration_t*)entry_ptr;
  iree_async_region_release(&registration->region);
}

static iree_status_t iree_async_generic_proactor_register_buffer(
    iree_async_proactor_t* base_proactor,
    iree_async_buffer_registration_state_t* state, iree_byte_span_t buffer,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  *out_entry = NULL;
  iree_async_generic_buffer_registration_t* registration = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(base_proactor->allocator,
                                             sizeof(*registration),
                                             (void**)&registration));
  memset(registration, 0, sizeof(*registration));
  iree_async_region_t* region = &registration->region;
  iree_atomic_ref_count_init(&region->ref_count);
  region->proactor = base_proactor;
  region->destroy_fn = iree_async_generic_buffer_registration_destroy;
  region->type = IREE_ASYNC_REGION_TYPE_NONE;
  region->access_flags = access_flags;
  region->base_ptr = (void*)buffer.data;
  region->length = buffer.data_length;

  iree_async_buffer_registration_entry_t* entry = &registration->entry;
  entry->proactor = base_proactor;
  entry->cleanup_fn = iree_async_generic_buffer_registration_cleanup;
  entry->region = region;
  iree_async_buffer_registration_state_add(state, entry);
  *out_entry = entry;
  return iree_ok_status();
}

static iree_status_t iree_async_generic_proactor_register_dmabuf(
    iree_async_proactor_t* base_proactor,
    iree_async_buffer_registration_state_t* state, int dmabuf_fd,
    uint64_t offset, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  (void)base_proactor;
  (void)state;
  (void)dmabuf_fd;
  (void)offset;
  (void)length;
  (void)access_flags;
  *out_entry = NULL;
  return iree_async_generic_unavailable("register_dmabuf");
}

static void iree_async_generic_proactor_unregister_buffer(
    iree_async_proactor_t* base_proactor,
    iree_async_buffer_registration_entry_t* entry,
    iree_async_buffer_registration_state_t* state) {
  (void)base_proactor;
  iree_async_buffer_registration_state_remove(state, entry);
  entry->cleanup_fn(entry, base_proactor);
}

static iree_status_t iree_async_generic_proactor_register_slab(
    iree_async_proactor_t* base_proactor, iree_async_slab_t* slab,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_region_t** out_region) {
  (void)base_proactor;
  (void)slab;
  (void)access_flags;
  *out_region = NULL;
  return iree_async_generic_unavailable("register_slab");
}

static iree_status_t iree_async_generic_proactor_import_fence(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t fence,
    iree_async_semaphore_t* semaphore, uint64_t signal_value) {
  (void)base_proactor;
  (void)fence;
  (void)semaphore;
  (void)signal_value;
  return iree_async_generic_unavailable("import_fence");
}

static iree_status_t iree_async_generic_proactor_export_fence(
    iree_async_proactor_t* base_proactor, iree_async_semaphore_t* semaphore,
    uint64_t wait_value, iree_async_primitive_t* out_fence) {
  (void)base_proactor;
  (void)semaphore;
  (void)wait_value;
  *out_fence = iree_async_primitive_none();
  return iree_async_generic_unavailable("export_fence");
}

static const iree_async_proactor_vtable_t iree_async_generic_proactor_vtable = {
    .destroy = iree_async_generic_proactor_destroy,
    .query_capabilities = iree_async_generic_proactor_query_capabilities,
    .submit = iree_async_generic_proactor_submit,
    .poll = iree_async_generic_proactor_poll,
    .wake = iree_async_generic_proactor_wake,
    .cancel = iree_async_generic_proactor_cancel,
    .create_socket = iree_async_generic_proactor_create_socket,
    .import_socket = iree_async_generic_proactor_import_socket,
    .destroy_socket = iree_async_generic_proactor_destroy_socket,
    .import_file = iree_async_generic_proactor_import_file,
    .destroy_file = iree_async_generic_proactor_destroy_file,
    .create_event = iree_async_generic_proactor_create_event,
    .destroy_event = iree_async_generic_proactor_destroy_event,
    .register_event_source = iree_async_generic_proactor_register_event_source,
    .unregister_event_source =
        iree_async_generic_proactor_unregister_event_source,
    .create_notification = iree_async_generic_proactor_create_notification,
    .create_notification_shared =
        iree_async_generic_proactor_create_notification_shared,
    .destroy_notification = iree_async_generic_proactor_destroy_notification,
    .notification_signal = iree_async_generic_proactor_notification_signal,
    .notification_wait = iree_async_generic_proactor_notification_wait,
    .register_relay = iree_async_generic_proactor_register_relay,
    .unregister_relay = iree_async_generic_proactor_unregister_relay,
    .register_buffer = iree_async_generic_proactor_register_buffer,
    .register_dmabuf = iree_async_generic_proactor_register_dmabuf,
    .unregister_buffer = iree_async_generic_proactor_unregister_buffer,
    .register_slab = iree_async_generic_proactor_register_slab,
    .import_fence = iree_async_generic_proactor_import_fence,
    .export_fence = iree_async_generic_proactor_export_fence,
    .set_message_callback = NULL,
    .send_message = NULL,
    .subscribe_signal = NULL,
    .unsubscribe_signal = NULL,
};

iree_status_t iree_async_proactor_create_generic(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  (void)options;
  *out_proactor = NULL;
  iree_async_generic_proactor_t* proactor = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, sizeof(*proactor),
                                             (void**)&proactor));
  memset(proactor, 0, sizeof(*proactor));
  iree_async_proactor_initialize(&iree_async_generic_proactor_vtable,
                                 IREE_SV("generic"), allocator,
                                 &proactor->base);
  *out_proactor = &proactor->base;
  return iree_ok_status();
}
