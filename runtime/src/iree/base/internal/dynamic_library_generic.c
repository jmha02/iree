// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/dynamic_library.h"

#if defined(IREE_PLATFORM_GENERIC)

struct iree_dynamic_library_t {
  int unused;
};

iree_status_t iree_dynamic_library_load_from_file(
    const char* file_path, iree_dynamic_library_flags_t flags,
    iree_allocator_t allocator, iree_dynamic_library_t** out_library) {
  (void)file_path;
  (void)flags;
  (void)allocator;
  *out_library = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "dynamic libraries are unavailable on Generic");
}

iree_status_t iree_dynamic_library_load_from_files(
    iree_host_size_t search_path_count, const char* const* search_paths,
    iree_dynamic_library_flags_t flags, iree_allocator_t allocator,
    iree_dynamic_library_t** out_library) {
  (void)search_path_count;
  (void)search_paths;
  return iree_dynamic_library_load_from_file(NULL, flags, allocator,
                                             out_library);
}

iree_status_t iree_dynamic_library_load_from_memory(
    iree_string_view_t identifier, iree_const_byte_span_t buffer,
    iree_dynamic_library_flags_t flags, iree_allocator_t allocator,
    iree_dynamic_library_t** out_library) {
  (void)identifier;
  (void)buffer;
  return iree_dynamic_library_load_from_file(NULL, flags, allocator,
                                             out_library);
}

void iree_dynamic_library_retain(iree_dynamic_library_t* library) {
  (void)library;
}

void iree_dynamic_library_release(iree_dynamic_library_t* library) {
  (void)library;
}

iree_status_t iree_dynamic_library_lookup_symbol(
    iree_dynamic_library_t* library, const char* symbol_name, void** out_fn) {
  (void)library;
  (void)symbol_name;
  *out_fn = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "dynamic symbols are unavailable on Generic");
}

void* iree_dynamic_library_try_lookup_symbol(iree_dynamic_library_t* library,
                                             const char* symbol_name) {
  (void)library;
  (void)symbol_name;
  return NULL;
}

iree_status_t iree_dynamic_library_attach_symbols_from_file(
    iree_dynamic_library_t* library, const char* file_path) {
  (void)library;
  (void)file_path;
  return iree_ok_status();
}

iree_status_t iree_dynamic_library_attach_symbols_from_memory(
    iree_dynamic_library_t* library, iree_const_byte_span_t buffer) {
  (void)library;
  (void)buffer;
  return iree_ok_status();
}

iree_status_t iree_dynamic_library_append_symbol_path_to_builder(
    void* for_symbol, iree_string_builder_t* builder) {
  (void)for_symbol;
  (void)builder;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "dynamic symbol paths are unavailable on Generic");
}

#endif  // IREE_PLATFORM_GENERIC
