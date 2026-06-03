# PAFKIP TTA IREE/Saturn Patch Summary

This checkout is a shareable upstream-style IREE tree for the PAFKIP on-device
TTA spike. It was cloned from `https://github.com/jmha02/iree.git` and only
ports the changes needed for the IREE + RISC-V core + Saturn/FlexiNPU path.

## Scope

Included:

- FlexiNPU dialect and RoCC lowering support.
- RISC-V/Saturn compiler plumbing for `+flexinpu`.
- PAFKIP/TTA reduction tiling support for BN/channel-style reductions.
- Runtime workarounds needed by the local-sync/static RISC-V execution path.

Excluded:

- Gemmini support.
- Overlap, row-bundle, and other research optimization patches.
- Extra Flexi/private repository changes unrelated to PAFKIP TTA.

## Compiler Changes

### FlexiNPU dialect

Added `compiler/src/iree/compiler/Dialect/FlexiNPU/`.

Purpose:

- Represents FlexiNPU DMA/load/store/execute/flush and generic RoCC instruction
  operations in MLIR.
- Provides the dialect used by the Saturn/FlexiNPU lowering path.
- Adds compatibility builders required by the current upstream MLIR generated op
  wrappers.

### RoCC/FlexiNPU lowering

Added `compiler/src/iree/compiler/Codegen/LLVMCPU/RoCC/`.

Purpose:

- Tensorizes supported linalg matmul/vector-matmul patterns into FlexiNPU ops.
- Lowers FlexiNPU ops into helper calls, then into RoCC inline assembly.
- Keeps only FlexiNPU support. Gemmini-specific code was intentionally removed.

### LLVMCPU pass registration

Changed:

- `compiler/src/iree/compiler/Codegen/LLVMCPU/CMakeLists.txt`
- `compiler/src/iree/compiler/Codegen/LLVMCPU/Passes.cpp`
- `compiler/src/iree/compiler/Codegen/LLVMCPU/Utils.{h,cpp}`

Purpose:

- Builds and registers the RoCC/FlexiNPU passes in the LLVMCPU codegen pipeline.
- Adds `hasFlexiNPUFeature(...)`, gated by the `+flexinpu` CPU feature string.

Note:

- LLVM itself does not know `+flexinpu`, so `iree-compile` may print an LLVM
  warning saying the feature is ignored by the target. IREE still sees the
  feature string and uses it for the FlexiNPU path.

### FlexiNPU to LLVM lowering

Changed:

- `compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp`

Purpose:

- Registers the FlexiNPU dialect as a dependent dialect.
- Lowers `flexinpu.rocc_instruction` to RISC-V RoCC inline assembly.
- Lowers `flexinpu.dummy_memref` into the LLVM memref ABI shape expected by the
  helper path.

### BN/channel reduction tiling

Changed:

- `compiler/src/iree/compiler/Codegen/LLVMCPU/KernelDispatch.cpp`

Purpose:

- Detects projected reductions with one output dimension, which appear in
  BN/statistics/update-style PAFKIP TTA kernels.
- Scalarizes the parallel distribution tile for those reductions to avoid
  problematic distribution shapes on the RISC-V/Saturn path.

## Runtime Changes

Changed:

- `runtime/src/iree/hal/buffer.c`
- `runtime/src/iree/hal/buffer_heap.c`
- `runtime/src/iree/hal/command_buffer.h`
- `runtime/src/iree/hal/command_buffer_validation.c`
- `runtime/src/iree/hal/local/executable_library.h`
- `runtime/src/iree/modules/hal/module.c`

Purpose:

- Allows transient/static local-sync bindings whose memory metadata is reported
  as `NONE`.
- Avoids premature discard of buffers needed across the local-sync/static
  RISC-V execution path.
- Recovers zero-length command-buffer bindings when the underlying buffer has a
  valid byte length.
- Raises HAL binding limits for the larger PAFKIP/TTA loop module.
- Adds a diagnostic print for the specific zero-byte binding case.
- Adjusts split heap allocation lifetime handling used by this path.

Risk:

- These runtime changes are pragmatic spike workarounds, not polished upstream
  API changes. They should be narrowed or replaced with a proper static
  RISC-V/Saturn HAL allocation/binding model before upstreaming.

## Upstream-equivalent Fixes Not Ported

The following Flexi-local fixes were not applied because this upstream checkout
already has equivalent behavior:

- Hoist-into-globals erase-order fix.
- NCHW/channel-first convolution lowering support.

## Build And Smoke Tests

Environment used:

```bash
set +u
cd /root/flexi
source env.sh
set -u
cd /root/iree-upstream
unset LD_PRELOAD
```

Configure:

```bash
cmake -G Ninja -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_PYTHON_BINDINGS=OFF
```

Verified:

```bash
cmake --build build --target iree-compile -j$(nproc)
cmake --build build --target iree-run-module -j$(nproc)
build/tools/iree-compile --version
```

Compile smoke tests:

- Host LLVMCPU VMFB generation passed.
- RISC-V LLVMCPU VMFB generation passed with:

```bash
--iree-hal-target-backends=llvm-cpu
--iree-llvmcpu-target-triple=riscv64-unknown-elf
--iree-llvmcpu-target-cpu=generic-rv64
--iree-llvmcpu-target-abi=lp64d
--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+zvl128b,+flexinpu
```

Runtime smoke test:

```bash
build/tools/iree-run-module \
  --module=/tmp/iree_upstream_add_host.vmfb \
  --device=local-sync \
  --function=add \
  --input=4xf32=1,2,3,4 \
  --input=4xf32=10,20,30,40
```

Observed result:

```text
4xf32=11 22 33 44
```

Hygiene checks:

```bash
git diff --check
rg -n "Gemmini|gemmini|DispatchOverlap|overlap|row_bundle|ExtractRowBundle|GemminiFuncParams|injectRocc" \
  compiler/src/iree/compiler/Codegen/LLVMCPU/RoCC \
  compiler/src/iree/compiler/Dialect/FlexiNPU \
  compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp \
  compiler/src/iree/compiler/Codegen/LLVMCPU/CMakeLists.txt \
  compiler/src/iree/compiler/Codegen/LLVMCPU/Passes.cpp \
  compiler/src/iree/compiler/Dialect/HAL/IR/HALDialect.cpp \
  compiler/src/iree/compiler/Dialect/HAL/IR/CMakeLists.txt
```

Both checks passed.
