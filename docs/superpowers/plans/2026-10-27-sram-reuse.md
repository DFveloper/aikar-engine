# CUDA training SRAM reuse implementation plan

**Goal:** Improve P620 single and parallel training throughput without changing optimizer cadence or training semantics.
**Architecture:** Keep the existing tiled matrix multiplications. Parallelize long repeat-back reductions, fuse same-shape F32 MUL followed by reduction when the intermediate has no other users, and cache attention score/probability intermediates on chip for bounded rows. Reuse existing fusion dependency and memory-range checks. Retain generic paths for other shapes.
**Constraints:** Preserve all pre-existing changes, CPU behavior, checkpoints, and QLion2 statistics. No commits or pushes. Existing user authorization permits implementation choices and stopping test training.

- [x] Preserve baseline binaries and current checkpoint; measure P1 and P2 with identical data and arguments.
- [x] Extend backend reduction tests with long, strided, parallel and fused product cases; record baseline timings as performance regression evidence.
- [x] Implement block-parallel reduction and safe MUL fusion in binbcast.cu/cuh and ggml-cuda.cu.
- [x] Implement bounded shared-memory attention softmax and register reuse in softmax gradient; preserve masks, sinks and fallback.
- [x] Build, compare real training P1/P2, check finite weights and loss, and run CUDA backend tests and memcheck. Reject regressions.
- [x] Review final diff and resume the foundation run from the latest complete checkpoint; save commands, profiles, results and source in the requested test directory.

Review focus: noncontiguous source tensors; products with additional users/output flags; partially masked attention rows; reduction floating-point error; large rows beyond shared memory bounds; parallel tail batches and checkpoint resume.

Decisions and review notes:
- Retain existing matrix-multiplication tiling; no cross-update weight caching, which would use stale weights.
- Add a packed-row offset path to avoid repeated 64-bit division for dense reductions.
- Use warp shuffles plus shared warp partials for attention reductions. This changes floating-point addition order; exact optimizer trajectories are not guaranteed.
- Independent review found no blocking correctness issues. Added requested dim2/dim3 strided reduction, independently strided MUL operand, additional-user and OUTPUT-flag cases.
- Final verification uses whole-graph execution for fusion cases; per-node comparison would not execute the fusion.

Verification: P1/P2 throughput both +15.7%, fixture VRAM unchanged. CPU/CUDA backend42/42 each; memcheck0errors; modified-kernel racecheck0hazards; related host CTest5/5. Old cuBLAS racecheck hazard reproduced against pre-change library, retained as a vendor limitation. Register cache uses91registers/thread and0profiled local-memory bytes/thread. P2/A2/micro64 training30steps: loss9.03796848 ->8.34708826; all86model and237state tensors finite,79nonzero momenta.
