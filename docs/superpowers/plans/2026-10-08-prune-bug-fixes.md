# Pruning bug fixes

Fix the confirmed session bugs without changing importance formulas or adding architectures. Work in the existing dirty workspace and preserve unrelated edits. No commit or PR.

1. Reproduce scaled aikar output loss and routing view duplication with the existing test file and a callback harness. Record each normalized selection once, collect final pre-gate output for both metrics, and check scalar/count overflow.
2. Require corrected collector version for every metric. Remove legacy cache migration; record tokenized dataset, execution/build/environment fingerprint and carry provenance through profiles/reports. Reject malformed or mismatched cache statistics. Detect colliding ratio filenames before calibration.
3. Publish hard model/report with report invalidation and rollback, preserving old output on failure and preventing source aliases. Keep output files absent rather than misleading after interruption. Test filesystem failure and rollback through a rename failure injection harness.
4. Run CPU and CUDA/Vulkan test-moe-prune, synthetic callback/cache/publication checks, and tiny real-model CPU calibration. Verify all three metrics share identical observations and the aikar/REAP conditional score matches. Update usage and fix report; preserve historical audit evidence.
