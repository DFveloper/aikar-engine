# GSQ/RCO PoC implementation plan

Goal: Validate Gemma 4 E2B GSQ and RCO components with real GGUF inputs and measured results.

Architecture: Independent C++ optimization tool; existing GGML matmul, codecs, llama calibration callback, and GGUF writer. Preserve source models. No commits or pushes.

Execution: Inline. User requirements and the preceding technical design define the scope.

1. Record resources, pinned model revision and SHA256; verify metadata and CPU inference.
2. Add failing numerical checks for GSQ gradients, two-beta Lion, Q2_0 packing and RCO geometry/budget. Extend existing test infrastructure.
3. Implement FP32 GSQ with reproducible samples and existing GGML reconstruction matmul/weight gradients. Train one actual Gemma linear from captured activations.
4. Stream GGUF replacement with direct integer packing; reload and compare held-out logits/PPL/KL against teacher and matching PTQ.
5. Implement RCO projection/retraction/first-moment transport and exact byte-budget assignment; create actual Q2_0/Q4_0/Q8_0 candidates. Label reconstruction proxies separately from global RCO.
6. Expand block reconstruction, full-Hessian initialization and global RCO only after lower stages pass and resource measurements permit it. Record unimplemented stages explicitly.
7. Verify existing quantization/optimizer tests, inspect Lumen metadata locally, record artifacts and limitations, restore any authorized stopped service.

Constraints: No original model overwrite, no private checkpoint upload, no unsupported format fallback. Calibration/evaluation inputs disjoint. C++ source in worktree, experiments under /mnt/openwebui/AIKAR/GSQ-RCO.

Review focus: RNG replay, scale sign, partial blocks, invalid/nonfinite inputs, infeasible budget, tensor dimensions, no-clobber output and cleanup on failure.

Progress: Model downloaded and SHA256 verified against pinned Hugging Face LFS. Native FP32 GSQ, exact Q2_0 packing and RCO manifold/proxy modules implemented. One Gemma linear trained with native RTN and upstream full-Hessian GPTQ priors. Five partial GGUF outputs reloaded and evaluated. CPU/CUDA gradient oracles and quantization regression passed. Actual three-tensor RCO and GSQ+RCO candidates stayed within the GGUF byte budget. Existing service restored and health checked. Block reconstruction, whole-model streaming and global CE/KL RCO remain unimplemented.
