# Full-context native QLion training verification

The launcher invokes `llama-finetune-qlion` directly with the existing CLI. It selects CUDA0 and CUDA1 with tensor splitting, context 16384, recomputation, and a CPU gradient accumulator. CUDA2 (P620) is excluded. No new trainer CLI option was added.

Single-sequence Gemma4/Lumen training uses fixed projection tiles of 512 tokens and fixed query/FFN/head tiles of 256 tokens. The requested ubatch must be at least min(256, context). The complete window remains differentiable, including prefix K/V. Fixed tiles preserve the arithmetic and reduction order when ubatch changes. Layer checkpoints and reduced gradients use host RAM. Attention mask views are shared across layers.

`changes.patch` records this task's delta against the saved working files, including pre-existing local modifications in the baseline. It is not a patch against upstream HEAD. `source-hashes.json` identifies the verified production sources. Unrelated working-tree changes were preserved.

Evidence:

- `unit-cpu-cuda-meta.log`: CPU, CUDA and tensor backend gradient, offload, split allocation, copied metadata and serialization checks.
- `gpu-integration.json`: 32 executions, dense/MoE, QLion/QLion2, four ubatches, 16 updates, exact packed weights and optimizer tensors, nonzero momentum.
- `small-16k-tensor.json`: eight tensor executions at context 16384, dense/MoE, four ubatches, two updates, exact weights and state.
- `kd-tensor.json`: six tensor distillation executions including shared K/V, gradient accumulation and partial final groups.
- `validation.json`: actual Lumen run results and limitations.
- `unsloth-reference.json`: settings extracted from the user-supplied official notebook.

The actual Lumen smoke uses one training window and one validation window. The ub768 run saves a full GGUF directly; the ub256 run saves an adapter with optimizer state, then `llama-export-qlion` merges it with the original base for the full tensor comparison. The first QLion2 update initializes nonzero momentum while the packed weights remain unchanged; the fixture regressions cover multi-step weight trajectories. It does not establish full-epoch stability or throughput. Bit equality is scoped to identical hardware/backend, threads and other settings; it is not equivalence to the previous variable-tile graph or across different devices.

Unsloth speed and memory superiority is unverified. GPT-OSS 20B QLoRA and Lumen native QLion2 have different models, parameter updates and training objectives. The Lumen 16K placement uses two V100 16GB cards. It is not evidence of fitting this run on one 16GB card.
