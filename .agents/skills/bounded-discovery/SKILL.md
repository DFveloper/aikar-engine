---
name: bounded-discovery
description: Use when a repository investigation or performance task requires locating implementation, tests, build commands, and local assets across large documentation or filesystem trees.
---

# Bounded repository discovery

Discovery should produce the next runnable experiment. Keep room for implementation, verification, and every requested artifact.

1. Keep a short checklist of the requested workstreams and deliverables. For each workstream, collect the owning source/symbol, a related existing test, the project's build/run command, and any required input path. Keep unresolved items explicit; one workstream's history search must not consume the others' execution budget.
2. Read required instructions once and retain their paths and applicable constraints. On revisiting a file, retrieve the section needed for the current decision rather than dumping the whole file again. For optional documentation, search headings or symbols with `rg -n`, then read the relevant range. A shared model or feature name alone does not make an entire historical report relevant.
3. Separate searches for code, runnable assets, and historical logs. Start in the request's source directories and known asset roots. Filter by the actual file type and role before limiting displayed results. Do not mix old logs, checkpoints, release archives, and model candidates into one `head`-limited stream; early matches can hide the required input. If the first focused asset search is empty, expand to its parent or one additional plausible root once, then record searched locations and the missing input. Repeating the same search is not new evidence.
4. Enough discovery means each workstream has a concrete next command and the source/test locations needed to interpret it, or a specific evidenced blocker. Run the smallest relevant baseline/build next; reopen discovery only for a question raised by source inspection or execution. Preserve the task's required verification and deliverable shape. Before finishing, reopen changed files or inspect the diff and read the actual check output; a successful write is not verification.
