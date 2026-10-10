# Changelog

## 1.3.0: Image conversations resume (2026-10-10)

- Port Mia's AI Lab's `0054-glm-image-prompt-reuse` as patch `0089-glm-image-prompt-reuse` (89 patches total):
  image prompts resume from kept prompt states under `--parallel`, keyed by each picture's content (a 62-bit
  key of its kind, shape and pixels in place of its placeholder tokens), and only the pictures past the resume
  point are encoded again. The spill tier (0088) stores and restores image states by their keyed ids, so
  screenshot-bearing agent sessions now resume like text ones -- the failure mode that motivated this port.
- Port her `0079-glm-picture-cache` unchanged as patch `0090-glm-picture-cache`: a picture's decode, fit and
  hash are remembered keyed by its source bytes (29.8 ms a picture measured on two Sparks), with the last few
  canvases held for the tower.
- Adaptations to this series: the int64-array resume comparison is kept (`_key_array` beside `_ids_array`),
  the take-over and shared-prefix comparisons match keys, and kept-cap victims stay plain LRU.
- The one-stream engine (`--parallel 1`) still resumes no image state. Exactness gates for this entry were
  run for text (unchanged hashes vs the 1.2.0 baseline); image-turn resume equality is checked by the probe
  in the validation log.

## 1.2.0: Spill tier (2026-10-09)

- Port Mia's AI Lab's disk tier for kept prompt states as patch `0088-glm-spill-tier` (87 patches total):
  her DGX-Spark `0088` by @wojo (Apache-2.0, PR #78), adapted to this series (no display-span row moves,
  no picture content keys, LRU kept-cap victims, and the scheduler hooks on this recipe's plain and ranked
  admission paths). Off by default; `SPILL_GIB` turns it on.
- An evicted or restarted conversation resumes from disk instead of re-prefilling; with `PARALLEL` above 1 the
  states eviction would take next are written early past `SPILL_HIGHWATER`; a clean stop writes what is kept
  within `TF_SPILL_FLUSH_S` and `./stop.sh` waits for it.
- New settings `SPILL_GIB`, `SPILL_DIR`, `SPILL_HIGHWATER`, `SPILL_MIN_TOKENS`, `SPILL_MIN_FREE_GIB`,
  `SPILL_KEEP_BUILDS`, `TF_SPILL_FLUSH_S` (plus `TF_SPILL_WRITERS`, `TF_SPILL_DIRECT`), validated by the launcher
  and budgeted in its disk check. `spill` counters in `/health` and `/metrics`.
- Stored states leave a snapshot's warm-turn rendering record out (`TF_W1_INCREMENTAL`): a restored prompt
  re-renders on its next turn. Prompts with images are never written.
- The series applies strictly to TensorFold v0.6.6 (no fuzz, offsets, `.rej` or `.orig`); the applied tree
  compiles. GPU exactness gates (`tf_bench gate gatelong`) were not run for this entry; run them before
  production use, and keep references per numerical configuration.

## 1.1.0: TensorFold v0.6.6 rebase (2026-10-08)

- Rebase the 86-patch recipe onto TensorFold v0.6.6 at `cb2ebf0540f42604e2759b2ddef497861e928248`.
- Merge the new `--name-priority ID=background` server option with the recipe's priority lanes.
  Explicit request priorities and title requests keep their existing behavior.
- Regenerate six patches for the rebase. The other 80 remain byte-for-byte unchanged.
  Mia's AI Lab's four regenerated patches change hunk positions only. Her 54 patches retain her attribution.
- Strict patch replay matches the expected tree. All 31 standalone CPU priority cases pass.
  The upstream name-priority pytest module was skipped because `tokenizers` was unavailable.
- A/B validation against v0.6.5 used the same checkpoint and settings on Server Edition GPUs.
  Gate and gatelong results match. First tokens match 40/40. Tool bursts remain 78/90 with default
  choice and 90/90 with required choice. Median speeds remain within about 1%, with no regression observed.
  The name-priority smoke passed. Published Max-Q measurements remain attributed to v0.6.5;
  retained older quality scores keep their original labels. See [validation notes](docs/NOTES.md#tensorfold-v066).

## 1.0.0: 2x RTX PRO 6000 recipe (2026-10-03)

- Two-GPU recipe by Aevonix Research for one Linux host with two RTX PRO 6000 Blackwell GPUs.
- One-command setup and launch: checks, pinned image build and model downloads, kernel compilation,
  one-time TP2 launch-table autotune, API readiness and a smoke test. Restart, dry run and saved log archives.
- TensorFold v0.6.2 with the same 87 patches: 55 from Mia's AI Lab and 32 from Aevonix Research.
- The image build includes TensorFold's C/CUDA package headers for runtime compilation (also shipped in the 4-GPU recipe's 1.0.1).
- Defaults: q4 dense, FP8 KV, stock DFlash2, four concurrent requests, 524,288-token context per request,
  32,768 maximum reply tokens, vision on and port 8022. Two full 512K sessions fit in one shared
  1,062,912-token KV pool with DFlash2 extents, using a 3.75 GiB spare-cache cap and 6 GiB memory reserve per GPU.
- Measurements use 300 W per GPU. Startup warns with `W_POWER` when a selected GPU differs and never changes power.
- TP2 kernel-cache isolation, a smaller verify window and prompt chunks, sharded lane projections,
  and exactness checks with two and four concurrent requests. Engine patches are unchanged.
- The [README](README.md) receives final performance, memory and quality measurements from the release facts file.
  Images passed 3/3 checks. FP8 dense is not measured in this two-GPU release.
