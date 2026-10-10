# Measurement and validation notes

This page records current validation and retains historical v0.6.2 notes. Published performance
measurements in the [README](../README.md) come from the v0.6.5 build. Older quality scores retain
their original labels. Engine-port details are in [the port table](PORT-v0.6.5.md).

This two-GPU recipe uses two RTX PRO 6000 Blackwell GPUs with CUDA peer access on one PCIe host.
The [README](../README.md) receives the measured performance, memory and quality tables from
its release facts file. The measurements describe the tested pair; they do not predict another topology.
Vision is enabled by default. Historical image checks passed 3/3. FP8 dense has not been measured in this two-GPU release.

## TensorFold v0.6.6

The recipe now runs TensorFold v0.6.6 at `cb2ebf0540f42604e2759b2ddef497861e928248`.
The 2026-10-08 A/B check compared A, the current v0.6.5 build, with B, the v0.6.6 rebase,
using the same checkpoint and settings. Flash2x used two GPUs on the 4x RTX PRO 6000 Blackwell
Server Edition validation host, at 600 W per GPU. This is a regression check on a different host
from the published Max-Q measurements. It does not reproduce those throughput measurements.

Gate and gatelong passed identically at concurrency levels 2 and 4. Long probes covered about
8K, 32K and 100K tokens. First tokens and complete returned sequences matched 40/40.
Tool bursts were identical between A and B: 78/90 with default choice and 90/90 with required
choice. The 12 default-choice misses remain failures in both builds. These checks establish
identity for the tested checkpoint and settings, not original-model fidelity or broad model quality.

Median single-stream and four-stream decode and cold 8K/32K prefill rates stayed within about
1%, with no regression observed. Timing used A0/B0/B1/A1 blocks with six observations per arm.
The original A1 block lacked the matching priority prime and was replaced with a correctly primed
block. A B0 four-stream outlier remains in the final spread. These samples do not establish a
speed gain or zero overhead at sub-percent precision. Validation ran natively because Docker
was unavailable; host and runtime differences further limit comparison with the published container results.

The `--name-priority ID=background` smoke passed. The engine's model-name background default
is merged with the recipe's priority lanes when request priority is omitted. Explicit priorities
still win, and title requests remain background. Without the option, B retains A's behavior.

The 89 patches replay strictly to the expected tree. Six patch files were regenerated for the
rebase and 80 remain byte-for-byte unchanged. Mia's AI Lab's four regenerated patches change
hunk positions only, and her attribution is retained. The standalone CPU priority check passed
31/31 cases. The upstream pytest module was skipped because `tokenizers` was unavailable.
See [the rebase notes](PORT-v0.6.5.md#rebase-to-v066) for patch details.

## Keep the workload fixed

Use the same client, prompts, token budget, sampling settings and weights for every TensorFold run.
Keep aggregate decode separate from per-request speed, and decode-only speed separate from time to
first token. Cold prefill and cached warm turns are different workloads.

TensorFold serves an omitted `top_k` as 20. Send `top_k` explicitly when checking sampled outputs and
record the effective setting. Keep fixed seeds, drafter, precision and prompt arithmetic in both arms.

Exactness references belong to a checkpoint, precision, TP layout and prompt arithmetic. Compare
drafted replies against `"draft": false`, concurrent against solo, and long prompts as well as short
prompts. q4 dense, FP8 KV and chunked KDA need their own references. Matching those references does
not establish BF16 fidelity. Use the [retained tools](../tools/bench/README.md) with gate levels `2,4`.
Add `--gate-vision` to include the image gate with the default `VISION=1` server configuration.
Small functional probes do not establish broad model quality.

## TP2 defaults and memory

Final defaults use 300 W per GPU, a 524,288-token context per request, and a shared 1,062,912-token
KV pool that accommodates two full 512K sessions plus DFlash2 extents. Prompt chunks have 1,024 rows;
`KV_POOL_GIB=3.75` and `MEMORY_RESERVE_GIB=6`. The startup estimate is 83.89 GiB of an 87.92 GiB
engine budget per GPU. The README records sampled physical memory peaks separately from that estimate.

Both ranks use the same checkpoint and settings. Prompt and index exchange use the actual rank
count. The two prompt lanes are independent of GPU count. The sparse fast kernel requires a
four-rank head shape, so this two-GPU recipe disables it and keeps its reference path. Lane-input weight replication
and two-shot decode exchange are off. A smaller verify window, prompt chunks and kept-state count
leave room for the shared KV pool. The first run tunes TP2 kernel shapes and checks each candidate
against the reference; cached tables are separated by image hash and TP count.

`CONTEXT` includes prompt and reply for one request. Concurrent long requests share the pool rather
than receiving a full pool each. `KV_POOL_GIB` caps spare cache beyond the base context; it is not the
total KV allocation. The engine allocation budget already excludes `MEMORY_RESERVE_GIB`, so do not
subtract that reserve again. Physical peak use includes CUDA, communicators, graphs and allocator
workspaces outside the planner's named weight estimate. See the generated memory table in the README.
The vision tower uses 1.05 GiB plus 0.75 GiB of workspace on GPU 0; `VISION=0` frees this memory.

Keep first-use exactness checks and fallback kernels enabled. Changing quantization, concurrency,
context, vision, prompt chunks, transport or graph settings requires fresh validation. The scripts
check free memory and peer access; they do not change GPU power limits or platform settings.

## Reproduce the patch tree without GPUs

The current series has **89 patches**, 58 from Mia's AI Lab and 32 from Aevonix Research, applied in
global filename order to TensorFold v0.6.6 at `cb2ebf0540f42604e2759b2ddef497861e928248`.

```bash
python3 tools/check_apply.py --release-patches /path/to/reference-patches \
  --upstream /path/to/unmodified-TensorFold
```

The check reads the reference and upstream clone without changing them, applies both series to
temporary trees and compares their files and modes. No network or GPU is required. Without
`--upstream`, it verifies patch identity and global order only. This packaging check does not rerun
GPU measurements or validate a new host.
