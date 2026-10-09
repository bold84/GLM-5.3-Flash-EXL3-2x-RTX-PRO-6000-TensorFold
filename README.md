<!-- releasekit:header:start -->
<p align="center">
  <a href="https://aevonix.com">
    <img src=".github/aevonix-logo.png" alt="Aevonix Research" width="360">
  </a>
</p>

<h1 align="center">GLM-5.3-Flash EXL3 on 2x RTX PRO 6000 with TensorFold</h1>

<p align="center">A one-command recipe for 2x RTX PRO 6000, with text and vision on one Linux host.</p>

<p align="center"><sub>by <a href="https://aevonix.com">Aevonix Research</a> · in collaboration with <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub></p>

<p align="center">
  <a href="https://aevonix.com"><img src="https://img.shields.io/badge/Aevonix_Research-aevonix.com-EEB07E?style=for-the-badge&amp;labelColor=151615" alt="Aevonix Research: aevonix.com"></a>
  <a href="https://huggingface.co/Mia-AiLab"><img src="https://img.shields.io/badge/In_collaboration_with-Mia%27s_AI_Lab-D99288?style=for-the-badge&amp;labelColor=151615" alt="In collaboration with Mia&#x27;s AI Lab"></a>
  <a href="https://github.com/ashhart/TensorFold"><img src="https://img.shields.io/badge/TensorFold-v0.6.6-A9D5CE?style=for-the-badge&amp;labelColor=151615" alt="TensorFold v0.6.6"></a>
  <img src="https://img.shields.io/badge/GPUs-2x_RTX_PRO_6000-EEB07E?style=for-the-badge&amp;labelColor=151615" alt="2x RTX PRO 6000">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-D99288?style=for-the-badge&amp;labelColor=151615" alt="Apache-2.0"></a>
</p>

<p align="center"><img src=".github/image.png" alt="GLM-5.3-Flash EXL3 on 2x RTX PRO 6000 with TensorFold" width="100%"></p>
<!-- releasekit:header:end -->

<!-- releasekit:summary:start -->
This one-command recipe serves **GLM-5.3-Flash** on one Linux host with **two 96 GB RTX PRO 6000 Blackwell GPUs**, through an
OpenAI-compatible API, with **4 concurrent requests**, **524,288 tokens per request**, tool calling
and structured outputs. Images are supported by default. This recipe combines [TensorFold](https://github.com/ashhart/TensorFold) v0.6.6,
Mia's AI Lab's EXL3 checkpoint and recipe patches, and Aevonix Research's single-host PCIe optimizations.
One command builds, downloads, tunes and starts it.

- **Checkpoint:** [`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold),
  EXL3 routed experts at 4 bits per weight, with dense layers converted to 4-bit at load time.
- **Drafter:** [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2),
  plus copy drafts. DFlash2 is non-commercial; `DRAFTER=mtp` uses the checkpoint's own MTP head instead.
- **API model id:** `glm-5.3-flash`.
- **Context:** 524,288 tokens per request, sharing one 1,062,912-token FP8 KV pool. Long requests share that pool.
- **Concurrency:** 4 requests by default, with a 32-row verification window.
- **One command:** `./start.sh`; restart with `./start.sh restart`, stop with `./stop.sh`.
<!-- releasekit:summary:end -->

## Performance

The recipe now runs TensorFold v0.6.6. Published performance measurements below come from the v0.6.5 build. A/B validation compared v0.6.5 and v0.6.6 with the same checkpoint and settings on a host with four RTX PRO 6000 Blackwell Server Edition GPUs. Flash2x used two of those GPUs. Gate and gatelong results were identical, first tokens matched 40/40, and tool bursts were unchanged at 78/90 for default choice and 90/90 for required choice. Speed medians stayed within about 1%, with no systematic regression. The `--name-priority` smoke passed. This was separate from the published Max-Q measurements. See [validation details](docs/NOTES.md#tensorfold-v066).

<!-- releasekit:performance:start -->
Two RTX PRO 6000 Blackwell Max-Q GPUs, one PCIe host.

q4 dense, FP8 KV, stock DFlash2, fnc7:0.3 and vision on.

[sparkDash](https://github.com/MiaAI-Lab/sparkDash) v1.8.9 measured TensorFold on 2026-10-04.

Power: **300 W per GPU**.

sampling: greedy; temperature: 0; top_p: 1; thinking: off; reply tokens: 400.

sparkDash Bench-controller workload. Decode aggregate is total throughput across concurrent requests; per-request speed and TTFT are medians. Prefill uses actual prompt token counts derived from the controller rate and TTFT.

Historical v0.6.2: At 250 W, an earlier run measured 226 tok/s single-stream prose and 416 tok/s at 4 concurrent requests.
<!-- releasekit:performance:end -->

**Decode** (aggregate across the concurrent requests, median per request, and median time to first token)

<!-- releasekit:decode:start -->
| Concurrent requests | Prose | Prose, per request | TTFT | Structured | Structured, per request | TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 238.3 tok/s | 238.3 tok/s | 41 ms | 445.0 tok/s | 445.0 tok/s | 34 ms |
| 2 | 325.6 tok/s | 163.9 tok/s | 54 ms | 702.9 tok/s | 354.1 tok/s | 48 ms |
| 4 | 448.7 tok/s | 114.9 tok/s | 72 ms | 651.2 tok/s | 256.7 tok/s | 63 ms |

With PARALLEL=4, waves of 8 and 16 requests queue behind the four slots, measuring 423.0 and 424.8 tok/s prose aggregate respectively.
<!-- releasekit:decode:end -->

Within a fixed configuration, the [exactness checks](#checks) compare replies served together with the same
requests served alone, and drafted replies with serial replies.

**Code** (the same decode measurements)

<!-- releasekit:code:start -->
| Concurrent requests | Code | Code, per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 350.8 tok/s | 350.8 tok/s | 48 ms |
| 2 | 486.0 tok/s | 247.7 tok/s | 64 ms |
| 4 | 584.5 tok/s | 163.3 tok/s | 327 ms |
<!-- releasekit:code:end -->

**Prefill** (cold prompts, with actual prompt token counts)

<!-- releasekit:prefill:start -->
cold prompts, unique prefixes.

| Prompt | Prefill | Time to first token |
| --- | ---: | ---: |
| 8,215 tokens | 5,729 tok/s | 1.43 s |
| 16,408 tokens | 6,307 tok/s | 2.60 s |
| 32,793 tokens | 6,884 tok/s | 4.76 s |
| 65,560 tokens | 6,908 tok/s | 9.49 s |
| 131,097 tokens | 6,730 tok/s | 19.48 s |
| 262,165 tokens | 6,372 tok/s | 41.14 s |
<!-- releasekit:prefill:end -->

**Prompt reuse** (separate cold-prompt and warm-conversation workloads)

<!-- releasekit:prompt-reuse:start -->
Separate client workload from sparkDash, measured alone. Cold values are medians of three fresh prompts; warm values are TTFT p50 for turns 2-5 of a conversation beginning near 130K tokens. Warm requests use temperature 0.6; thinking is explicitly off or enabled at low effort. Prompt sizes differ, so these are not matched cache-speedup ratios.

| Prompt | First time | Next time |
| --- | ---: | ---: |
| 32K-token prompt | 6.16 s | Not measured by this workload |
| 100K-token prompt | 19.63 s | Not measured by this workload |
| 140K-token prompt | 27.79 s | Not measured by this workload |
| A 130K-token conversation's next turn, thinking off | 25.907 s | 0.5115 s |
| A 130K-token conversation's next turn, low effort | 25.341 s | 0.5390 s |

| Thinking | First-turn actual tokens | Next-turn actual tokens | Warm maximum TTFT |
| --- | ---: | ---: | ---: |
| Off | 130,339 | 132,357, 134,388, 136,409, 138,425 | 0.515 s |
| Low effort | 129,936 | 131,954, 133,985, 136,006, 138,022 | 0.543 s |
<!-- releasekit:prompt-reuse:end -->

**Quality.** Small functional checks for this configuration.

<!-- releasekit:quality:start -->
| Benchmark | Score | Scope |
| --- | ---: | --- |
| Tool calls | 40/40 | Historical v0.6.2 measurement, not rerun for this port. Fixed original tools fixture; all missed cases count as failures |
| JSON, instruction only | 20/20 | Historical v0.6.2 measurement, not rerun for this port. Bare valid JSON, thinking off; fenced JSON does not count |
| JSON with response_format | 20/20 | Historical v0.6.2 measurement, not rerun for this port. Bare valid JSON, thinking off; fenced JSON does not count |
| Needles at 100K / 200K / 250K / 500K | 3/3 / 3/3 / 3/3 / 3/3 | Historical v0.6.2 measurement, not rerun for this port. Three inserted codes recovered from one request at each length |
| Images | 3/3 | Historical v0.6.2 measurement, not rerun for this port. Red, green and blue 256x256 PNGs; text before image, temperature 0, thinking off |
| Tool bursts, default choice / required | 78/90 / 90/90 | Historical v0.6.2 measurement, not rerun for this port. 30 bursts of three concurrent requests per mode; all default-choice misses retained |
| Effort mapping | PASS; 4/4 levels | Historical v0.6.2 measurement, not rerun for this port. Original tools/effort fixture; low/high/high/max mapping |
| Effort contract at 524,288 context | PASS; 5/5 cases | Historical v0.6.2 measurement, not rerun for this port. Original request cases with the current context expectation |

Historical functional and dense-fidelity scores are retained from the v0.6.2 recipe; they are not new v0.6.5 measurements.

Historical v0.6.2: These small fixed checks do not establish broad model equivalence. Scores include every case without selective retries; the default-choice tool burst had 12 model-output misses.

Historical v0.6.2: Images are supported by default; the final run passed red, green and blue PNG checks.

Historical v0.6.2: At TP2 q4, the two-GPU recipe client and tf_bench gate each passed drafted versus serial (8/8), sampled two-request (2/2) and greedy two-request (2/2) checks; tf_bench also passed sampled and greedy four-request checks (4/4 each), with serial references checked independently.

Historical v0.6.2: Exactness gates compare drafted versus serial and concurrent versus solo output within a fixed checkpoint, precision and prompt arithmetic; they do not establish TP4, FP8 dense, BF16 or original-model fidelity.

Historical v0.6.5 validation passed drafted versus serial, concurrent versus solo, image input, long-context saved hashes, and fresh versus resumed replay against a separately started v0.6.2 baseline. Exactness is within the fixed checkpoint, precision and prompt arithmetic and does not establish original-model fidelity.

These tables retain the first user-flow sweep, including its latency outliers. Matched v0.6.2/v0.6.5 runs reproduced the cold-prompt latencies. Initially larger decode gaps did not persist in a focused five-repeat comparison: the original configurations differed by less than one percent in the three investigated aggregate-rate medians. Swapping the saved autotune table did not consistently improve performance, so the original recipe defaults remain. First-use code TTFT spikes also occurred on v0.6.2. These limited samples do not establish zero overhead at sub-percent precision.

Details on the [model card](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) for the checkpoint's benchmark scores.
<!-- releasekit:quality:end -->

**Why 4-bit dense is the two-GPU default.** It leaves room for the target and drafter on two GPUs, with a fidelity cost.

<!-- releasekit:fidelity:start -->
Historical dense-format evidence from the v0.6.2 recipe. No dense-format fidelity comparison was measured in this final two-GPU run. The two-GPU recipe uses q4 dense to fit the two-GPU memory budget; FP8 dense was not tested on two GPUs. Dense and KV quantization change numerics, and chunked KDA changes prompt arithmetic. These checks do not establish fidelity to the original unquantized model.

The [full recipe](https://github.com/Aevonix/GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold) uses FP8 dense for quality.
<!-- releasekit:fidelity:end -->

Dense and KV quantization change numerics; chunked KDA also changes arithmetic. The exactness gates check drafted
versus serial and concurrent versus solo output within a fixed configuration. See [validation and limitations](docs/NOTES.md)
and [verification tools](tools/bench/README.md).

<!-- releasekit:cards:start -->
<p align="center">
  <img src=".github/sparkdash-decode.png" alt="sparkDash two-GPU recipe decode benchmark card" width="49%">
  <img src=".github/sparkdash-prefill.png" alt="sparkDash two-GPU recipe prefill benchmark card" width="49%">
</p>
<!-- releasekit:cards:end -->

## Requirements

- **Linux x86_64**, two **96 GB RTX PRO 6000 Blackwell** GPUs, working NVIDIA drivers and CUDA peer access
  between the selected GPUs. Keep them idle for the first compilation and launch-table autotune.
- **Docker with NVIDIA Container Toolkit**, a reachable Docker daemon and membership in the `docker` group
  (root can also run it). GPU access is tested inside the image before tuning.
- **At least 90,000 MiB free on each selected GPU** at launch. The defaults reserve 6 GiB per rank and cap
  spare KV cache at 3.75 GiB per rank; the engine checks whether the requested context fits.
- **Disk:** budget 180 GiB for the checkpoint, 10 GiB for DFlash2, 40 GiB for image/build space and 10 GiB for
  kernels. The script adds these requirements when locations share a filesystem and skips ready downloads.
- Bash, Python 3, Git, GNU `patch`, `curl`, `gzip`, GNU coreutils and `flock`. Initial setup needs access to
  GitHub, NVIDIA's container registry, package indexes and Hugging Face.
- Optional **`HF_TOKEN`**, or a Hugging Face token file. The public checkpoint can be downloaded without one.

## Quick start

```bash
git clone https://github.com/Aevonix/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold.git
cd GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold
./start.sh
```

<p align="center"><img src=".github/banner.gif" alt="Two-GPU recipe terminal opener with two-GPU defaults and measured decode results" width="100%"></p>

The first run checks the host, builds TensorFold v0.6.6 at the pinned commit with all 87 patches, downloads the pinned
checkpoint and drafter, compiles kernels and runs the one-time TP2 launch-table autotune. Keep the GPUs idle.
Later runs reuse compiled kernels and the matching launch table. The script starts two local ranks, waits for the API, sends a short greedy
smoke request and prints `GLM-5.3-Flash-EXL3 is now LIVE! on port 8022` with the endpoint.

To avoid DFlash2's non-commercial weights, use `DRAFTER=mtp ./start.sh` on the first run. This selects the
checkpoint's own MTP head and defaults to **one concurrent request**. Other weights and software retain their
own licenses. Put this setting in `scripts/local.sh` to keep it for future starts.

Preview every setup and Docker command without Docker or GPUs:

```bash
DRY_RUN=1 ./start.sh
```

The API base URL is `http://127.0.0.1:8022/v1`. The default bind address is `0.0.0.0`; remote clients use the
server's address. Requests can override the default thinking-off setting with `chat_template_kwargs`.

```bash
curl -s http://127.0.0.1:8022/v1/models
curl -s http://127.0.0.1:8022/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "glm-5.3-flash",
    "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
    "temperature": 0,
    "max_tokens": 2000
  }'

curl -s http://127.0.0.1:8022/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "glm-5.3-flash",
    "messages": [{"role": "user", "content": "Return a JSON object with status set to ok."}],
    "response_format": {"type": "json_object"},
    "temperature": 0,
    "max_tokens": 128
  }'

./start.sh restart
./stop.sh
docker logs -f glm53-tf-2x
curl -s http://127.0.0.1:8022/health
```

`stop.sh` stops the ranks and saves timestamped, gzipped logs under `logs/` before removing the container.
It keeps the newest **10** `glm53-*.tar.gz` archives (`LOG_KEEP`), containing both rank logs and Docker
output. Read a rank log with `tar -xOzf logs/<archive>.tar.gz ./rank0.log`; the live files are under
`logs/current/`. Restart saves the previous run the same way. Stopping interrupts active requests.

**Troubleshooting.** Launcher checks print stable codes on failure. Successful checks print `I_PLATFORM`, `I_GPU`, `I_DISK`,
`I_P2P` and `I_HF_TOKEN`; `I_DRY_RUN` means the checks and commands were only planned.

| Message | What to do |
| --- | --- |
| `E_CONFIG`: invalid setting | Use the allowed values in `scripts/config.sh`; MTP requires `PARALLEL=1`, and GPU indices must be distinct. |
| `E_PATCHES`: expected 87 patches | Restore the complete release's two patch folders. |
| `E_PLATFORM`: requires Linux x86_64 | Run this recipe on a supported Linux host. |
| `E_DEPENDENCY`: missing command | Install the named command from Requirements. |
| `E_DOCKER`: cannot talk to Docker | Start Docker and check access to its socket. |
| `E_DOCKER_GROUP`: login not in docker group | Add the account to the `docker` group and start a new login session. |
| `E_NVIDIA_RUNTIME`: NVIDIA runtime missing | Install and configure NVIDIA Container Toolkit for Docker. |
| `E_GPU`: cannot query GPU | Check the NVIDIA driver and the two indices in `GPUS`. |
| `E_GPU_MEMORY`: insufficient free memory | Stop other GPU work; each selected GPU needs at least `MIN_GPU_FREE_MIB`. |
| `E_GPU_BUSY`: GPU in use | Stop competing compute processes before startup and autotune. |
| `W_POWER`: selected GPU power limit differs | Published measurements used 300 W per GPU. Check the selected GPU's power limit; the launcher warns and never changes it. |
| `E_DISK`: insufficient space | Free space on the named filesystem or move `DATA_DIR`, model directories or Docker storage. Requirements are added when they share a filesystem. |
| `I_HF_TOKEN`: token absent | Public downloads can proceed. Supply `HF_TOKEN` or `HF_TOKEN_PATH` if access or rate limits require it. |
| `E_IMAGE`: build failed | Read the first build error; check disk space and access to the pinned upstream, base image and package indexes. |
| `E_DOWNLOAD`: download failed | Check free space, network access and optional token; rerun to resume the pinned download. |
| `E_P2P`: CUDA peer access failed | Check the driver and platform PCIe topology, ACS and IOMMU configuration. Every selected GPU pair must support CUDA peer access. |
| `E_AUTOTUNE`: compile/tuning failed | Read the compiler or kernel-check error, leave the GPUs idle and rerun. Incomplete tuning is never marked ready. |
| `E_PORT`: port in use | Stop its owner or change `PORT` / `MASTER_PORT`. |
| `E_LOCK`: operation in progress | Wait for the existing start, preparation or stop operation to finish. |
| `E_START`: container/rank failed | Inspect `logs/current/rank*.log` and `docker logs glm53-tf-2x`; `./stop.sh` archives them. |
| `E_API`: readiness timeout | Inspect rank 0's log; increase `WAIT_TIMEOUT` if first-run kernels are still compiling. |
| `E_SMOKE`: request failed or empty reply | Inspect `logs/current/smoke.json` and the rank logs, then restart. |
| `E_LOG`: cannot archive logs | Fix `LOG_DIR` space or permissions. The stopped container is retained to preserve evidence. |
| `E_STOP`: stop/remove failed | Check `docker ps -a`, correct the Docker error and retry. |

An engine context-budget refusal means the requested window does not fit that start's memory budget. Free
GPU memory or set a smaller `CONTEXT`; the launcher does not silently change the measured default.

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) to include the settings, startup code and relevant logs.

## Images and video

Images are supported by default (`VISION=1`). See the quality results for image-validation details.
The earlier v0.6.5 recipe also passed three image checks and served an image alongside two full
512K sessions; that concurrent full-context stress test was not repeated for this port.
`VISION=0 ./start.sh restart` disables the vision tower and frees its memory.
Send images as `image_url` content parts in a user message.
This example reads a local JPEG and sends it as a data URL:

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://127.0.0.1:8022/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}},
    {"type": "text", "text": "What is in this picture?"}]}],
  "temperature": 0,
  "max_tokens": 2000
}'
```

| Input | Content part |
| --- | --- |
| Image | `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}` |
| Video | `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}` |

The vision tower runs on rank 0. The recipe accepts data URLs; external media URLs are disabled.
The engine checks the memory budget again when vision is enabled.

## What `start.sh` and `scripts/prepare.sh` do

`./start.sh` shows each step as it runs:

1. Check the settings, Linux, Docker, NVIDIA runtime, GPU availability, free disk space and optional download token.
2. Build the pinned TensorFold v0.6.6 image and apply all 87 release patches, skipping a matching image.
3. Download the pinned checkpoint and selected drafter, skipping ready downloads.
4. Check CUDA peer access, compile kernels and autotune the TP2 launch table, reusing a matching cache.
5. Start one container with two local TensorFold ranks. Rank 0 serves the API and each rank writes its own log.
6. Wait for the API, send a short greedy smoke request and print the LIVE message with the endpoint.

A running server is left in place unless `restart` is requested. Settings are validated before a restart
stops it. Stop and restart save the previous logs. If a rank exits, the container stops the other ranks.

`scripts/prepare.sh` performs the setup without starting the API. It builds from the pinned NVIDIA image,
installs TensorFold v0.6.6 with the patches, PyAV and xgrammar, downloads the weights, checks peer access, and
compiles and tunes kernels. Image and tuning hashes decide what needs rebuilding. Re-running it skips ready work.

The startup validation below was recorded on the v0.6.5 build.

<!-- releasekit:autotune:start -->
One-time TP2 launch-table autotune.

The fresh-clone preparation, image build, kernel compilation, tuning, start, smoke and stop flow completed on the target GPU host. Later starts reuse the cached launch table.
<!-- releasekit:autotune:end -->

```bash
scripts/prepare.sh             # prepare the image, weights and kernels
scripts/prepare.sh --image     # prepare only the image
```

The revision pins are listed in [Configuration](#configuration). Downloads come directly from the publishers.
Serving uses the local weight directories with Hugging Face offline mode enabled.

## KV pool and memory

Requests share one pool for the per-token DSA cache and sparse indexer keys. KDA layers also keep
recurrent state. The per-request window and the shared pool are different limits:

| Setting | Default |
| --- | ---: |
| Concurrent requests (`PARALLEL`) | 4 |
| Maximum prompt plus reply (`CONTEXT`) | 524,288 tokens per request (512K) |
| Shared KV pool | 1,062,912 tokens shared across requests and DFlash2 extents |
| KV representation (`KV`) | FP8 |
| Free-memory reserve (`MEMORY_RESERVE_GIB`) | 6 GiB per rank |
| Spare KV cache cap (`KV_POOL_GIB`) | 3.75 GiB per rank |
| Required free memory at launch (`MIN_GPU_FREE_MIB`) | 90,000 MiB per GPU |

The memory and capacity measurements below were recorded on the v0.6.5 build.

<!-- releasekit:memory:start -->
The measured shared KV pool holds **1,062,912 tokens**. Each request allows up to **524,288 tokens** for prompt plus reply.
Long requests share the pool. One PCIe host with CUDA peer access; the two-GPU recipe uses GPUs on the same host bridge.

The maximum-context check accepted a **524,280-token prompt** plus an **8-token reply budget**. It returned **2 tokens**.

| Memory per GPU | Recorded value |
| --- | ---: |
| GPU 0, sampled peak during measurements | 92,862 MiB of 97,887 MiB |
| GPU 1, sampled peak during measurements | 91,870 MiB of 97,887 MiB |

Sampled at approximately three-second intervals across candidate preparation, startup and serving checks in an isolated window; brief transients can be missed.

Sampled peak GPU temperatures were GPU 0: 90.0 C, GPU 1: 87.0 C.
<!-- releasekit:memory:end -->

Long requests share that pool; four full context windows at once are not promised. `/health` reports
`pool_tokens` and `pool_free_tokens`. The spare-cache cap is additional cache beyond the base context,
not the size of the whole pool. The engine budget already excludes the free-memory reserve.

The engine checks the requested window against available memory. A context-budget refusal requires freeing
GPU memory or reducing `CONTEXT`; the launcher does not silently shrink it. `KV=bf16` needs more memory.
Dense and KV quantization and chunked KDA have separate numerical limits; see [Checks](#checks).

## Spill tier

Off by default; turn it on with `SPILL_GIB=64 ./start.sh restart` (any value above 0). A kept prompt state that
leaves the KV pool — evicted for another conversation, dropped at the kept-prompt cap, or left by a restart — is
written to `SPILL_DIR` on every rank (patch 0088, Mia's AI Lab's disk tier by @wojo, ported from her DGX-Spark
recipe) and read back when a later request extends the same tokens, instead of a prefill. With `PARALLEL` above 1,
past `SPILL_HIGHWATER` (0.70) of the pool the states eviction would take next are written early in the background,
so an eviction frees its rows at once; a clean stop (`./stop.sh`, `restart`) writes what is still kept within
`TF_SPILL_FLUSH_S` (60 s), and the stop waits for it.

What is and is not stored: prompts of at least `SPILL_MIN_TOKENS` (8,192) tokens, on the prompt grid, without
images (their rows are not their token ids' rows); states a longer state of the same conversation supersedes are
skipped. Files cost about 6.7 KB per token per rank (~1.4 GB per rank for a 200k-token conversation), are owned by
your user, hold your prompts' token ids, and the oldest go past `SPILL_GIB` or below the `SPILL_MIN_FREE_GIB`
floor. A restored conversation equals its fresh-prefill reply (a CRC-32 per 8 MiB and a per-build compatibility
hash; a failed read falls back to a prefill). `TF_SPILL_WRITERS` (2) and `TF_SPILL_DIRECT` (1) tune the writers;
point `SPILL_DIR` at a tmpfs to keep the tier in host RAM (restarts then resume only what the flush wrote before
the tmpfs went away). The tier's counters are in `/health` under `spill` and in Prometheus `/metrics`.

## Configuration

Settings come from the **environment**, then **`scripts/local.sh`**, then **`.env`**, then the defaults in
[`scripts/config.sh`](scripts/config.sh), in that order. `local.sh` is Bash; `.env` contains literal `KEY=value`
lines and is never executed. Both are ignored by Git. Copy `scripts/local.sh.example` for a starting point.

```bash
GPUS=0,1 ./start.sh restart    # select a pair with CUDA peer access
PORT=9000 ./start.sh           # another API port
NO_ANIM=1 ./start.sh           # static command deck, without the opener animation
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `GPUS` / `TP` | `0,1` / `2` | Two local GPU indices with CUDA peer access |
| `PORT` / `HOST` | `8022` / `0.0.0.0` | API listener |
| `SERVED_NAME` | `glm-5.3-flash` | API model id |
| `CONTAINER_NAME` / `IMAGE` | `glm53-tf-2x` / `tensorfold-glm53:1.1.0-2x` | Docker container and local image |
| `DENSE` | `q4` | 4-bit dense; `fp8` and checkpoint `bf16` are untested on two GPUs |
| `DRAFTER` | `dflash2` | `mtp` uses the checkpoint head with `PARALLEL=1`; `none` disables drafting |
| `DRAFT_POLICY` | `fnc7:0.3` | DFlash2 stopping policy |
| `PARALLEL` | `4` (`1` with MTP) | Concurrent requests; larger settings need separate memory and exactness checks |
| `CONTEXT` | `524288` | 512K maximum prompt plus reply tokens per request |
| `KV` | `fp8` | KV representation; `bf16` needs more memory |
| `TF_GLM_MULTI_WINDOW` | `32` | Batched verify rows |
| `TF_GLM_PREFILL_ROWS` / `TF_GLM_CACHE_ENTRIES` | `1024` / `4` | Prompt chunk rows and kept prompt states |
| `TF_GLM_TUNE` | `/cache/launch-table.json` | One-time TP2 autotune; `0` selects stock kernel launches |
| `TF_GLM_LANE_INPUTS` / `TF_GLM_TWOSHOT_ROWS` / `TF_GLM_SPARSE_FAST` | `0` / `0` / `0` | Leave replicated lane projections, two-shot exchange and the shape-specific sparse fast kernel off |
| `THINKING` / `REASONING_EFFORT` | `0` / `high` | Default thinking mode and effort |
| `MAX_TOKENS` | `32768` | Default completion budget when omitted by the request |
| `VISION` | `1` | Image and video input; set to `0` to free the tower and workspace on GPU 0 |
| `MEMORY_RESERVE_GIB` / `KV_POOL_GIB` | `6` / `3.75` | Free-memory reserve and spare KV cache cap per GPU |
| `SPILL_GIB` | `0` | Spill tier (patch 0088): disk a rank for kept prompt states, so an evicted conversation resumes instead of re-prefilling; `0` turns it off. See [Spill tier](#spill-tier) |
| `SPILL_DIR` / `SPILL_HIGHWATER` / `SPILL_MIN_TOKENS` | `${DATA_DIR}/spill` / `0.70` / `8192` | Both ranks share the directory; past the high-water fraction the next evictions are written early; shorter prompts are not written |
| `SPILL_MIN_FREE_GIB` / `SPILL_KEEP_BUILDS` / `TF_SPILL_FLUSH_S` | `50` / `1` / `60` | Disk free floor, other builds kept, and a clean stop's write budget (the stop waits for it) |
| `MEASURED_POWER_W` | `300` | Published power limit in watts per GPU; warns with `W_POWER` on a difference and never changes power |
| `DATA_DIR` | `${XDG_CACHE_HOME:-$HOME/.cache}/aevonix-glm53-2x` | Download and kernel cache root |
| `MODEL_DIR` / `DFLASH2_DIR` / `KERNEL_CACHE` | Under `DATA_DIR` | Separate paths can be supplied |
| `LOG_DIR` / `LOG_KEEP` | `./logs` / `10` | Archived server logs |
| `WAIT_TIMEOUT` | `2400` | Seconds to wait for API readiness |
| `HF_TOKEN_PATH` | `${HF_HOME:-$HOME/.cache/huggingface}/token` | Optional login token file; `HF_TOKEN` takes precedence |
| `DRY_RUN` | `0` | `1` prints the plan and skips hardware checks |
| `NO_ANIM` | Unset | Set to `1` to show the static command deck immediately |
| `NO_COLOR` | Unset | Set to `1` to show the banner facts as plain text |

The terminal command deck uses the Aevonix theme and shows the two-GPU defaults. Redirected output
contains no banner; `NO_COLOR` selects plain text.

The engine switches in `scripts/config.sh` use NCCL collectives over PCIe, copy-engine prompt exchanges,
two prompt lanes, packed GPU sampling, fast DFlash2 and incremental warm turns. The two-GPU recipe leaves lane input
replication and two-shot exchange off. Each switch and its effect is listed in
[docs/PATCHES.md](docs/PATCHES.md). Kernel and tuning caches are keyed to the build and GPU configuration.

TensorFold v0.6.6 supports `--name-priority ID=background` to give requests for that model id a background default when `priority` is omitted. This is merged with the recipe priority lanes. Explicit priorities and title requests are unchanged. The recipe launcher does not expose an extra server argument setting for this engine option.

Pinned inputs:

| Input | Revision |
| --- | --- |
| TensorFold v0.6.6 | `cb2ebf0540f42604e2759b2ddef497861e928248` |
| Mia's EXL3 checkpoint | `78353f1f6eb2c96fa6c62de57b44345d1f497c38` |
| Inco AI DFlash2 | `bf582e4eacc1810f76656d1811693ff6c6737d2a` |

The recipe retains thinking off and the pinned checkpoint's earlier-turn reasoning clearing.
`TF_GLM_CLEAR_THINKING=0` opts into upstream's keep-reasoning default. API keys are off unless
`TENSORFOLD_API_KEY` is explicitly configured in the environment or `.env`. Send it as a Bearer token
or `x-api-key`; `/health` remains open and `/metrics` requires the key when authentication is enabled.
The launcher authenticates its own readiness and smoke requests and does not print the key in dry runs.

### Thinking and sampling

The default is thinking off (`THINKING=0`), with `REASONING_EFFORT=high` when thinking is enabled.
A request can override the thinking setting with `"chat_template_kwargs": {"enable_thinking": true}`
or `false`. Reasoning is returned in `reasoning_content`; the answer is in `content`.
A request that omits `max_tokens` starts with `MAX_TOKENS` (32,768 by default), reduced to the space
left in the context window.

Use `temperature: 0` for greedy decoding. Sampled requests can set `temperature`, `top_p`, `top_k`,
`min_p` and `seed`. **An omitted `top_k` is served as 20.** Send `top_k`
explicitly (for example `-1` to disable it) for reproducible sampled runs.

### API notes

| Feature | Request or endpoint |
| --- | --- |
| Chat completions | `/v1/chat/completions`, with model `glm-5.3-flash` |
| Models | `/v1/models` |
| Health and metrics | `/health` and Prometheus `/metrics` |
| Tokenization | `/tokenize` and `/detokenize`, also under `/v1/` |
| Tool calls | `tools` and `tool_calls`; complete calls are streamed with schema-aware arguments |
| Structured outputs | `response_format` with `json_object` or `json_schema`, enforced with xgrammar |
| Prompt reuse | Reuses kept prompt state; shared prefixes and incremental warm turns are enabled by default |

A prompt plus an explicit completion budget that exceeds the context window is refused with
`context_length_exceeded`. Tool calls cut off by the completion limit are not sent as complete calls.
See the curl examples in [Quick start](#quick-start).

## What the patches change

The image contains every release patch, applied in global filename order to the pinned TensorFold source.
The [complete patch table](docs/PATCHES.md) lists each change, its switch and exactness boundary.

| Area | What changes |
| --- | --- |
| Weights, KV and API | Mia's patches add dense quantization, FP8 KV, drafting and prompt reuse, concurrent serving, vision, tool calling and API diagnostics. The recipe also enables structured output through xgrammar. |
| One host over PCIe | Aevonix Research's patches add CUDA IPC exchanges, separate NCCL communicators, prompt lanes and faster prompt kernels. |
| Concurrent decode | Batched drafting, verify windows and round caps serve concurrent requests. The two-GPU recipe uses a smaller window; priority lanes schedule latency-sensitive work. |
| Warm turns | Incremental tokenization and prompt reuse reduce warm-turn preparation. |
| Compatibility and checks | The series is rebased onto TensorFold v0.6.6. Launch-table candidates are checked against their reference kernel before timing. |

TP2 uses 87 patches: 55 from Mia's AI Lab and 32 from Aevonix Research. Row splitting, copy-engine
exchange, index IPC and prompt lanes support two ranks. The two-GPU recipe leaves replicated lane projections and two-shot
exchange off to limit memory use. The sparse fast kernel is guarded for a different head shape and stays off;
TP2 uses its reference path. Launch tables are tuned for TP2, with the same first-use exactness checks.

Dense and KV quantization change numerics. Chunked KDA changes prompt arithmetic. The patch table distinguishes
those changes from optimizations intended to preserve the fixed configuration's outputs.

## Checks

**Exactness.** The gates compare drafted replies with serial replies (`"draft": false`), concurrent
requests with solo requests, and long prompts with saved token hashes. These checks use a fixed checkpoint,
precision and prompt arithmetic. Passing them does not establish fidelity to BF16 dense layers or to the
original unquantized model. Keep separate references for each intentional numerical change.

The [tools/bench guide](tools/bench/README.md) describes the commands and result fields. The tools use Python,
`requests` and Pillow. They talk to the running server at `http://127.0.0.1:8022/v1` by default;
`--base` and `--model` override the endpoint and model.

Start the baseline configuration and save its references, then start the candidate with the same numerical
settings and compare:

```bash
python3 tools/bench/tf_bench.py --id baseline --ref-tag=-tp2-q4 \
  --reference-dir ./bench-references --gate-levels 2,4 \
  --gate-write-ref --no-telemetry gate gatelong
# Start the candidate with the same checkpoint and numerical settings.
python3 tools/bench/tf_bench.py --id candidate --ref-tag=-tp2-q4 \
  --reference-dir ./bench-references --gate-levels 2,4 \
  --no-telemetry gate gatelong
```

Inspect `pass_`, `fails` and `reference_status` in the result JSON. The runner exit status is not a verdict.
Creating a baseline is not an agreement check. Missing references fail; these gates require TensorFold's
token-hash extension.

**Quality.** The small checks in [Performance](#performance) cover tools, JSON, needles, images and tool bursts.
The bundled probes include tool bursts, fixed-answer agreement and greedy token-prefix comparisons:

```bash
python3 tools/bench/toolburst.py --out results/production/toolburst.json
python3 tools/bench/agree.py --out results/baseline/agree.json
python3 tools/bench/agree.py --out results/candidate/agree.json \
  --compare results/baseline/agree.json
python3 tools/bench/first_tokens.py --source-tree ./TensorFold --n 100 \
  --out results/baseline/first-tokens.json
python3 tools/bench/first_tokens.py --source-tree ./TensorFold --n 100 \
  --out results/candidate/first-tokens.json --compare results/baseline/first-tokens.json
```

Run each baseline and candidate command against its corresponding server. Use the same prompt source tree
in both token-prefix runs. The original JSON-contract and qualification fixtures are not bundled, so these
commands do not reproduce every recorded quality row. For a trusted local contract file:

```bash
python3 tools/bench/json_variants.py --contracts-file /path/to/contracts.py \
  --out results/production/json-variants.json
```

`tf_bench.py` also provides speed, prefill and warm-turn workloads. They use a separate protocol from the
README's sparkDash measurements. Vision is enabled by default; include `--gate-vision` to run the image
exactness check. See [tools/bench](tools/bench/README.md) for those commands and
[validation and limitations](docs/NOTES.md) for the recorded runs.

The patch reproduction check needs an independent release patch export and an unmodified upstream clone.
It applies both series to temporary trees and compares file bytes and modes, without GPUs:

```bash
python3 tools/check_apply.py --release-patches /path/to/reference-patches \
  --upstream /path/to/unmodified-TensorFold
```

## Repository layout

```text
.github/      Aevonix Research hero, logo, measured sparkDash cards and terminal GIFs
start.sh      prepare and start two local ranks; wait for the API and run a smoke request
stop.sh       stop the ranks and archive their logs
scripts/      configuration, preparation, patch application, launch-table tuning and banner
patches/      Mia's AI Lab and Aevonix Research patches, applied in global filename order
tools/bench/  endpoint benchmarks, exactness gates and quality probes
tools/check_apply.py  patch identity and reproduction check
docs/         complete patch table and measurement notes
CHANGELOG.md  release changes
CREDITS.md    upstream work and contributors
LICENSE       Apache License 2.0
NOTICE        third-party notices
```

## License

This repository's code and documentation are **Apache-2.0** ([LICENSE](LICENSE)). Upstream code and model
weights keep their own licenses. **DFlash2 weights are CC BY-NC-ND 4.0: non-commercial, not redistributed
here.** `DRAFTER=mtp` avoids downloading or loading them and uses the checkpoint's own MTP head at
`PARALLEL=1`. No model weights, private data or credentials are included.

## Credits

<!-- releasekit:credits:start -->
This release is developed in collaboration with [Mia's AI Lab](https://x.com/MiaAI_lab).

[TensorFold](https://github.com/ashhart/TensorFold) is by **Ash Hart**
([X @ashxhart](https://x.com/ashxhart)). The **EXL3 checkpoint and original recipe patches** are by
[Mia's AI Lab](https://huggingface.co/Mia-AiLab) ([X @MiaAI_lab](https://x.com/MiaAI_lab)), which approved
publication. **DFlash2** is by [Inco AI](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2). The single-host release and added
patches are by [Aevonix Research](https://aevonix.com). See [CREDITS.md](CREDITS.md) and [NOTICE](NOTICE).
<!-- releasekit:credits:end -->
