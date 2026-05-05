# Qwen3.6-27B BF16 + DFlash=8 — TP=2 — repne/vllm:latest (NEW IMAGE 2026-05-05)

**Hardware:** Xeon w9-3495X (56C/112T) · 256GB DDR5-6000 · 2× RTX PRO 6000 Blackwell (96GB) · PCIe Gen 5 x16
**Image:** `repne/vllm:latest` (digest `sha256:5e7583ca4df9...`, pulled 2026-05-05 10:57 MSK)
**Manifest:** `sha256:190a64a96d6b2decb76d02ac51a4ac88e77a82c8482fbf5af338213f38e78ffb` (Hub `latest`, pushed 2026-05-05T07:08:42Z)
**Engine:** `vLLM 0.1.dev16359+ga3e24c99b.d20260505` (built 2026-05-05)
**Tuning:** governor=perf · EPP=perf · HWP boost · Uncore lock 2.5GHz · C1E off · iommu=pt
**Bench tool:** `voipmonitor/llm-inference-bench` (llm_decode_bench.py) · 30s sustained per cell · max_tokens=2048
**KV budget:** `--kv-budget 948399` (engine-reported)

---

## Hard Gates (all PASS)

| Gate | Result | Detail |
|:---:|:---:|:---|
| **T0 boot ≤ 6 min** | ✅ PASS | 234 s cold boot (KV alloc + cudagraph compile) |
| **T3 tool call** | ✅ PASS | `get_weather({city:"Paris"})` clean, finish_reason=tool_calls |
| **T4 multi-turn** | ✅ PASS | Stateful recall of name + arithmetic on remembered number |
| **T5 reasoning** | ✅ PASS | 47 × 83 = 3,901 (reasoning trace 1,251 chars) |

---

## Decode Throughput — BF16 + DFlash=8 (aggregate tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 |
|:----:|:---:|:---:|:---:|:---:|
| **0** | 100.9 | 185.7 | 329.7 | **590.1** |
| **32k** | 90.2 | 179.0 | 347.5 | 545.6 |
| **128k** | 93.5 | 159.0 | 299.6 | ∅ |
| **244k** | 67.3 | 139.7 | ∅ | ∅ |
| **256k** | 75.5\* | 139.2\* | ∅ | ∅ |

\* capacity-limited (running with reduced effective concurrency due to KV budget)
∅ skipped: cell exceeds KV cache budget (948,399 tokens)

## Per-user decode tok/s

| ctx \ conc | 1 | 2 | 4 | 8 |
|:----:|:---:|:---:|:---:|:---:|
| **0** | 102.6 | 92.8 | 84.4 | 78.7 |
| **32k** | 98.8 | 94.0 | 91.7 | 95.4 |
| **128k** | 96.8 | 87.9 | 80.9 | ∅ |
| **244k** | 79.8 | 89.7 | ∅ | ∅ |
| **256k** | 84.8\* | 82.6\* | ∅ | ∅ |

## TTFT under sustained load (mean, ms)

| ctx \ conc | 1 | 2 | 4 | 8 |
|:----:|:---:|:---:|:---:|:---:|
| **0** | 71 | 559 | 196 | 180 |
| **32k** | 325 | 484 | 754 | 1,348 |
| **128k** | 761 | 1,023 | 1,590 | ∅ |
| **244k** | 1,293 | 1,541 | ∅ | ∅ |
| **256k** | 1,232\* | 1,475\* | ∅ | ∅ |

## Request latency (mean, s) — wall time per 2,048-token request

| ctx \ conc | 1 | 2 | 4 | 8 |
|:----:|:---:|:---:|:---:|:---:|
| **0** | 20.30 | 24.12 | 24.71 | 27.92 |
| **32k** | 23.78 | 23.62 | 24.99 | 31.17 |
| **128k** | 21.94 | 26.99 | 28.33 | ∅ |
| **244k** | 29.89 | 29.99 | ∅ | ∅ |
| **256k** | 27.10\* | 30.17\* | ∅ | ∅ |

## Spec-decode acceptance rate

| ctx \ conc | 1 | 2 | 4 | 8 |
|:----:|:---:|:---:|:---:|:---:|
| **0** | 17.3% | 18.0% | 17.9% | 25.9% |
| **32k** | 52.7% | 31.5% | 27.9% | 27.2% |
| **128k** | 17.7% | 30.8% | 23.8% | ∅ |
| **244k** | 19.6% | 25.2% | ∅ | ∅ |
| **256k** | 24.5%\* | 26.9%\* | ∅ | ∅ |

## Prefill (single request)

| ctx | tokens | TTFT s | tok/s |
|:---:|:---:|:---:|:---:|
| 8k | 8,192 | 1.01 | 8,096 |
| 32k | 32,292 | 4.24 | 7,610 |
| 64k | 64,415 | 9.04 | 7,124 |
| 128k | 128,682 | 21.14 | 6,088 |

---

## NEW vs OLD image — same DFlash=8 BF16 config

Both runs: same launch args (TP=2, mem=0.85, 256k max, DFlash=8, gumbel, flash_attn, flashinfer).
Old image bench: 30s/cell, contexts {0,32k,128k,244k}, conc {1,2,4} (no c=8).

| conc | ctx | OLD (f0e7574) | NEW (5e7583c) | Δ |
|:----:|:---:|:--:|:--:|:--:|
| 1 | 0 | 91.6 | **100.9** | **+10.2%** |
| 2 | 0 | 169.9 | **185.7** | **+9.3%** |
| 4 | 0 | 352.8 | 329.7 | **-6.5%** |
| 1 | 32k | 94.2 | 90.2 | -4.2% |
| 2 | 32k | 166.0 | **179.0** | **+7.8%** |
| 4 | 32k | 337.8 | **347.5** | **+2.9%** |
| 1 | 128k | 82.6 | **93.5** | **+13.2%** |
| 2 | 128k | 160.3 | 159.0 | -0.8% |
| 4 | 128k | 272.0 | **299.6** | **+10.1%** |
| 1 | 244k | 73.8 (250k) | 67.3 | -8.8% |
| 2 | 244k | 160.3 (250k) | 139.7 | -12.9% |

**Image-over-image net (avg of 11 paired cells):** **+1.8%** aggregate tok/s
**Strongest wins:** ctx=128k +10–13% across c=1,4 · short-context c=1,2 +9–10%
**Regressions:** c=4 ctx=0 (-6.5%) · long-ctx c=2 244k/250k (-13%)

> Note: prior bench used 250k context-target where new bench uses 244k/256k. Old `250k` is closest to new `244k` for the comparison rows above.

## NEW image: c=8 (NEW dimension, no prior data)

| ctx | tok/s | per-user tok/s | TTFT ms |
|:---:|:---:|:---:|:---:|
| 0 | **590.1** | 78.7 | 180 |
| 32k | **545.6** | 95.4 | 1,348 |

c=8 at 128k+ exceeds KV budget (8 × 128k = 1,048,576 > 948,399 tokens).

---

## Critical NEW image change: KV cache budget jumped 10×

**Old image** (`f0e7574`, 2026-05-03 build): `97,248 tokens` reported (with this same launch).
**New image** (`5e7583c`, 2026-05-05 build): `948,399 tokens` reported.
**Production** (FP8+MTP=3 on new image, post-restoration): `1,844,943 tokens` — max concurrency 7.04× at full 256k context per request.

This is the dominant change in the new image and unlocks substantially more concurrency headroom at long context. The bench tool's `--kv-budget` flag now meaningfully gates infeasible cells where it previously had to guess.

---

## Capacity feasibility map (KV budget = 948,399, BF16 dflash)

| ctx | Max sustainable concurrency |
|:---:|:---:|
| 0 / 32k | 8+ (largely batch/scheduler-bound, not KV-bound) |
| 128k | ~7 (8 streams over budget by 100k) |
| 244k | ~3 (4 streams over budget by 50k) |
| 256k | ~3 (4 streams over budget by 100k) |

256k c=1 ran but bench flagged it `capacity_limited` (KV margin too tight for the prefill scout overlap).

---

## Full launch command (NEW image)

```bash
docker run -d --name qwen-vllm-bf16-dflash \
    --gpus all --ipc=host --shm-size=32g \
    --ulimit memlock=-1 --ulimit stack=67108864 --network host \
    --volume ~/.cache/huggingface:/root/.cache/huggingface \
    --volume ~/.cache/vllm:/root/.cache/vllm \
    --volume ~/.cache/flashinfer:/root/.cache/flashinfer \
    --volume ~/.triton/cache:/root/.triton/cache \
    --env OMP_NUM_THREADS=8 \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
    --env HUGGING_FACE_HUB_TOKEN=<your_hf_token> \
    repne/vllm:latest \
        -O3 --model Qwen/Qwen3.6-27B \
        --served-model-name Qwen3.6-27B qwen3.6-27b \
        --port 11435 \
        --tensor-parallel-size 2 --gpu-memory-utilization 0.85 \
        --max-model-len 262144 \
        --max-num-seqs 128 --max-num-batched-tokens 32768 \
        --max-cudagraph-capture-size 256 --language-model-only \
        --enable-auto-tool-choice \
        --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
        --enable-prefix-caching \
        --speculative-config.method dflash \
        --speculative-config.draft_sample_method gumbel \
        --speculative-config.model z-lab/Qwen3.6-27B-DFlash \
        --speculative-config.num_speculative_tokens 8 \
        --speculative-config.attention_backend flash_attn \
        --speculative-config.use_local_argmax_reduction true \
        --attention-backend flashinfer \
        --default-chat-template-kwargs.preserve_thinking true
```

---

## Bench command

```bash
.venv/bin/python llm_decode_bench.py \
    --host localhost --port 11435 \
    --model Qwen3.6-27B \
    --concurrency 1,2,4,8 \
    --contexts 0,32k,128k,244k,256k \
    --duration 30 \
    --decode-warmup-seconds 20 \
    --kv-budget 948399 \
    --display-mode plain --no-hw-monitor \
    --output bench/bf16_tp2_dflash_v3_newimage/results.json
```

Total runtime: **14:30** (870 s) for 17 measured decode cells + warmup + integrated prefill scouts.

---

## Files

- Results JSON: `/home/josh/qwen-vllm-test/bench/bf16_tp2_dflash_v3_newimage/results.json`
- Bench log: `/home/josh/qwen-vllm-test/bench/bf16_tp2_dflash_v3_newimage/bench.log`
- Launcher: `/home/josh/sisyphus/run-bf16-dflash-newimage.sh`
- Restore script: `/home/josh/sisyphus/restore-production-fp8-mtp3.sh`
- Gate evidence: `/home/josh/sisyphus/dflash-newimage-results/gates/{t3_tool_call,t4_multi_turn,t5_reasoning}.json`

---

## Production state

✅ **Restored.** `qwen-vllm-fp8-tp2` running on `repne/vllm:latest` (same NEW image, FP8+MTP=3+mem=0.85), port 11435, tool calling verified. Production now benefits from the new image's KV cache improvements (1.84M tokens budget vs prior).
