---
title: "Building the Poor Man's AI Supercomputer"
date: 2026-04-20
description: "Running 400B+ parameter LLMs locally on an EPYC + single-GPU budget box, by splitting inference into its decode and prefill phases."
tags: ["AI", "LLM", "inference", "hardware", "performance", "EPYC", "GPU"]
showDate: false
layout: "simple"
---

Running massive, 400B+ parameter AI models locally is no longer restricted to multi-million-dollar server farms. If you need top-tier intelligence and absolute data privacy, you don't need a rack of Nvidia H100s. You just need to understand the physics of how Large Language Models (LLMs) actually process data.

To build a cost-effective powerhouse, we have to stop treating the LLM as a single, monolithic program and split it into its two fundamental phases: **Decode** and **Prefill**.

## 1. The Decode Phase (The Bottleneck)

*Generating the response, one token at a time.*

The biggest misconception in AI hardware is that generating text requires massive compute power. It doesn't. Decode is strictly **memory-bandwidth bound**.

To generate a single word, the system has to load the entire massive model into the processor's cores, do a tiny bit of math, and spit out the word. If your model is larger than your GPU's VRAM, pushing those gigabytes of data back and forth across a PCIe slot for every single token will throttle your speed to absolute zero.

**The solution: do the decode in place.** This is why the foundation of the Poor Man's Supercomputer is a heavy-duty server processor, like an AMD EPYC. Instead of the 2 memory channels found in a desktop PC, an EPYC board has 12 channels. If you populate all of them with DDR5 memory, you create a massive, terabyte-sized ocean of RAM with a firehose bandwidth of ~500 GB/s.

**The software trick:** to actually hit that 500 GB/s, you have to be smart about the math. Decode is just vector-matrix multiplication. To prevent the EPYC's internal highway (the Infinity Fabric) from clogging, we use NUMA-aware execution:

- Shard (slice) the model and cache matrices, pinning them to specific physical memory nodes.
- Broadcast the tiny token vector to all CPU cores simultaneously.
- Every core multiplies the vector against its local shard in total isolation.
- Combine the tiny results at the end (an All-Reduce operation).

Result: maximum memory saturation, zero PCIe bottlenecks, and highly readable token generation speeds.

## 2. The Prefill Phase (The Sprint)

*Reading the prompt and your massive documents.*

While Decode is bandwidth-bound, Prefill is **compute bound**. When you drop a 50-page highly sensitive PDF into the prompt, the system can process all of those thousands of tokens simultaneously.

If we ask our EPYC CPU to do this, it will take minutes. CPUs simply do not have the core count for massive parallel matrix multiplication.

**The solution: push the prefill through a GPU.** Because we process all the tokens at once, the math takes long enough that the PCIe bandwidth is no longer the bottleneck. We can feed the model weights layer-by-layer through a single prosumer GPU (like an RTX 4090 or 4500). The GPU digests the massive prompt blindingly fast, generates the "memory" of that prompt (the KV Cache), and dumps that cache over the PCIe slot into the EPYC's massive RAM pool.

## 3. The Advanced Play: Managing Massive Contexts

In a long, multi-turn conversation the KV cache grows until even a single layer's worth of it no longer fits into GPU VRAM. Once that threshold is crossed, we can't compute a layer in one pass — we have to process it block by block.

**Blockwise Compute (Ring / Flash Attention):** the GPU processes the prompt in chunks. It pulls a block of the cache from host RAM over PCIe, runs its math on that block, streams the partial result back, and pulls the next. The full layer never has to live on the GPU at any single moment.

**Paged Attention:** because those blocks can now land anywhere in the EPYC's huge RAM pool, there's no need for contiguous allocations. The cache is chopped into small fixed-size pages and scattered wherever there's free space. When new tokens arrive, the GPU computes only the new tokens against the old cache and appends new pages.

## 4. Bringing It Together: Disaggregated Serving

To make this run like an enterprise appliance, we orchestrate these phases as independent workers:

- **Prefill Worker (GPU):** pulls heavy prompts from the queue, crunches them layer-by-layer, generates the KV Cache, and dumps it into storage (system RAM / fast NVMe).
- **Decode Worker (EPYC CPU):** pulls ready-to-go KV Caches from storage, runs the NUMA-aware matrix multiplication, and streams the text back to the user.

Because they are separated, the GPU never waits around for slow token generation, and the CPU generation never stutters when a new 50-page document is dropped into the queue.

## The Final Hardware Blueprint

The recipe we've been building up looks like this:

- 1× AMD EPYC processor (32 to 64 cores)
- 768 GB to 1.5 TB of DDR5 ECC RAM (filling all 12 memory channels)
- 1× prosumer GPU (e.g. RTX 3090, 4090, or RTX 4500 Ada) purely to act as the Prefill acceleration engine

## Alternatives in the Same Category

**Mac Studio.** Decent unified-memory bandwidth, enough capacity to run models well beyond what fits in a consumer GPU, and the same "not the fastest, but it runs" trade-off. The ceiling is the difference: an M4-class chip caps under 128 GB of unified memory, while an EPYC board can take you into the terabyte range. For mid-sized models, the Mac is often the simpler buy; for the 400B+ class, you need the room a server platform gives you.

**GPU mining rig.** A pile of cheap consumer GPUs (3090s, used P40s) on PCIe risers. Here you're essentially restricted to *pipeline parallelism* — the model is split across GPUs layer-by-layer, and activations are passed down the chain. Tensor parallelism is effectively off the table: it needs fast all-reduce traffic between GPUs, which means NVLink, which mining rigs don't have. Data parallelism doesn't help either, since it requires each GPU to hold the full model.

That leaves three sharp downsides. **Power draw** — a fleet of 350 W cards adds up fast. **Interconnect bandwidth** — go full "bitcoin farm" with x1 PCIe risers and you're moving activations at ~1 GB/s between layers, which will dominate token latency. **Pipeline bubbles** — with naive PP, only one GPU is actually working on a given prompt at a time, so your aggregate throughput is divided by N; the GPUs mostly take turns. You can recover it by scheduling different prompts onto each stage, phase-shifted, so every GPU is always busy on something — but that only pays off under continuous batched load, exactly the 24/7 scenario this whole approach is trying to avoid.

## What We're Actually Aiming At

We're not aiming at fast — we're aiming at **possible**. This box won't outrun a rack of H100s; nothing at this price will. What it can do is run 400B+ models *at all*, on your own hardware, with your own data. For a realistic workload — a small team running local agents or processing documents at a bursty, sparse pace — that's the whole point. A half-million-dollar server only starts to pay for itself under 24/7 batched inference, and almost nobody outside a hyperscaler actually runs that way.