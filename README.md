# Algocell

**Artificial life emerging from random bytes — running entirely in your browser.**

![Algocell screenshot](image.png)

Watch self-replicating Z80 programs spontaneously emerge from a grid of random bytes. A 200x200 grid of cells, each containing 16 random bytes, is continuously executed as Z80 machine code. Within seconds, self-replicating programs appear and compete for space — digital life from pure noise.

Based on [Hartley & Colton (2024)](https://arxiv.org/abs/2406.19108). Re-implemented from the [original Python/JAX code](https://github.com/znah/zff).

## What makes this different?

The original implementation requires a **Python environment with JAX and CUDA** — meaning you need a local setup with GPU drivers, Python dependencies, and a CUDA-capable NVIDIA GPU.

Algocell runs **entirely in the browser** using WebGPU compute shaders. No installation, no dependencies, no CUDA. Just open a URL.

| | Original (znah/zff) | Algocell |
|---|---|---|
| **Platform** | Python + JAX/CUDA | Browser (WebGPU) |
| **GPU requirement** | NVIDIA + CUDA | Any GPU with WebGPU support |
| **Setup** | Install Python, JAX, CUDA drivers | Open a URL |
| **Performance** | Requires dedicated GPU server | **Up to 8B ops/sec on a MacBook Air** |
| **Visualization** | Matplotlib / separate viewer | Real-time integrated UI |
| **Interaction** | Script parameters | Live controls, zoom, pan, tooltips |

**Life emerges in seconds** — even on a laptop GPU. At 8x speed on a MacBook Air, the simulation reaches ~8 billion Z80 operations per second and self-replicating programs typically appear within moments. The entire Z80 CPU, pair selection, mutation, and rendering pipeline runs as WebGPU compute shaders dispatched every frame.

## How it works

1. **Grid**: 200x200 cells, each holding 16 bytes (640KB total)
2. **Each step**: Random adjacent cell pairs are selected. Their 32 bytes are concatenated and executed as a Z80 program for 128 steps. The modified memory is written back.
3. **Mutation**: Random bytes are flipped at a configurable rate (default 1/2⁴)
4. **Emergent behavior**: The Z80 CPU starts with all registers zeroed, so random code tends to write zeros — NOP (0x00) accumulates rapidly. Then self-replicating programs (typically `POP HL` + `EX (SP),HL` loops) emerge and outcompete the NOPs.

## Controls

| Key | Action |
|---|---|
| **Space** | Play / Pause |
| **R** | Reset simulation |
| **H** | Help / Guide |
| **1-8** | Set speed multiplier |
| **Scroll** | Zoom in/out |
| **Click + Drag** | Pan |
| **Hover** | Inspect cell (Z80 disassembly) |

## Parameters

- **Seed** — Random seed for initial grid state
- **Mutation Rate** — Probability of random byte flips (1/2ⁿ)
- **Pairs/Batch** — Cell pairs evaluated per GPU dispatch (controls throughput vs. GPU load)
- **Colormap** — Visual theme (Default, Ocean, Thermal, Rainbow)

## Development

```bash
npm install
npm run dev
```

## Building

```bash
npm run build
```

Outputs a static site (via `@sveltejs/adapter-static`) that can be deployed anywhere.

## Tech stack

- **SvelteKit** + TypeScript
- **WebGPU** compute shaders (Z80 emulation, pair selection, mutation, byte counting)
- **WebGPU** render pipeline (colormap visualization)
- **Mermaid.js** (help modal diagrams)
- **Tailwind CSS**

## Credits

- Paper: *"Self-Replicating Programs in a Z80 Virtual Machine"* — Hartley & Colton (2024) ([arXiv:2406.19108](https://arxiv.org/abs/2406.19108))
- Original implementation: [znah/zff](https://github.com/znah/zff)
