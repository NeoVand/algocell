<p align="center">
  <img src="src/lib/assets/favicon.svg" width="64" height="64" alt="Algocell icon" />
</p>

<h1 align="center">Algocell</h1>

<p align="center">
  <strong>Artificial life emerging from random bytes — running entirely in your browser.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/SvelteKit-FF3E00?logo=svelte&logoColor=white" alt="SvelteKit" />
  <img src="https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/WebGPU-4285F4?logo=google-chrome&logoColor=white" alt="WebGPU" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-06B6D4?logo=tailwindcss&logoColor=white" alt="Tailwind CSS" />
  <img src="https://img.shields.io/badge/Z80-333333?logoColor=white" alt="Z80" />
</p>

<p align="center">
  <a href="https://neovand.github.io/algocell/"><strong>Try the live demo</strong></a>
</p>

![Algocell screenshot](image.png)

Watch self-replicating Z80 programs spontaneously emerge from a grid of random bytes. Each cell in the grid contains a handful of random bytes that are continuously executed as Z80 machine code. Within seconds, self-replicating programs appear and compete for space — digital life from pure noise.

Based on [Agüera y Arcas et al. (2024)](https://arxiv.org/abs/2406.19108) and the [original Python/JAX implementation](https://github.com/znah/zff) by Alexander Mordvintsev. This version re-implements the simulation using WebGPU compute shaders, so it runs directly in the browser with no installation or GPU drivers required. At max settings on a MacBook Air M3, it reaches over 50 billion Z80 operations per second.

## How it works

1. **Grid**: A configurable grid of cells (default 200×200). Choose between **square** (4×4 = 16 bytes per cell) or **hexagonal** (19 bytes per cell in a 3-4-5-4-3 hex arrangement) topologies.
2. **Each step**: Random adjacent cell pairs are selected. Their bytes are concatenated and executed as a Z80 program for a configurable number of steps. The modified memory is written back.
3. **Mutation**: Random bytes are flipped at a configurable rate (default 1/2⁴).
4. **Emergent behavior**: The Z80 CPU starts with all registers zeroed, so random code tends to write zeros — NOP (0x00) accumulates rapidly. Then self-replicating programs (typically `POP HL` + `EX (SP),HL` loops) emerge and outcompete the NOPs.

## Grid Topologies

### Square Grid

The classic mode. Each cell is a 4×4 block of bytes (16 bytes). Cells interact with their 4 cardinal neighbors (up, down, left, right).

### Hexagonal Grid

Each cell holds 19 bytes arranged in a 3-4-5-4-3 hexagonal pattern. Cells interact with 6 neighbors, creating more organic-looking emergent structures. The hex topology uses odd-r offset coordinates — odd rows are shifted right, producing a natural honeycomb layout with rectangular grid boundaries.

## Controls

| Key              | Action                                 |
| ---------------- | -------------------------------------- |
| **Space**        | Play / Pause                           |
| **R**            | Reset simulation                       |
| **H**            | Help / Guide                           |
| **1-8**          | Set speed multiplier                   |
| **Scroll**       | Zoom in/out                            |
| **Click + Drag** | Pan                                    |
| **Hover**        | Inspect cell (Z80 disassembly tooltip) |

## Parameters

- **Grid Type** — Switch between **Square** and **Hex** topologies. Changing resets the simulation.
- **Grid Size** (W × H) — Width and height of the grid in cells (default 200×200). Changing resets the simulation.
- **Seed** — Random seed for initial grid state. Same seed produces the same starting state.
- **Mutation Rate** — Probability of random byte flips (1/2ⁿ, from 1/2 to 1/2¹²). Too low and replicators can't emerge; too high and they can't survive. The sweet spot is usually 3–5.
- **Pairs/Batch** — Cell pairs evaluated per GPU dispatch (controls throughput vs. GPU load). At max, roughly a quarter of all cells are updated per step.
- **Z80 Steps** — Maximum CPU cycles per pair execution (16–1024). Lower values make the simulation faster but limit program complexity — simple replicators emerge quickly. Higher values allow more complex programs to develop but slow down the simulation. The default (128) balances speed and complexity well.
- **Colormap** — Visual theme (Rainbow, Ocean, Thermal). Each maps Z80 opcode categories to distinct colors.

## Cell Tooltips

Hover over any cell to see its bytes disassembled as Z80 instructions. In square mode, this shows a 4×4 grid; in hex mode, a hexagonal cluster of 19 bytes matching the cell's shape. Opcode bytes show the instruction mnemonic, while operand bytes (consumed by multi-byte instructions) appear slightly dimmed with their raw hex value.

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

## Credits

- Paper: _"Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction"_ — Agüera y Arcas, Alakuijala, Evans, Laurie, Mordvintsev, Niklasson, Randazzo & Versari (2024) ([arXiv:2406.19108](https://arxiv.org/abs/2406.19108))
- Original implementation: [znah/zff](https://github.com/znah/zff) by Alexander Mordvintsev
- Developed by [Neo Mohsenvand](https://github.com/NeoVand) with the help of [Claude Code](https://claude.ai)
