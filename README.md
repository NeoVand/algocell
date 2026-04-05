# Algocell

**Artificial life via Z80 machine code.** Watch self-replicating programs emerge from random bytes in a WebGPU-accelerated simulation running over 50 billion operations per second.

![Algocell screenshot](static/screenshot.png)

## What is it?

Algocell fills a grid with random bytes and executes them as Z80 machine code. Pairs of neighboring cells combine their bytes into a shared memory space, and a single Z80 CPU — starting with all registers zeroed — executes across both, reading and writing freely within the combined tape. No fitness function, no selection pressure — just raw execution.

Within seconds, self-replicating loops emerge spontaneously (typically `POP HL` + `EX (SP),HL` patterns that copy bytes forward). Once a replicator appears, it spreads exponentially, displacing the noise. You can suppress dominant patterns to see if alternative replication strategies evolve.

## Features

- **WebGPU compute shaders** — entire simulation runs on GPU, including Z80 instruction decoding
- **Square and hexagonal grids** — hex mode produces more organic emergent behavior with 6 neighbors instead of 4
- **Configurable grid size** — from tiny experiments to large-scale ecosystems
- **Real-time frequency analysis** — live chart tracking the most common byte values (opcodes) over time
- **Opcode suppression** — block dominant replicators by treating their instructions as NOPs
- **Simple/Detailed view** — toggle between averaged cell colors and individual byte-level rendering
- **Appearance controls** — multiple colormaps, brightness/contrast/saturation adjustments, grid line toggle
- **Cell inspection** — hover any cell to see its full Z80 disassembly and byte layout
- **Mobile support** — touch-optimized with tap-to-inspect and responsive layout

## Getting started

```bash
npm install
npm run dev
```

Requires a browser with [WebGPU support](https://caniuse.com/webgpu) (Chrome 113+, Edge 113+, Firefox Nightly, Safari 18.2+).

## How it works

1. A grid of cells is initialized with random bytes (the "primordial soup")
2. Each simulation step, random pairs of neighboring cells are selected
3. Both cells' bytes are placed into a shared memory space (Cell A's tape followed by Cell B's, wrapping at the boundary). A single Z80 CPU executes from the start of this memory with all registers set to zero, reading and writing anywhere in the shared space
4. The modified memory is written back to both cells, and random bit-flip mutations are applied at a configurable rate
5. The process repeats — no goal, no reward, just execution and emergence

## Tech stack

- [SvelteKit](https://svelte.dev/) + TypeScript
- [WebGPU](https://www.w3.org/TR/webgpu/) compute and render shaders (WGSL)
- [Tailwind CSS](https://tailwindcss.com/)

## License

MIT
