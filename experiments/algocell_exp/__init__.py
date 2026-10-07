"""Headless Algocell: runs the deployed simulation's exact WGSL via wgpu.

Nothing here re-implements the simulation. The shaders in ./shader are
exported verbatim from src/lib/gpu/shaders.ts (npm run export:sim) and the
ISA model from src/lib/z80-opcodes.ts; this package only supplies the host
side (buffers, dispatch, PRNG, readback) and the measurements.
"""
