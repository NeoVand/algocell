# Notes for Zilion from the free-running world

Zilion 0.3.1 already did the hard part. These are the places where Algocell had to adapt it, written as possible
upgrades. Nothing in Zilion was changed. The second design of the world (processors that land, addresses measured
from the landing byte, see `free-running.md`) needed nothing more from Zilion than the first.

## What worked as shipped

- Library mode with `init: { state: 'perLane' }` and `outputs: { state: true }` generates `zil_load_state(o)` and
  `zil_store_state(o)` against a host-named buffer (`bindingNames.out` / `stateIn`). A processor can be stepped one
  instruction and put away again with no reset. Both the 0.2 `classic` and 0.3 `zilog-nmos` Z80 cores.
- `manifest.record` gives the state layout, so the host does not hard-code register positions (25 fields in 0.2,
  28 in 0.3).
- `HALT` leaves the program counter on the HALT byte in both cores, which is what a world with no interrupts wants.

## Where the host had to work around it

1. **Shared memory.** The world contract assumes lane-private memory (`zil_m`) loaded and stored around a run. A
   world where lanes share one memory has no such thing. Algocell replaces the generated `zil_pos`, `mem_read` and
   `mem_write` by pattern and throws if the pattern is not found. Upgrade: a `memory: { kind: 'host' }` world in
   which the host declares `mem_read` / `mem_write` and Zilion emits no lane memory, no load/store helpers and no
   dummy segments.
2. **Deferred writes.** Any shared-memory world that wants a race-free update needs the same three things: a small
   per-lane write log, reads that see the lane's own pending writes, and a rule for two lanes writing one byte.
   This could be a Zilion world option rather than host code. The manifest should state the largest number of
   distinct bytes one instruction can write on each ISA (Algocell assumes at most four; the Z80 needs two; 8086
   string and stack instructions need more).
3. **A step entry point.** The host calls the core's `z80_step()` directly and skips `zil_run`, so the dial indices
   (`zil_step`, `zil_lane`, `zil_seed`) are never set. Upgrade: a documented `zil_step_once(lane, seed, step)`.
4. **HALT policy.** The host clears the Z80-private `cpu_halted` after each step. Other cores expose only
   `zil_halt_reason()`. Upgrade: `halting: { onHaltInstruction: 'stop' | 'wait' | 'pass' }`, or a `zil_clear_halt()`.
5. **Unused required fields.** A segments memory and a step budget have to be declared although neither is used.
6. **Restart in place.** When a processor lifts off, the host zeroes its state words by hand, because
   `zil_reset_lane` belongs to the run loop. Upgrade: a `zil_reset_state()` that sets the private registers to the
   world's declared reset state, so the host stays ignorant of what "reset" means on each chip.
7. **Per-lane random numbers.** The host keeps one counter per processor and hashes it with `zil_mix32`. A
   documented `zil_random(lane_counter)` would make this the same on every world.

## What a next step would need

- **Read and write sets per instruction.** To put the reversible law under single instructions the host needs the
  positions an instruction read and wrote, in order. A per-step trace record would give that, and would also give
  the causal graph of a run (which instruction read a byte that which other instruction wrote).
- **An exact reference stepper.** The comparison against `z80-emulator` stops at the first `DD`/`FD` prefix because
  the two cores split prefix chains differently, and undocumented flag bits differ. A CPU-side single-step
  reference that matches the WGSL core exactly (from the same source as zilion-vectors) would let a shared-memory
  world be checked for as long as one likes.
- **Other chips.** The 6502 and 8086 cores have not been tried in this world. The address map is the same for any
  chip with 16-bit addresses (low byte = column, high byte = row). The 6502's stack is fixed at page 1, which here
  is the row below the landing byte. The 8086's segment registers would need a decision.
- **Position-independent chips.** The landing origin makes a program mean the same thing wherever it sits, as long
  as a processor starts at its beginning. A program still cannot move its own origin. A core whose data pointers are
  relative to the program counter (BFF-style heads) would remove the origin altogether.
