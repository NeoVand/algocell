// All WGSL shaders — factory functions for topology variants
import { getTapeLength, type GridType } from '$lib/sim/constants';
import { Z80_CORE_WGSL } from '@neovand/zilion';

// ============================================================
// SIMULATION COMPUTE SHADER
// Z80 emulator + batch management, fully parallel on GPU
// ============================================================

// Neighbor selection code block for square grid (4 cardinal neighbors)
// Neighbor selection for square grid: 4 von Neumann neighbors. At an edge the
// direction is forced INWARD (reflected), matching the zff reference — so edge
// cells always interact with a valid distinct neighbor instead of pairing with
// themselves and being skipped. (zff/wasm/main.c prepare_batch.)
const SQUARE_NEIGHBOR_SELECTION = /* wgsl */ `
    let dir = rand_bounded(4u);
    var nx = x;
    var ny = y;
    switch(dir) {
        case 0u: { if (x + 1u < w) { nx = x + 1u; } else { nx = x - 1u; } }  // right, reflect at edge
        case 1u: { if (y + 1u < h) { ny = y + 1u; } else { ny = y - 1u; } }  // down, reflect at edge
        case 2u: { if (x > 0u) { nx = x - 1u; } else { nx = x + 1u; } }       // left, reflect at edge
        case 3u: { if (y > 0u) { ny = y - 1u; } else { ny = y + 1u; } }       // up, reflect at edge
        default: {}
    }
    let j = ny * w + nx;
`;

// Neighbor selection for hex grid: cells on axial lattice (6 axial neighbors, no parity needed)
const HEX_NEIGHBOR_SELECTION = /* wgsl */ `
    let dir = rand_bounded(6u);
    var nx = x;
    var ny = y;
    let is_odd = (y & 1u) != 0u;
    switch(dir) {
        case 0u: { nx = min(x + 1u, w - 1u); }                                        // right
        case 1u: { if (x > 0u) { nx = x - 1u; } }                                     // left
        case 2u: {                                                                      // NE
            if (is_odd) { nx = min(x + 1u, w - 1u); }
            if (y > 0u) { ny = y - 1u; }
        }
        case 3u: {                                                                      // NW
            if (!is_odd && x > 0u) { nx = x - 1u; }
            if (y > 0u) { ny = y - 1u; }
        }
        case 4u: {                                                                      // SE
            if (is_odd) { nx = min(x + 1u, w - 1u); }
            ny = min(y + 1u, h - 1u);
        }
        case 5u: {                                                                      // SW
            if (!is_odd && x > 0u) { nx = x - 1u; }
            ny = min(y + 1u, h - 1u);
        }
        default: {}
    }
    let j = ny * w + nx;
`;

// ============================================================
// Host bits shared by the simulation shader and the dev test shader, so the
// differential harness exercises exactly the code the simulation runs.
// ============================================================

// Instruction suppression hook for the zilion core. Fires once per executed
// instruction after prefix resolution (see @neovand/zilion's host contract).
// IX/IY forms of a base opcode follow the base set; DDCB/FDCB follow the CB set.
const SUPPRESS_HOOK_WGSL = /* wgsl */ `
fn on_fetch_opcode(prefix: u32, op: u32) -> bool {
    var page = 0u;
    if (prefix == 0xcbu || prefix == 0xddcbu || prefix == 0xfdcbu) { page = 1u; }
    else if (prefix == 0xedu) { page = 2u; }
    let word = params.suppress[page * 2u + (op >> 7u)][(op >> 5u) & 3u];
    return ((word >> (op & 31u)) & 1u) != 0u;
}
`;

// Initial SP: the largest 16-bit value that aliases to the last byte of the
// pair, so the stack grows down from the end of the second program for every
// tape length (equals the real-Z80 reset value 0xFFFF whenever pair_length
// divides 65536, i.e. for 16-byte tapes). Mirrors spInit() in constants.ts.
const SP_INIT_WGSL = /* wgsl */ `
fn sp_init() -> u32 {
    let want = params.pair_length - 1u;
    return 0xffffu - ((0xffffu - want) % params.pair_length);
}
`;

// Pair memory layout. pair_data holds each cell word-padded (words_per_cell
// u32 per cell: 4 for 16-byte tapes, 5 for 19-byte hex tapes). The Z80 sees
// A's tape_length bytes at mem[0..tape_length) immediately followed by B's at
// mem[tape_length..pair_length) — no padding byte inside the address space.
const PAIR_MEM_WGSL = /* wgsl */ `
fn load_pair_mem(base: u32) {
    let wpc = (params.tape_length + 3u) / 4u;
    for (var b = 0u; b < params.tape_length; b++) {
        let w = b >> 2u;
        let sh = (b & 3u) * 8u;
        mem[b] = (pair_data[base + w] >> sh) & 0xffu;
        mem[params.tape_length + b] = (pair_data[base + wpc + w] >> sh) & 0xffu;
    }
}
fn store_pair_mem(base: u32) {
    let wpc = (params.tape_length + 3u) / 4u;
    for (var w = 0u; w < wpc; w++) {
        var wa = 0u;
        var wb = 0u;
        for (var k = 0u; k < 4u; k++) {
            let b = w * 4u + k;
            if (b < params.tape_length) {
                wa |= mem[b] << (k * 8u);
                wb |= mem[params.tape_length + b] << (k * 8u);
            }
        }
        pair_data[base + w] = wa;
        pair_data[base + wpc + w] = wb;
    }
}
`;

export function createSimShader(gridType: GridType, tapeLength: number = getTapeLength(gridType)): string {
	const neighborBlock = gridType === 'hex' ? HEX_NEIGHBOR_SELECTION : SQUARE_NEIGHBOR_SELECTION;
	// Private pair memory: both word-padded tapes (the Z80 only addresses pair_length of it).
	const memWords = Math.ceil(tapeLength / 4) * 2 * 4;

	return /* wgsl */ `

// === Bindings ===
struct Params {
    soup_width: u32,
    soup_height: u32,
    tape_length: u32,
    pair_length: u32,
    pair_count: u32,
    mutation_count: u32,
    z80_steps: u32,
    batch_seed: u32,
    // Instruction suppression: three 256-bit sets, one per opcode page
    // (0 = base, 1 = CB, 2 = ED), two vec4<u32> each. Bit (op & 31) of
    // u32 word (op >> 5). Packed on the host by suppressionMasks().
    suppress: array<vec4<u32>, 6>,
}

@group(0) @binding(0) var<storage, read_write> soup: array<u32>;
@group(0) @binding(1) var<storage, read_write> pairs: array<u32>;       // [cell_i, cell_j] per pair
@group(0) @binding(2) var<storage, read_write> pair_data: array<u32>;   // 8 u32s (32 bytes) per pair
@group(0) @binding(3) var<storage, read_write> write_counts: array<u32>; // 2 per pair (tape A, tape B)
@group(0) @binding(4) var<storage, read_write> rng_states: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> pair_active: array<u32>;
@group(0) @binding(7) var<storage, read_write> byte_counts: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> collision_mask: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> cell_hashes: array<u32>;

// === PRNG (PCG-based) ===
fn pcg(state: ptr<private, u32>) -> u32 {
    let s = *state;
    *state = s * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (word >> 22u) ^ word;
}

var<private> rng: u32;

fn rand() -> u32 {
    return pcg(&rng);
}

fn rand_bounded(bound: u32) -> u32 {
    return rand() % bound;
}

// === Z80 memory + suppression (host bits for the shared zilion Z80 core) ===
// The Z80 instruction logic lives in the zilion package (single source of
// truth). Algocell provides its own memory model (wrapping mod pair_length,
// with A/B write counting) and an opcode-suppression hook.
var<private> cpu_writes_a: u32;
var<private> cpu_writes_b: u32;
var<private> mem: array<u32, ${memWords}>;
fn mem_read(addr: u32) -> u32 {
    return mem[addr % params.pair_length];
}
fn mem_write(addr: u32, val: u32) {
    let a = addr % params.pair_length;
    mem[a] = val & 0xffu;
    if (a < params.tape_length) { cpu_writes_a += 1u; } else { cpu_writes_b += 1u; }
}
${SUPPRESS_HOOK_WGSL}
${SP_INIT_WGSL}
${PAIR_MEM_WGSL}

${Z80_CORE_WGSL}

// === Soup byte access helpers ===
fn read_soup_byte(cell: u32, byte_idx: u32) -> u32 {
    let wpc = (params.tape_length + 3u) / 4u;
    let word_idx = cell * wpc + (byte_idx >> 2u);
    let shift = (byte_idx & 3u) * 8u;
    return (soup[word_idx] >> shift) & 0xffu;
}

fn write_soup_byte(cell: u32, byte_idx: u32, val: u32) {
    let word_idx = cell * ((params.tape_length + 3u) / 4u) + (byte_idx >> 2u);
    let shift = (byte_idx & 3u) * 8u;
    let mask = ~(0xffu << shift);
    soup[word_idx] = (soup[word_idx] & mask) | ((val & 0xffu) << shift);
}

// ============================================================
// COMPUTE ENTRY POINTS
// ============================================================

// --- Clear collision mask ---
@compute @workgroup_size(256)
fn clear_collision(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= params.soup_width * params.soup_height) { return; }
    atomicStore(&collision_mask[idx], 0u);
}

// --- Prepare batch: generate pairs, claim cells, copy data ---
@compute @workgroup_size(64)
fn prepare_batch(@builtin(global_invocation_id) id: vec3u) {
    let pair_id = id.x;
    if (pair_id >= params.pair_count) { return; }

    // Seed RNG from batch_seed + pair_id
    rng = params.batch_seed * 1099087573u + pair_id * 2654435761u + 1u;
    rand(); // warm up

    let w = params.soup_width;
    let h = params.soup_height;
    let x = rand_bounded(w);
    let y = rand_bounded(h);
    let i = y * w + x;

    // --- Topology-specific neighbor selection ---
${neighborBlock}

    // Collision detection with atomics
    var is_active = 0u;
    if (i != j) {
        let claim_i = atomicCompareExchangeWeak(&collision_mask[i], 0u, 1u);
        if (claim_i.exchanged) {
            let claim_j = atomicCompareExchangeWeak(&collision_mask[j], 0u, 1u);
            if (claim_j.exchanged) {
                is_active = 1u;
            } else {
                atomicStore(&collision_mask[i], 0u); // release i
            }
        }
    }

    pair_active[pair_id] = is_active;
    pairs[pair_id * 2u] = i;
    pairs[pair_id * 2u + 1u] = j;

    if (is_active != 0u) {
        // Copy tape data into pair_data
        let words_per_cell = (params.tape_length + 3u) / 4u;
        let words_per_pair = words_per_cell * 2u;
        let base = pair_id * words_per_pair;
        for (var w_idx = 0u; w_idx < words_per_cell; w_idx++) {
            pair_data[base + w_idx] = soup[i * words_per_cell + w_idx];
        }
        for (var w_idx = 0u; w_idx < words_per_cell; w_idx++) {
            pair_data[base + words_per_cell + w_idx] = soup[j * words_per_cell + w_idx];
        }
    }

    write_counts[pair_id * 2u] = 0u;
    write_counts[pair_id * 2u + 1u] = 0u;
}

// --- Z80 Execute: run Z80 on each pair ---
@compute @workgroup_size(64)
fn z80_execute_batch(@builtin(global_invocation_id) id: vec3u) {
    let pair_id = id.x;
    if (pair_id >= params.pair_count) { return; }
    if (pair_active[pair_id] == 0u) { return; }

    // Load pair memory: A at mem[0..tape_length), B at mem[tape_length..pair_length).
    let base = pair_id * ((params.tape_length + 3u) / 4u) * 2u;
    load_pair_mem(base);

    // Reset CPU state
    cpu_a = 0u; cpu_f = 0u; cpu_b = 0u; cpu_c = 0u;
    cpu_d = 0u; cpu_e = 0u; cpu_h = 0u; cpu_l = 0u;
    cpu_sp = sp_init(); cpu_pc = 0u; // stack grows down from the end of B (0xFFFF for 16-byte tapes; see sp_init)
    cpu_a2 = 0u; cpu_f2 = 0u; cpu_b2 = 0u; cpu_c2 = 0u;
    cpu_d2 = 0u; cpu_e2 = 0u; cpu_h2 = 0u; cpu_l2 = 0u;
    cpu_ix = 0u; cpu_iy = 0u;
    cpu_halted = 0u;
    cpu_iff1 = 0u; cpu_iff2 = 0u;
    cpu_writes_a = 0u;
    cpu_writes_b = 0u;

    // Run Z80 steps
    for (var step = 0u; step < params.z80_steps; step++) {
        if (cpu_halted != 0u) { break; }
        z80_step();
    }

    // Save pair memory back (padding bytes are written as zero)
    store_pair_mem(base);

    write_counts[pair_id * 2u] = cpu_writes_a;
    write_counts[pair_id * 2u + 1u] = cpu_writes_b;
}

// --- Absorb: write results back to soup ---
@compute @workgroup_size(64)
fn absorb_results(@builtin(global_invocation_id) id: vec3u) {
    let pair_id = id.x;
    if (pair_id >= params.pair_count) { return; }
    if (pair_active[pair_id] == 0u) { return; }

    let i = pairs[pair_id * 2u];
    let j = pairs[pair_id * 2u + 1u];
    let words_per_cell = (params.tape_length + 3u) / 4u;
    let words_per_pair = words_per_cell * 2u;
    let base = pair_id * words_per_pair;

    for (var w_idx = 0u; w_idx < words_per_cell; w_idx++) {
        soup[i * words_per_cell + w_idx] = pair_data[base + w_idx];
    }
    for (var w_idx = 0u; w_idx < words_per_cell; w_idx++) {
        soup[j * words_per_cell + w_idx] = pair_data[base + words_per_cell + w_idx];
    }
}

// --- Mutate: apply random mutations to soup ---
@compute @workgroup_size(64)
fn mutate_soup(@builtin(global_invocation_id) id: vec3u) {
    let mut_id = id.x;
    if (mut_id >= params.mutation_count) { return; }

    rng = params.batch_seed * 3266489917u + mut_id * 668265263u + 7u;
    rand(); // warm up

    let total_bytes = params.soup_width * params.soup_height * params.tape_length;
    let pos = rand_bounded(total_bytes);
    let val = rand() & 0xffu;

    // Map flat byte position to word-aligned soup buffer
    let cell = pos / params.tape_length;
    let byte_in_cell = pos % params.tape_length;
    let wpc = (params.tape_length + 3u) / 4u;
    let word_idx = cell * wpc + (byte_in_cell >> 2u);
    let shift = (byte_in_cell & 3u) * 8u;
    let mask = ~(0xffu << shift);
    soup[word_idx] = (soup[word_idx] & mask) | (val << shift);
}

// --- Count bytes for statistics ---
@compute @workgroup_size(256)
fn count_bytes(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    let total_words = params.soup_width * params.soup_height * ((params.tape_length + 3u) / 4u);
    if (idx >= total_words) { return; }

    // Skip the padding bytes of a cell's last word (hex tapes are 19 bytes in 5 words).
    let wpc = (params.tape_length + 3u) / 4u;
    let valid = min(4u, params.tape_length - (idx % wpc) * 4u);
    let word = soup[idx];
    for (var b = 0u; b < valid; b++) {
        atomicAdd(&byte_counts[(word >> (b * 8u)) & 0xffu], 1u);
    }
}

// --- Clear byte counts ---
@compute @workgroup_size(256)
fn clear_byte_counts(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < 256u) {
        atomicStore(&byte_counts[id.x], 0u);
    }
}

// --- Hash cells (FNV-1a per cell for species tracking) ---
@compute @workgroup_size(256)
fn hash_cells(@builtin(global_invocation_id) id: vec3u) {
    let cell_idx = id.x;
    let cell_count = params.soup_width * params.soup_height;
    if (cell_idx >= cell_count) { return; }

    let words_per_cell = (params.tape_length + 3u) / 4u;
    let base = cell_idx * words_per_cell;

    // FNV-1a hash
    var hash = 2166136261u;
    for (var w = 0u; w < words_per_cell; w++) {
        let word = soup[base + w];
        let bytes_remaining = params.tape_length - w * 4u;
        let byte_count = min(4u, bytes_remaining);
        for (var b = 0u; b < byte_count; b++) {
            let byte_val = (word >> (b * 8u)) & 0xffu;
            hash = hash ^ byte_val;
            hash = hash * 16777619u;
        }
    }
    cell_hashes[cell_idx] = hash;
}
`;
}

// ============================================================
// RENDER SHADER
// Full-screen quad rendering the soup as colored tiles/hexagons
// ============================================================

const SQUARE_RENDER_FRAGMENT = /* wgsl */ `
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let sw = f32(rparams.soup_width);
    let sh = f32(rparams.soup_height);
    let ts = rparams.tile_size;

    // Zoom/pan: convert UV to grid coordinates
    let aspect = rparams.canvas_width / rparams.canvas_height;
    let cells_x = rparams.zoom;
    let cells_y = rparams.zoom / aspect;

    // Grid position in cell units (with sub-cell precision)
    let grid_x = in.uv.x * cells_x + rparams.offset_x;
    let grid_y = (1.0 - in.uv.y) * cells_y + rparams.offset_y;

    // Clamp to grid bounds
    if (grid_x < 0.0 || grid_x >= sw || grid_y < 0.0 || grid_y >= sh) {
        return vec4f(0.04, 0.04, 0.06, 1.0);
    }

    let cell_x = u32(grid_x);
    let cell_y = u32(grid_y);
    let cell_idx = cell_y * u32(sw) + cell_x;

    // Sub-cell position (0..1 within the cell)
    let frac_x = fract(grid_x);
    let frac_y = fract(grid_y);

    // Determine pixel size relative to cell for grid lines
    let pixel_cell_size = cells_x / rparams.canvas_width; // cells per pixel

    if (rparams.show_average != 0u || pixel_cell_size > 0.25) {
        // Average mode OR zoomed out too far to see individual bytes
        var acc = vec3f(0.0);
        for (var i = 0u; i < rparams.tape_length; i++) {
            let byte_val = read_soup_byte(cell_idx, i);
            let col = unpack_color(colormap[byte_val]);
            acc += col.rgb * col.rgb;
        }
        var avg = sqrt(acc / f32(rparams.tape_length));

        // Highlight hovered cell
        if (rparams.hover_cell >= 0 && cell_idx == u32(rparams.hover_cell)) {
            avg = avg * 1.4 + vec3f(0.05);
        }

        // Cell boundary grid lines (fade out when zoomed out)
        if (rparams.show_grid != 0u) {
            let avg_line_fade = smoothstep(0.15, 0.04, pixel_cell_size);
            if (avg_line_fade > 0.01) {
                let cell_dist = min(min(frac_x, 1.0 - frac_x), min(frac_y, 1.0 - frac_y));
                let cell_lw = pixel_cell_size * 2.5;
                let cell_t = smoothstep(0.0, cell_lw, cell_dist);
                let lined = mix(vec3f(0.18), avg, cell_t);
                avg = mix(avg, lined, avg_line_fade);
            }
        }

        return vec4f(apply_bcs(avg), 1.0);
    }

    // Tile mode: show individual bytes (zoomed in enough)
    let local_x = u32(frac_x * f32(ts));
    let local_y = u32(frac_y * f32(ts));
    let byte_idx = local_y * ts + local_x;
    let byte_val = read_soup_byte(cell_idx, byte_idx);
    var color = unpack_color(colormap[byte_val]).rgb;

    // Highlight hovered cell
    if (rparams.hover_cell >= 0 && cell_idx == u32(rparams.hover_cell)) {
        color = color * 1.4 + vec3f(0.08);
    }

    // Grid lines (only when enabled)
    if (rparams.show_grid != 0u) {
        let line_fade = smoothstep(0.25, 0.05, pixel_cell_size);

        // Byte grid lines (thin, within cell)
        if (pixel_cell_size < 0.05) {
            let byte_fade = smoothstep(0.05, 0.02, pixel_cell_size);
            let bfx = fract(frac_x * f32(ts));
            let bfy = fract(frac_y * f32(ts));
            let byte_dist = min(min(bfx, 1.0 - bfx), min(bfy, 1.0 - bfy));
            let byte_lw = pixel_cell_size * f32(ts) * 1.0;
            let byte_t = smoothstep(0.0, byte_lw, byte_dist);
            let byte_lined = mix(vec3f(0.06), color, byte_t);
            color = mix(color, byte_lined, byte_fade);
        }

        // Cell boundary lines (thicker, between cells)
        if (line_fade > 0.01) {
            let cell_dist = min(min(frac_x, 1.0 - frac_x), min(frac_y, 1.0 - frac_y));
            let cell_lw = pixel_cell_size * 2.5;
            let cell_t = smoothstep(0.0, cell_lw, cell_dist);
            let cell_lined = mix(vec3f(0.18), color, cell_t);
            color = mix(color, cell_lined, line_fade);
        }
    }

    return vec4f(apply_bcs(color), 1.0);
}
`;

const HEX_RENDER_FRAGMENT = /* wgsl */ `
const HEX_H: f32 = 0.8660254;  // sqrt(3)/2
const CELL_SPACING: i32 = 5;   // hex-byte distance between cell centers

// Map from axial offset (dq+2, dr+2) in 5x5 grid to byte index, -1 = gap
// 19 valid positions within hex distance 2, 6 invalid corners
const BYTE_FROM_OFFSET: array<i32, 25> = array<i32, 25>(
    -1, -1, 14, 13, 12,
    -1, 15,  5,  4, 11,
    16,  6,  0,  3, 10,
    17,  1,  2,  9, -1,
    18,  7,  8, -1, -1
);

// Center of a hex byte in the global hex-byte grid (offset coords)
fn byte_hex_center(hx: i32, hy: i32) -> vec2f {
    let odd = (hy & 1) == 1;
    return vec2f(f32(hx) + 0.5 + select(0.0, 0.5, odd), (f32(hy) + 0.5) * HEX_H);
}

// Center of a CELL in pixel space for simple-mode rendering
// Clean hex grid: cells spaced CELL_SPACING apart with odd-row offset
fn cell_px(cx: i32, cy: i32) -> vec2f {
    let odd = (cy & 1) == 1;
    let sx = f32(CELL_SPACING);
    let sy = f32(CELL_SPACING) * HEX_H;
    return vec2f(
        (f32(cx) + 0.5) * sx + select(0.0, sx * 0.5, odd),
        (f32(cy) + 0.5) * sy
    );
}

// Find nearest cell directly from pixel coordinates (bypasses byte-level hex grid)
fn find_nearest_cell(px: f32, py: f32) -> vec2i {
    let sy = f32(CELL_SPACING) * HEX_H;
    let sx = f32(CELL_SPACING);
    let cy_f = py / sy - 0.5;
    let cy0 = i32(floor(cy_f));

    var best = vec2i(0, 0);
    var best_d = 1e10;
    for (var dcy = 0; dcy <= 1; dcy++) {
        let cy = cy0 + dcy;
        let odd = (cy & 1) == 1;
        let cx_f = (px - select(0.0, sx * 0.5, odd)) / sx - 0.5;
        let cx0 = i32(floor(cx_f));
        for (var dcx = 0; dcx <= 1; dcx++) {
            let cx = cx0 + dcx;
            let c = cell_px(cx, cy);
            let d = (px - c.x) * (px - c.x) + (py - c.y) * (py - c.y);
            if (d < best_d) {
                best_d = d;
                best = vec2i(cx, cy);
            }
        }
    }
    return best;
}

// Find nearest hex-byte in the global hex grid
fn nearest_byte_hex(gx: f32, gy: f32) -> vec2i {
    let row_f = gy / HEX_H - 0.5;
    let row = i32(round(row_f));
    let odd = (row & 1) == 1;
    let col_f = gx - 0.5 - select(0.0, 0.5, odd);
    let col = i32(round(col_f));

    var best = vec2i(col, row);
    var best_d = 1e10;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let cc = col + dx;
            let cr = row + dy;
            let c = byte_hex_center(cc, cr);
            let d = (gx - c.x) * (gx - c.x) + (gy - c.y) * (gy - c.y);
            if (d < best_d) {
                best_d = d;
                best = vec2i(cc, cr);
            }
        }
    }
    return best;
}

// Find which cell a hex byte at offset (col, row) belongs to
// Cells on offset hex grid: cell (cx, cy) center at offset (cx*5 + (cy&1)*2, cy*5)
// Returns vec3i(cell_x, cell_y, byte_index) or byte_index=-1 if gap
const ODD_SHIFT: i32 = 2;

fn find_cell(col: i32, row: i32) -> vec3i {
    let cy0 = i32(floor(f32(row) / f32(CELL_SPACING)));

    var best_cx = 0;
    var best_cy = 0;
    var best_dist = 100;

    for (var dcy = 0; dcy <= 1; dcy++) {
        let cy = cy0 + dcy;
        let center_row = cy * CELL_SPACING;
        let shift = (cy & 1) * ODD_SHIFT;
        let adj_col = col - shift;
        let cx0 = i32(floor(f32(adj_col) / f32(CELL_SPACING)));

        for (var dcx = 0; dcx <= 1; dcx++) {
            let cx = cx0 + dcx;
            let center_col = cx * CELL_SPACING + shift;

            // Convert both to axial for hex distance
            let q_byte = col - (row - (row & 1)) / 2;
            let q_cell = center_col - (center_row - (center_row & 1)) / 2;
            let dq = q_byte - q_cell;
            let dr = row - center_row;
            let dist = max(max(abs(dq), abs(dr)), abs(dq + dr));

            if (dist < best_dist) {
                best_dist = dist;
                best_cx = cx;
                best_cy = cy;
            }
        }
    }

    if (best_dist > 2) {
        return vec3i(best_cx, best_cy, -1); // gap between cells
    }

    // Recompute axial diff for byte lookup
    let center_col = best_cx * CELL_SPACING + (best_cy & 1) * ODD_SHIFT;
    let center_row = best_cy * CELL_SPACING;
    let q_byte = col - (row - (row & 1)) / 2;
    let q_cell = center_col - (center_row - (center_row & 1)) / 2;
    let dq = q_byte - q_cell;
    let dr = row - center_row;
    let lookup_idx = (dq + 2) * 5 + (dr + 2);
    let byte_idx = BYTE_FROM_OFFSET[lookup_idx];

    return vec3i(best_cx, best_cy, byte_idx);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let sw = i32(rparams.soup_width);
    let sh = i32(rparams.soup_height);

    let aspect = rparams.canvas_width / rparams.canvas_height;
    let zoom = rparams.zoom;

    // Map UV to hex-byte pixel space
    let pixel_x = in.uv.x * zoom + rparams.offset_x;
    let pixel_y = (1.0 - in.uv.y) * zoom / aspect + rparams.offset_y;

    // ── Simple mode: completely bypass byte-level hex grid ──
    if (rparams.show_average != 0u) {
        let pt = vec2f(pixel_x, pixel_y);

        // Find nearest cell directly from pixel coords (no byte grid)
        let cell = find_nearest_cell(pixel_x, pixel_y);
        let cx = cell.x;
        let cy = cell.y;

        if (cx < 0 || cx >= sw || cy < 0 || cy >= sh) {
            return vec4f(0.04, 0.04, 0.06, 1.0);
        }

        let cell_idx = u32(cy) * u32(sw) + u32(cx);

        // Average color of all bytes in cell
        var acc = vec3f(0.0);
        for (var i = 0u; i < rparams.tape_length; i++) {
            let bv = read_soup_byte(cell_idx, i);
            let col = unpack_color(colormap[bv]);
            acc += col.rgb * col.rgb;
        }
        var avg = sqrt(acc / f32(rparams.tape_length));

        if (rparams.hover_cell >= 0 && cell_idx == u32(rparams.hover_cell)) {
            avg = avg * 1.4 + vec3f(0.05);
        }

        // Voronoi boundary: check all 6 cell neighbors
        if (rparams.show_grid != 0u) {
            let cc = cell_px(cx, cy);
            let dist_self = distance(pt, cc);
            let is_odd = (cy & 1) == 1;
            var min_nd = 1e10;
            min_nd = min(min_nd, distance(pt, cell_px(cx - 1, cy)));
            min_nd = min(min_nd, distance(pt, cell_px(cx + 1, cy)));
            if (is_odd) {
                min_nd = min(min_nd, distance(pt, cell_px(cx,     cy - 1)));
                min_nd = min(min_nd, distance(pt, cell_px(cx + 1, cy - 1)));
                min_nd = min(min_nd, distance(pt, cell_px(cx,     cy + 1)));
                min_nd = min(min_nd, distance(pt, cell_px(cx + 1, cy + 1)));
            } else {
                min_nd = min(min_nd, distance(pt, cell_px(cx - 1, cy - 1)));
                min_nd = min(min_nd, distance(pt, cell_px(cx,     cy - 1)));
                min_nd = min(min_nd, distance(pt, cell_px(cx - 1, cy + 1)));
                min_nd = min(min_nd, distance(pt, cell_px(cx,     cy + 1)));
            }

            let boundary = (min_nd - dist_self) * 0.5;
            let bpp = zoom / rparams.canvas_width;
            let line_w = bpp * 3.0 + 0.04;
            let edge_t = 1.0 - smoothstep(0.0, line_w, boundary);
            avg = mix(avg, vec3f(0.06), edge_t * 0.8);
        }

        return vec4f(apply_bcs(avg), 1.0);
    }

    // ── Detailed mode: use byte-level hex grid ──

    // Find nearest hex byte (offset coords)
    let hex = nearest_byte_hex(pixel_x, pixel_y);
    let hx = hex.x;
    let hy = hex.y;

    // Find which cell this byte belongs to (using offset coords)
    let cell_info = find_cell(hx, hy);
    let cell_a = cell_info.x;
    let cell_b = cell_info.y;
    let byte_idx_i = cell_info.z;

    // Out of bounds
    if (cell_a < 0 || cell_a >= sw || cell_b < 0 || cell_b >= sh) {
        return vec4f(0.04, 0.04, 0.06, 1.0);
    }

    let cell_idx = u32(cell_b) * u32(sw) + u32(cell_a);
    let in_gap = byte_idx_i < 0;

    // LOD: bytes per screen pixel
    let bytes_per_pixel = zoom / rparams.canvas_width;

    // Zoomed out: average all 19 bytes per cell
    if (bytes_per_pixel > 1.0) {
        var acc = vec3f(0.0);
        for (var i = 0u; i < rparams.tape_length; i++) {
            let bv = read_soup_byte(cell_idx, i);
            let col = unpack_color(colormap[bv]);
            acc += col.rgb * col.rgb;
        }
        var avg = sqrt(acc / f32(rparams.tape_length));
        if (rparams.hover_cell >= 0 && cell_idx == u32(rparams.hover_cell)) {
            avg = avg * 1.4 + vec3f(0.05);
        }
        if (in_gap) {
            let gap_blend = smoothstep(0.2, 1.0, bytes_per_pixel);
            avg = mix(avg * 0.3, avg, gap_blend);
        }
        return vec4f(apply_bcs(avg), 1.0);
    }

    // Gap between cells: blend from dark to cell avg as zoom decreases
    if (in_gap) {
        let gap_blend = smoothstep(0.1, 0.6, bytes_per_pixel);
        if (gap_blend < 0.01) {
            return vec4f(0.04, 0.04, 0.06, 1.0);
        }
        var acc = vec3f(0.0);
        for (var i = 0u; i < rparams.tape_length; i++) {
            let bv = read_soup_byte(cell_idx, i);
            let col = unpack_color(colormap[bv]);
            acc += col.rgb * col.rgb;
        }
        var avg = sqrt(acc / f32(rparams.tape_length));
        let gap_color = mix(vec3f(0.04, 0.04, 0.06), avg * 0.7, gap_blend);
        return vec4f(apply_bcs(gap_color), 1.0);
    }

    let byte_idx = u32(byte_idx_i);

    // Read byte value and get color
    let byte_val = read_soup_byte(cell_idx, byte_idx);
    var color = unpack_color(colormap[byte_val]).rgb;

    // Highlight hovered cell
    if (rparams.hover_cell >= 0 && cell_idx == u32(rparams.hover_cell)) {
        color = color * 1.4 + vec3f(0.08);
    }

    // Grid lines (fade out as zoom increases)
    if (rparams.show_grid != 0u) {
        let line_fade = smoothstep(0.3, 0.08, bytes_per_pixel);
        if (line_fade > 0.01) {
            let center = byte_hex_center(hx, hy);
            let dist_to_center = distance(vec2f(pixel_x, pixel_y), center);

            var min_neighbor_dist = 1e10;
            let h_odd = (hy & 1) == 1;
            let offsets_even = array<vec2i, 6>(
                vec2i(-1, -1), vec2i(0, -1),
                vec2i(-1, 0), vec2i(1, 0),
                vec2i(-1, 1), vec2i(0, 1)
            );
            let offsets_odd = array<vec2i, 6>(
                vec2i(0, -1), vec2i(1, -1),
                vec2i(-1, 0), vec2i(1, 0),
                vec2i(0, 1), vec2i(1, 1)
            );
            for (var n = 0; n < 6; n++) {
                var off: vec2i;
                if (h_odd) { off = offsets_odd[n]; } else { off = offsets_even[n]; }
                let nc = byte_hex_center(hx + off.x, hy + off.y);
                let nd = distance(vec2f(pixel_x, pixel_y), nc);
                min_neighbor_dist = min(min_neighbor_dist, nd);
            }

            let boundary_raw = (min_neighbor_dist - dist_to_center) * 0.5;

            // Check if boundary is between cells or gap
            var is_cell_boundary = false;
            for (var n = 0; n < 6; n++) {
                var off: vec2i;
                if (h_odd) { off = offsets_odd[n]; } else { off = offsets_even[n]; }
                let nhx = hx + off.x;
                let nhy = hy + off.y;
                let n_cell = find_cell(nhx, nhy);
                if (n_cell.z < 0 || n_cell.x != cell_a || n_cell.y != cell_b) {
                    let nc = byte_hex_center(nhx, nhy);
                    let nd = distance(vec2f(pixel_x, pixel_y), nc);
                    if (abs(nd - min_neighbor_dist) < 0.01) {
                        is_cell_boundary = true;
                        break;
                    }
                }
            }

            let line_w = bytes_per_pixel * select(1.0, 2.5, is_cell_boundary);
            let line_color = select(vec3f(0.06), vec3f(0.18), is_cell_boundary);
            if (boundary_raw < line_w) {
                let t = smoothstep(0.0, line_w, boundary_raw);
                let lined = mix(line_color, color, t);
                color = mix(color, lined, line_fade);
            }
        }
    }

    return vec4f(apply_bcs(color), 1.0);
}
`;

export function createRenderShader(gridType: GridType): string {
	const fragmentBlock = gridType === 'hex' ? HEX_RENDER_FRAGMENT : SQUARE_RENDER_FRAGMENT;

	return /* wgsl */ `

struct RenderParams {
    soup_width: u32,
    soup_height: u32,
    tile_size: u32,
    canvas_width: f32,
    canvas_height: f32,
    show_average: u32,
    hover_cell: i32,
    zoom: f32,         // cells visible across width
    offset_x: f32,     // pan offset in cell units
    offset_y: f32,
    tape_length: u32,
    brightness: f32,   // -1..1, default 0
    contrast: f32,     // 0..2, default 1
    saturation: f32,   // 0..2, default 1
    show_grid: u32,    // 1 = show grid lines, 0 = hide
    _pad4: u32,
}

@group(0) @binding(0) var<storage, read> soup: array<u32>;
@group(0) @binding(1) var<storage, read> colormap: array<u32>;     // 256 RGBA packed
@group(0) @binding(2) var<uniform> rparams: RenderParams;
@group(0) @binding(3) var<storage, read> trace_image: array<u32>;  // trace overlay RGBA

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let pos = array<vec2f, 6>(
        vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1),
        vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1)
    );
    var out: VertexOutput;
    out.position = vec4f(pos[vi], 0, 1);
    out.uv = pos[vi] * 0.5 + 0.5;
    return out;
}

fn unpack_color(packed: u32) -> vec4f {
    return vec4f(
        f32(packed & 0xffu) / 255.0,
        f32((packed >> 8u) & 0xffu) / 255.0,
        f32((packed >> 16u) & 0xffu) / 255.0,
        f32((packed >> 24u) & 0xffu) / 255.0
    );
}

fn read_soup_byte(cell: u32, byte_idx: u32) -> u32 {
    let wpc = (rparams.tape_length + 3u) / 4u;
    let word_idx = cell * wpc + (byte_idx >> 2u);
    let shift = (byte_idx & 3u) * 8u;
    return (soup[word_idx] >> shift) & 0xffu;
}

// Apply brightness, contrast, saturation adjustments
fn apply_bcs(c: vec3f) -> vec3f {
    // Brightness: shift
    var col = c + rparams.brightness;
    // Contrast: scale around 0.5
    col = (col - 0.5) * rparams.contrast + 0.5;
    // Saturation: lerp toward luminance
    let lum = dot(col, vec3f(0.299, 0.587, 0.114));
    col = mix(vec3f(lum), col, rparams.saturation);
    return clamp(col, vec3f(0.0), vec3f(1.0));
}

${fragmentBlock}
`;
}

// ============================================================
// Z80 DIFFERENTIAL TEST SHADER (dev-only)
// Reuses the EXACT Z80 core sliced from the sim shader (single source of
// truth), wrapped in a standalone entry point that runs one random program
// per invocation. Never imported by the app runtime.
// ============================================================
export function createZ80TestShader(tapeLength: number = 16): string {
	// Private pair memory sized for the tape (both word-padded tapes); never
	// smaller than the 40 bytes the 32/38-byte differential tests use.
	const memWords = Math.max(40, Math.ceil(tapeLength / 4) * 2 * 4);
	return `
struct Params {
	soup_width: u32,
	soup_height: u32,
	tape_length: u32,
	pair_length: u32,
	pair_count: u32,
	mutation_count: u32,
	z80_steps: u32,
	batch_seed: u32,
	suppress: array<vec4<u32>, 6>,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> pair_data: array<u32>; // one word-padded A+B pair per case
@group(0) @binding(2) var<storage, read_write> regs: array<u32>;

// Same host bits (memory model, pair layout, suppression hook) as the sim,
// so this tests the exact shipping Z80 core from the zilion package.
var<private> cpu_writes_a: u32;
var<private> cpu_writes_b: u32;
var<private> mem: array<u32, ${memWords}>;
fn mem_read(addr: u32) -> u32 { return mem[addr % params.pair_length]; }
fn mem_write(addr: u32, val: u32) {
	let a = addr % params.pair_length;
	mem[a] = val & 0xffu;
	if (a < params.tape_length) { cpu_writes_a += 1u; } else { cpu_writes_b += 1u; }
}
${SUPPRESS_HOOK_WGSL}
${SP_INIT_WGSL}
${PAIR_MEM_WGSL}

${Z80_CORE_WGSL}

@compute @workgroup_size(64)
fn z80_test(@builtin(global_invocation_id) id: vec3u) {
	let case_id = id.x;
	if (case_id >= params.pair_count) { return; }
	let base = case_id * ((params.tape_length + 3u) / 4u) * 2u;
	load_pair_mem(base);
	cpu_a=0u; cpu_f=0u; cpu_b=0u; cpu_c=0u; cpu_d=0u; cpu_e=0u; cpu_h=0u; cpu_l=0u;
	cpu_sp=sp_init(); cpu_pc=0u;
	cpu_a2=0u; cpu_f2=0u; cpu_b2=0u; cpu_c2=0u; cpu_d2=0u; cpu_e2=0u; cpu_h2=0u; cpu_l2=0u;
	cpu_ix=0u; cpu_iy=0u;
	cpu_halted=0u; cpu_iff1=0u; cpu_iff2=0u; cpu_writes_a=0u; cpu_writes_b=0u;
	for (var s = 0u; s < params.z80_steps; s++) {
		if (cpu_halted != 0u) { break; }
		z80_step();
	}
	store_pair_mem(base);
	let rbase = case_id * 12u;
	regs[rbase+0u]=cpu_a; regs[rbase+1u]=cpu_f; regs[rbase+2u]=cpu_b; regs[rbase+3u]=cpu_c;
	regs[rbase+4u]=cpu_d; regs[rbase+5u]=cpu_e; regs[rbase+6u]=cpu_h; regs[rbase+7u]=cpu_l;
	regs[rbase+8u]=cpu_sp; regs[rbase+9u]=cpu_pc; regs[rbase+10u]=cpu_writes_a; regs[rbase+11u]=cpu_writes_b;
}
`;
}
