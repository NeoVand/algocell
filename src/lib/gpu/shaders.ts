// All WGSL shaders as string constants

// ============================================================
// SIMULATION COMPUTE SHADER
// Z80 emulator + batch management, fully parallel on GPU
// ============================================================
export const simShader = /* wgsl */ `

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

// === Z80 CPU State (per invocation) ===
var<private> cpu_a: u32;
var<private> cpu_f: u32;
var<private> cpu_b: u32;
var<private> cpu_c: u32;
var<private> cpu_d: u32;
var<private> cpu_e: u32;
var<private> cpu_h: u32;
var<private> cpu_l: u32;
var<private> cpu_sp: u32;
var<private> cpu_pc: u32;
var<private> cpu_a2: u32;
var<private> cpu_f2: u32;
var<private> cpu_b2: u32;
var<private> cpu_c2: u32;
var<private> cpu_d2: u32;
var<private> cpu_e2: u32;
var<private> cpu_h2: u32;
var<private> cpu_l2: u32;
var<private> cpu_halted: u32;
var<private> cpu_writes_a: u32;
var<private> cpu_writes_b: u32;
var<private> cpu_iff1: u32;
var<private> cpu_iff2: u32;

// Pair memory: 32 bytes, one byte per u32 element
var<private> mem: array<u32, 32>;

// Z80 flag bits
const CF: u32 = 0x01u;
const NF: u32 = 0x02u;
const PF: u32 = 0x04u;
const F3: u32 = 0x08u;
const HF: u32 = 0x10u;
const F5: u32 = 0x20u;
const ZF: u32 = 0x40u;
const SFl: u32 = 0x80u;

// === Memory Access ===
fn mem_read(addr: u32) -> u32 {
    return mem[addr & 31u];
}

fn mem_write(addr: u32, val: u32) {
    let a = addr & 31u;
    mem[a] = val & 0xffu;
    if (a < 16u) { cpu_writes_a += 1u; } else { cpu_writes_b += 1u; }
}

fn z80_fetch() -> u32 {
    let val = mem_read(cpu_pc);
    cpu_pc = (cpu_pc + 1u) & 0xffffu;
    return val;
}

fn z80_fetch_word() -> u32 {
    let lo = z80_fetch();
    let hi = z80_fetch();
    return (hi << 8u) | lo;
}

fn z80_push16(val: u32) {
    cpu_sp = (cpu_sp - 1u) & 0xffffu;
    mem_write(cpu_sp, (val >> 8u) & 0xffu);
    cpu_sp = (cpu_sp - 1u) & 0xffffu;
    mem_write(cpu_sp, val & 0xffu);
}

fn z80_pop16() -> u32 {
    let lo = mem_read(cpu_sp);
    cpu_sp = (cpu_sp + 1u) & 0xffffu;
    let hi = mem_read(cpu_sp);
    cpu_sp = (cpu_sp + 1u) & 0xffffu;
    return (hi << 8u) | lo;
}

fn signed_byte(b: u32) -> i32 {
    let sb = i32(b);
    if (sb > 127) { return sb - 256; }
    return sb;
}

// === Register Access ===
fn get_bc() -> u32 { return (cpu_b << 8u) | cpu_c; }
fn get_de() -> u32 { return (cpu_d << 8u) | cpu_e; }
fn get_hl() -> u32 { return (cpu_h << 8u) | cpu_l; }
fn get_af() -> u32 { return (cpu_a << 8u) | cpu_f; }

fn set_bc(v: u32) { cpu_b = (v >> 8u) & 0xffu; cpu_c = v & 0xffu; }
fn set_de(v: u32) { cpu_d = (v >> 8u) & 0xffu; cpu_e = v & 0xffu; }
fn set_hl(v: u32) { cpu_h = (v >> 8u) & 0xffu; cpu_l = v & 0xffu; }
fn set_af(v: u32) { cpu_a = (v >> 8u) & 0xffu; cpu_f = v & 0xffu; }

fn get_reg(idx: u32) -> u32 {
    switch(idx) {
        case 0u: { return cpu_b; }
        case 1u: { return cpu_c; }
        case 2u: { return cpu_d; }
        case 3u: { return cpu_e; }
        case 4u: { return cpu_h; }
        case 5u: { return cpu_l; }
        case 6u: { return mem_read(get_hl()); }
        case 7u: { return cpu_a; }
        default: { return 0u; }
    }
}

fn set_reg(idx: u32, val: u32) {
    let v = val & 0xffu;
    switch(idx) {
        case 0u: { cpu_b = v; }
        case 1u: { cpu_c = v; }
        case 2u: { cpu_d = v; }
        case 3u: { cpu_e = v; }
        case 4u: { cpu_h = v; }
        case 5u: { cpu_l = v; }
        case 6u: { mem_write(get_hl(), v); }
        case 7u: { cpu_a = v; }
        default: {}
    }
}

fn get_reg16(idx: u32) -> u32 {
    switch(idx) {
        case 0u: { return get_bc(); }
        case 1u: { return get_de(); }
        case 2u: { return get_hl(); }
        case 3u: { return cpu_sp; }
        default: { return 0u; }
    }
}

fn set_reg16(idx: u32, val: u32) {
    let v = val & 0xffffu;
    switch(idx) {
        case 0u: { set_bc(v); }
        case 1u: { set_de(v); }
        case 2u: { set_hl(v); }
        case 3u: { cpu_sp = v; }
        default: {}
    }
}

fn get_reg16_af(idx: u32) -> u32 {
    if (idx == 3u) { return get_af(); }
    return get_reg16(idx);
}

fn set_reg16_af(idx: u32, val: u32) {
    if (idx == 3u) { set_af(val); } else { set_reg16(idx, val); }
}

// === Flag Helpers ===
fn sz_flags(val: u32) -> u32 {
    var f = val & SFl;
    if (val == 0u) { f |= ZF; }
    f |= val & (F3 | F5);
    return f;
}

fn parity(val: u32) -> bool {
    var p = val;
    p ^= p >> 4u;
    p ^= p >> 2u;
    p ^= p >> 1u;
    return (p & 1u) == 0u;
}

fn check_cc(cc: u32) -> bool {
    switch(cc) {
        case 0u: { return (cpu_f & ZF) == 0u; }
        case 1u: { return (cpu_f & ZF) != 0u; }
        case 2u: { return (cpu_f & CF) == 0u; }
        case 3u: { return (cpu_f & CF) != 0u; }
        case 4u: { return (cpu_f & PF) == 0u; }
        case 5u: { return (cpu_f & PF) != 0u; }
        case 6u: { return (cpu_f & SFl) == 0u; }
        case 7u: { return (cpu_f & SFl) != 0u; }
        default: { return false; }
    }
}

// === ALU ===
fn z80_alu(op: u32, val: u32) {
    let a = cpu_a;
    let c = cpu_f & CF;
    switch(op) {
        case 0u: { // ADD
            let r = a + val;
            cpu_f = sz_flags(r & 0xffu) | select(0u, CF, r > 0xffu) |
                    ((a ^ val ^ r) & HF) |
                    select(0u, PF, ((~(a ^ val)) & (a ^ r) & 0x80u) != 0u);
            cpu_a = r & 0xffu;
        }
        case 1u: { // ADC
            let r = a + val + c;
            cpu_f = sz_flags(r & 0xffu) | select(0u, CF, r > 0xffu) |
                    ((a ^ val ^ r) & HF) |
                    select(0u, PF, ((~(a ^ val)) & (a ^ r) & 0x80u) != 0u);
            cpu_a = r & 0xffu;
        }
        case 2u: { // SUB
            let r = i32(a) - i32(val);
            let ru = u32(r) & 0xffu;
            cpu_f = sz_flags(ru) | NF | select(0u, CF, r < 0) |
                    ((a ^ val ^ u32(r)) & HF) |
                    select(0u, PF, (((a ^ val) & (a ^ u32(r))) & 0x80u) != 0u);
            cpu_a = ru;
        }
        case 3u: { // SBC
            let r = i32(a) - i32(val) - i32(c);
            let ru = u32(r) & 0xffu;
            cpu_f = sz_flags(ru) | NF | select(0u, CF, r < 0) |
                    ((a ^ val ^ u32(r)) & HF) |
                    select(0u, PF, (((a ^ val) & (a ^ u32(r))) & 0x80u) != 0u);
            cpu_a = ru;
        }
        case 4u: { // AND
            cpu_a = a & val;
            cpu_f = sz_flags(cpu_a) | HF | select(0u, PF, parity(cpu_a));
        }
        case 5u: { // XOR
            cpu_a = a ^ val;
            cpu_f = sz_flags(cpu_a) | select(0u, PF, parity(cpu_a));
        }
        case 6u: { // OR
            cpu_a = a | val;
            cpu_f = sz_flags(cpu_a) | select(0u, PF, parity(cpu_a));
        }
        case 7u: { // CP
            let r = i32(a) - i32(val);
            let ru = u32(r) & 0xffu;
            cpu_f = (ru & SFl) | select(0u, ZF, ru == 0u) | (val & (F3 | F5)) | NF |
                    select(0u, CF, r < 0) | ((a ^ val ^ u32(r)) & HF) |
                    select(0u, PF, (((a ^ val) & (a ^ u32(r))) & 0x80u) != 0u);
        }
        default: {}
    }
}

fn z80_inc8(val: u32) -> u32 {
    let r = (val + 1u) & 0xffu;
    cpu_f = (cpu_f & CF) | sz_flags(r) |
            select(0u, PF, val == 0x7fu) |
            select(0u, HF, (r & 0x0fu) == 0u);
    return r;
}

fn z80_dec8(val: u32) -> u32 {
    let r = (val - 1u) & 0xffu;
    cpu_f = (cpu_f & CF) | sz_flags(r) | NF |
            select(0u, PF, val == 0x80u) |
            select(0u, HF, (val & 0x0fu) == 0u);
    return r;
}

fn z80_add_hl(val: u32) {
    let hl = get_hl();
    let r = hl + val;
    cpu_f = (cpu_f & (SFl | ZF | PF)) |
            select(0u, CF, r > 0xffffu) |
            select(0u, HF, ((hl ^ val ^ r) & 0x1000u) != 0u) |
            ((r >> 8u) & (F3 | F5));
    set_hl(r & 0xffffu);
}

// === Rotate/Shift for accumulator ===
fn z80_rot_accum(y: u32) {
    let a = cpu_a;
    let c = cpu_f & CF;
    let keep = cpu_f & (SFl | ZF | PF);
    switch(y) {
        case 0u: { // RLCA
            cpu_a = ((a << 1u) | (a >> 7u)) & 0xffu;
            cpu_f = keep | (a >> 7u) | (cpu_a & (F3 | F5));
        }
        case 1u: { // RRCA
            cpu_a = ((a >> 1u) | (a << 7u)) & 0xffu;
            cpu_f = keep | (a & 1u) | (cpu_a & (F3 | F5));
        }
        case 2u: { // RLA
            cpu_a = ((a << 1u) | c) & 0xffu;
            cpu_f = keep | (a >> 7u) | (cpu_a & (F3 | F5));
        }
        case 3u: { // RRA
            cpu_a = ((a >> 1u) | (c << 7u)) & 0xffu;
            cpu_f = keep | (a & 1u) | (cpu_a & (F3 | F5));
        }
        case 4u: { // DAA
            var correction = 0u;
            var carry = c;
            if ((cpu_f & HF) != 0u || (a & 0x0fu) > 9u) { correction |= 0x06u; }
            if (c != 0u || a > 0x99u) { correction |= 0x60u; carry = 1u; }
            if ((cpu_f & NF) != 0u) { cpu_a = (a - correction) & 0xffu; }
            else { cpu_a = (a + correction) & 0xffu; }
            cpu_f = (cpu_f & NF) | sz_flags(cpu_a) | carry |
                    ((a ^ cpu_a) & HF) | select(0u, PF, parity(cpu_a));
        }
        case 5u: { // CPL
            cpu_a = (~a) & 0xffu;
            cpu_f = (cpu_f & (SFl | ZF | PF | CF)) | HF | NF | (cpu_a & (F3 | F5));
        }
        case 6u: { // SCF
            cpu_f = (cpu_f & (SFl | ZF | PF)) | CF | (cpu_a & (F3 | F5));
        }
        case 7u: { // CCF
            cpu_f = (cpu_f & (SFl | ZF | PF)) |
                    select(0u, HF, c != 0u) |
                    select(CF, 0u, c != 0u) |
                    (cpu_a & (F3 | F5));
        }
        default: {}
    }
}

// === CB Prefix (bit ops, rotates, shifts) ===
fn z80_cb_rot(op: u32, val: u32) -> u32 {
    let c = cpu_f & CF;
    var r = 0u;
    switch(op) {
        case 0u: { r = ((val << 1u) | (val >> 7u)) & 0xffu; cpu_f = sz_flags(r) | (val >> 7u) | select(0u, PF, parity(r)); }
        case 1u: { r = ((val >> 1u) | (val << 7u)) & 0xffu; cpu_f = sz_flags(r) | (val & 1u) | select(0u, PF, parity(r)); }
        case 2u: { r = ((val << 1u) | c) & 0xffu; cpu_f = sz_flags(r) | (val >> 7u) | select(0u, PF, parity(r)); }
        case 3u: { r = ((val >> 1u) | (c << 7u)) & 0xffu; cpu_f = sz_flags(r) | (val & 1u) | select(0u, PF, parity(r)); }
        case 4u: { r = (val << 1u) & 0xffu; cpu_f = sz_flags(r) | (val >> 7u) | select(0u, PF, parity(r)); }
        case 5u: { r = ((val >> 1u) | (val & 0x80u)) & 0xffu; cpu_f = sz_flags(r) | (val & 1u) | select(0u, PF, parity(r)); }
        case 6u: { r = ((val << 1u) | 1u) & 0xffu; cpu_f = sz_flags(r) | (val >> 7u) | select(0u, PF, parity(r)); }
        case 7u: { r = (val >> 1u) & 0xffu; cpu_f = sz_flags(r) | (val & 1u) | select(0u, PF, parity(r)); }
        default: { r = val; }
    }
    return r;
}

fn z80_exec_cb() {
    let op = z80_fetch();
    let x = (op >> 6u) & 3u;
    let y = (op >> 3u) & 7u;
    let z = op & 7u;
    let val = get_reg(z);
    switch(x) {
        case 0u: { set_reg(z, z80_cb_rot(y, val)); }
        case 1u: { // BIT
            cpu_f = (cpu_f & CF) | HF |
                    select(0u, ZF | PF, (val & (1u << y)) == 0u) |
                    select(0u, SFl, y == 7u && (val & 0x80u) != 0u) |
                    (val & (F3 | F5));
        }
        case 2u: { set_reg(z, val & ~(1u << y)); }
        case 3u: { set_reg(z, val | (1u << y)); }
        default: {}
    }
}

// === Block Transfer (ED prefix) ===
fn z80_ldi() {
    let val = mem_read(get_hl());
    mem_write(get_de(), val);
    set_hl((get_hl() + 1u) & 0xffffu);
    set_de((get_de() + 1u) & 0xffffu);
    set_bc((get_bc() - 1u) & 0xffffu);
    let n = (val + cpu_a) & 0xffu;
    cpu_f = (cpu_f & (SFl | ZF | CF)) |
            select(0u, PF, get_bc() != 0u) |
            (n & F3) | select(0u, F5, (n & 0x02u) != 0u);
}

fn z80_ldd() {
    let val = mem_read(get_hl());
    mem_write(get_de(), val);
    set_hl((get_hl() - 1u) & 0xffffu);
    set_de((get_de() - 1u) & 0xffffu);
    set_bc((get_bc() - 1u) & 0xffffu);
    let n = (val + cpu_a) & 0xffu;
    cpu_f = (cpu_f & (SFl | ZF | CF)) |
            select(0u, PF, get_bc() != 0u) |
            (n & F3) | select(0u, F5, (n & 0x02u) != 0u);
}

fn z80_cpi() {
    let val = mem_read(get_hl());
    let r = (cpu_a - val) & 0xffu;
    set_hl((get_hl() + 1u) & 0xffffu);
    set_bc((get_bc() - 1u) & 0xffffu);
    cpu_f = (cpu_f & CF) | sz_flags(r) | NF |
            ((cpu_a ^ val ^ r) & HF) |
            select(0u, PF, get_bc() != 0u);
}

fn z80_cpd() {
    let val = mem_read(get_hl());
    let r = (cpu_a - val) & 0xffu;
    set_hl((get_hl() - 1u) & 0xffffu);
    set_bc((get_bc() - 1u) & 0xffffu);
    cpu_f = (cpu_f & CF) | sz_flags(r) | NF |
            ((cpu_a ^ val ^ r) & HF) |
            select(0u, PF, get_bc() != 0u);
}

// === ED Prefix ===
fn z80_exec_ed() {
    let op = z80_fetch();
    switch(op) {
        case 0xa0u: { z80_ldi(); }
        case 0xa8u: { z80_ldd(); }
        case 0xb0u: { z80_ldi(); if (get_bc() != 0u) { cpu_pc = (cpu_pc - 2u) & 0xffffu; } } // LDIR
        case 0xb8u: { z80_ldd(); if (get_bc() != 0u) { cpu_pc = (cpu_pc - 2u) & 0xffffu; } } // LDDR
        case 0xa1u: { z80_cpi(); }
        case 0xa9u: { z80_cpd(); }
        case 0xb1u: { z80_cpi(); if (get_bc() != 0u && (cpu_f & ZF) == 0u) { cpu_pc = (cpu_pc - 2u) & 0xffffu; } }
        case 0xb9u: { z80_cpd(); if (get_bc() != 0u && (cpu_f & ZF) == 0u) { cpu_pc = (cpu_pc - 2u) & 0xffffu; } }
        // NEG
        case 0x44u, 0x4cu, 0x54u, 0x5cu, 0x64u, 0x6cu, 0x74u, 0x7cu: {
            let a = cpu_a; cpu_a = 0u; z80_alu(2u, a);
        }
        // RETN/RETI
        case 0x45u, 0x4du, 0x55u, 0x5du, 0x65u, 0x6du, 0x75u, 0x7du: {
            cpu_iff1 = cpu_iff2; cpu_pc = z80_pop16();
        }
        // LD I,A / LD R,A / LD A,I / LD A,R
        case 0x47u: {} // LD I,A - no I register in our sim
        case 0x4fu: {} // LD R,A
        case 0x57u: { cpu_f = (cpu_f & CF) | sz_flags(cpu_a); } // LD A,I simplified
        case 0x5fu: { cpu_f = (cpu_f & CF) | sz_flags(cpu_a); } // LD A,R simplified
        // LD (nn), rr
        case 0x43u, 0x53u, 0x63u, 0x73u: {
            let nn = z80_fetch_word();
            let rp = (op >> 4u) & 3u;
            let val = get_reg16(rp);
            mem_write(nn, val & 0xffu);
            mem_write((nn + 1u) & 0xffffu, (val >> 8u) & 0xffu);
        }
        // LD rr, (nn)
        case 0x4bu, 0x5bu, 0x6bu, 0x7bu: {
            let nn = z80_fetch_word();
            let rp = (op >> 4u) & 3u;
            let lo = mem_read(nn);
            let hi = mem_read((nn + 1u) & 0xffffu);
            set_reg16(rp, (hi << 8u) | lo);
        }
        // ADC HL, rr
        case 0x4au, 0x5au, 0x6au, 0x7au: {
            let rp = (op >> 4u) & 3u;
            let hl = get_hl();
            let val = get_reg16(rp);
            let c = cpu_f & CF;
            let r = hl + val + c;
            cpu_f = ((r >> 8u) & SFl) | select(0u, ZF, (r & 0xffffu) == 0u) |
                    select(0u, HF, ((hl ^ val ^ r) & 0x1000u) != 0u) |
                    select(0u, PF, ((~(hl ^ val)) & (hl ^ r) & 0x8000u) != 0u) |
                    select(0u, CF, r > 0xffffu) | ((r >> 8u) & (F3 | F5));
            set_hl(r & 0xffffu);
        }
        // SBC HL, rr
        case 0x42u, 0x52u, 0x62u, 0x72u: {
            let rp = (op >> 4u) & 3u;
            let hl = get_hl();
            let val = get_reg16(rp);
            let c = cpu_f & CF;
            let r = i32(hl) - i32(val) - i32(c);
            let ru = u32(r) & 0xffffu;
            cpu_f = ((ru >> 8u) & SFl) | select(0u, ZF, ru == 0u) | NF |
                    select(0u, HF, ((hl ^ val ^ u32(r)) & 0x1000u) != 0u) |
                    select(0u, PF, (((hl ^ val) & (hl ^ u32(r))) & 0x8000u) != 0u) |
                    select(0u, CF, r < 0) | ((ru >> 8u) & (F3 | F5));
            set_hl(ru);
        }
        // RRD
        case 0x67u: {
            let m = mem_read(get_hl());
            mem_write(get_hl(), ((cpu_a << 4u) | (m >> 4u)) & 0xffu);
            cpu_a = (cpu_a & 0xf0u) | (m & 0x0fu);
            cpu_f = (cpu_f & CF) | sz_flags(cpu_a) | select(0u, PF, parity(cpu_a));
        }
        // RLD
        case 0x6fu: {
            let m = mem_read(get_hl());
            mem_write(get_hl(), ((m << 4u) | (cpu_a & 0x0fu)) & 0xffu);
            cpu_a = (cpu_a & 0xf0u) | (m >> 4u);
            cpu_f = (cpu_f & CF) | sz_flags(cpu_a) | select(0u, PF, parity(cpu_a));
        }
        // IN r,(C) - simplified, just set to 0
        case 0x40u, 0x48u, 0x50u, 0x58u, 0x60u, 0x68u, 0x70u, 0x78u: {
            set_reg((op >> 3u) & 7u, 0u);
        }
        default: {} // unknown ED ops = NOP
    }
}

// === Main Opcode Execution ===
fn z80_exec_x0(y: u32, z: u32, p: u32, q: u32) {
    switch(z) {
        case 0u: {
            switch(y) {
                case 0u: {} // NOP
                case 1u: { // EX AF,AF'
                    var t = cpu_a; cpu_a = cpu_a2; cpu_a2 = t;
                    t = cpu_f; cpu_f = cpu_f2; cpu_f2 = t;
                }
                case 2u: { // DJNZ
                    let d = signed_byte(z80_fetch());
                    cpu_b = (cpu_b - 1u) & 0xffu;
                    if (cpu_b != 0u) { cpu_pc = u32(i32(cpu_pc) + d) & 0xffffu; }
                }
                case 3u: { // JR
                    let d = signed_byte(z80_fetch());
                    cpu_pc = u32(i32(cpu_pc) + d) & 0xffffu;
                }
                default: { // JR cc (y-4)
                    let d = signed_byte(z80_fetch());
                    if (check_cc(y - 4u)) { cpu_pc = u32(i32(cpu_pc) + d) & 0xffffu; }
                }
            }
        }
        case 1u: {
            if (q == 0u) { set_reg16(p, z80_fetch_word()); }
            else { z80_add_hl(get_reg16(p)); }
        }
        case 2u: {
            if (q == 0u) {
                switch(p) {
                    case 0u: { mem_write(get_bc(), cpu_a); }
                    case 1u: { mem_write(get_de(), cpu_a); }
                    case 2u: { let nn = z80_fetch_word(); mem_write(nn, cpu_l); mem_write((nn+1u) & 0xffffu, cpu_h); }
                    case 3u: { mem_write(z80_fetch_word(), cpu_a); }
                    default: {}
                }
            } else {
                switch(p) {
                    case 0u: { cpu_a = mem_read(get_bc()); }
                    case 1u: { cpu_a = mem_read(get_de()); }
                    case 2u: { let nn = z80_fetch_word(); cpu_l = mem_read(nn); cpu_h = mem_read((nn+1u) & 0xffffu); }
                    case 3u: { cpu_a = mem_read(z80_fetch_word()); }
                    default: {}
                }
            }
        }
        case 3u: {
            if (q == 0u) { set_reg16(p, (get_reg16(p) + 1u) & 0xffffu); }
            else { set_reg16(p, (get_reg16(p) - 1u) & 0xffffu); }
        }
        case 4u: { set_reg(y, z80_inc8(get_reg(y))); }
        case 5u: { set_reg(y, z80_dec8(get_reg(y))); }
        case 6u: { set_reg(y, z80_fetch()); }
        case 7u: { z80_rot_accum(y); }
        default: {}
    }
}

fn z80_exec_x3(y: u32, z: u32, p: u32, q: u32) {
    switch(z) {
        case 0u: { if (check_cc(y)) { cpu_pc = z80_pop16(); } }
        case 1u: {
            if (q == 0u) { set_reg16_af(p, z80_pop16()); }
            else {
                switch(p) {
                    case 0u: { cpu_pc = z80_pop16(); } // RET
                    case 1u: { // EXX
                        var t = cpu_b; cpu_b = cpu_b2; cpu_b2 = t;
                        t = cpu_c; cpu_c = cpu_c2; cpu_c2 = t;
                        t = cpu_d; cpu_d = cpu_d2; cpu_d2 = t;
                        t = cpu_e; cpu_e = cpu_e2; cpu_e2 = t;
                        t = cpu_h; cpu_h = cpu_h2; cpu_h2 = t;
                        t = cpu_l; cpu_l = cpu_l2; cpu_l2 = t;
                    }
                    case 2u: { cpu_pc = get_hl(); }
                    case 3u: { cpu_sp = get_hl(); }
                    default: {}
                }
            }
        }
        case 2u: { let nn = z80_fetch_word(); if (check_cc(y)) { cpu_pc = nn; } }
        case 3u: {
            switch(y) {
                case 0u: { cpu_pc = z80_fetch_word(); } // JP nn
                case 1u: { z80_exec_cb(); }
                case 2u: { z80_fetch(); } // OUT (n),A - consume, NOP
                case 3u: { cpu_a = z80_fetch() & 0xffu; } // IN A,(n) simplified
                case 4u: { // EX (SP),HL
                    let lo = mem_read(cpu_sp);
                    let hi = mem_read((cpu_sp + 1u) & 0xffffu);
                    mem_write(cpu_sp, cpu_l);
                    mem_write((cpu_sp + 1u) & 0xffffu, cpu_h);
                    cpu_h = hi; cpu_l = lo;
                }
                case 5u: { // EX DE,HL
                    let td = cpu_d; let te = cpu_e;
                    cpu_d = cpu_h; cpu_e = cpu_l;
                    cpu_h = td; cpu_l = te;
                }
                case 6u: { cpu_iff1 = 0u; cpu_iff2 = 0u; }
                case 7u: { cpu_iff1 = 1u; cpu_iff2 = 1u; }
                default: {}
            }
        }
        case 4u: { let nn = z80_fetch_word(); if (check_cc(y)) { z80_push16(cpu_pc); cpu_pc = nn; } }
        case 5u: {
            if (q == 0u) { z80_push16(get_reg16_af(p)); }
            else {
                switch(p) {
                    case 0u: { let nn = z80_fetch_word(); z80_push16(cpu_pc); cpu_pc = nn; } // CALL nn
                    case 1u, 3u: {} // DD/FD prefix handled in z80_step
                    case 2u: { z80_exec_ed(); }
                    default: {}
                }
            }
        }
        case 6u: { z80_alu(y, z80_fetch()); }
        case 7u: { z80_push16(cpu_pc); cpu_pc = y * 8u; } // RST
        default: {}
    }
}

fn z80_execute(op: u32) {
    let x = (op >> 6u) & 3u;
    let y = (op >> 3u) & 7u;
    let z = op & 7u;
    let p = (y >> 1u) & 3u;
    let q = y & 1u;
    switch(x) {
        case 0u: { z80_exec_x0(y, z, p, q); }
        case 1u: {
            if (y == 6u && z == 6u) { cpu_halted = 1u; }
            else { set_reg(y, get_reg(z)); }
        }
        case 2u: { z80_alu(y, get_reg(z)); }
        case 3u: { z80_exec_x3(y, z, p, q); }
        default: {}
    }
}

fn z80_step() {
    if (cpu_halted != 0u) { return; }
    var op = z80_fetch();
    // Handle DD/FD prefixes at top level to avoid recursion
    // (WGSL forbids recursive calls). Simplified: skip prefix, treat IX/IY as HL.
    for (var pfx = 0u; pfx < 4u; pfx++) {
        if (op != 0xddu && op != 0xfdu) { break; }
        op = z80_fetch();
    }
    z80_execute(op);
}

// === Soup byte access helpers ===
fn read_soup_byte(cell: u32, byte_idx: u32) -> u32 {
    let word_idx = cell * 4u + (byte_idx >> 2u);
    let shift = (byte_idx & 3u) * 8u;
    return (soup[word_idx] >> shift) & 0xffu;
}

fn write_soup_byte(cell: u32, byte_idx: u32, val: u32) {
    let word_idx = cell * 4u + (byte_idx >> 2u);
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
    let dir = rand_bounded(4u);
    let i = y * w + x;

    var nx = x;
    var ny = y;
    switch(dir) {
        case 0u: { nx = min(x + 1u, w - 1u); }
        case 1u: { ny = min(y + 1u, h - 1u); }
        case 2u: { if (x > 0u) { nx = x - 1u; } }
        case 3u: { if (y > 0u) { ny = y - 1u; } }
        default: {}
    }
    let j = ny * w + nx;

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
        let base = pair_id * 8u;
        for (var w_idx = 0u; w_idx < 4u; w_idx++) {
            pair_data[base + w_idx] = soup[i * 4u + w_idx];
        }
        for (var w_idx = 0u; w_idx < 4u; w_idx++) {
            pair_data[base + 4u + w_idx] = soup[j * 4u + w_idx];
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

    // Load pair memory into private array
    let base = pair_id * 8u;
    for (var i = 0u; i < 8u; i++) {
        let word = pair_data[base + i];
        mem[i * 4u] = word & 0xffu;
        mem[i * 4u + 1u] = (word >> 8u) & 0xffu;
        mem[i * 4u + 2u] = (word >> 16u) & 0xffu;
        mem[i * 4u + 3u] = (word >> 24u) & 0xffu;
    }

    // Reset CPU state
    cpu_a = 0u; cpu_f = 0u; cpu_b = 0u; cpu_c = 0u;
    cpu_d = 0u; cpu_e = 0u; cpu_h = 0u; cpu_l = 0u;
    cpu_sp = 0u; cpu_pc = 0u;
    cpu_a2 = 0u; cpu_f2 = 0u; cpu_b2 = 0u; cpu_c2 = 0u;
    cpu_d2 = 0u; cpu_e2 = 0u; cpu_h2 = 0u; cpu_l2 = 0u;
    cpu_halted = 0u;
    cpu_iff1 = 0u; cpu_iff2 = 0u;
    cpu_writes_a = 0u;
    cpu_writes_b = 0u;

    // Run Z80 steps
    for (var step = 0u; step < params.z80_steps; step++) {
        if (cpu_halted != 0u) { break; }
        z80_step();
    }

    // Save pair memory back
    for (var i = 0u; i < 8u; i++) {
        pair_data[base + i] = mem[i * 4u] |
                              (mem[i * 4u + 1u] << 8u) |
                              (mem[i * 4u + 2u] << 16u) |
                              (mem[i * 4u + 3u] << 24u);
    }

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
    let base = pair_id * 8u;

    for (var w_idx = 0u; w_idx < 4u; w_idx++) {
        soup[i * 4u + w_idx] = pair_data[base + w_idx];
    }
    for (var w_idx = 0u; w_idx < 4u; w_idx++) {
        soup[j * 4u + w_idx] = pair_data[base + 4u + w_idx];
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

    let word_idx = pos >> 2u;
    let shift = (pos & 3u) * 8u;
    let mask = ~(0xffu << shift);
    soup[word_idx] = (soup[word_idx] & mask) | (val << shift);
}

// --- Count bytes for statistics ---
@compute @workgroup_size(256)
fn count_bytes(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    let total_words = params.soup_width * params.soup_height * 4u; // 4 u32s per cell
    if (idx >= total_words) { return; }

    let word = soup[idx];
    atomicAdd(&byte_counts[word & 0xffu], 1u);
    atomicAdd(&byte_counts[(word >> 8u) & 0xffu], 1u);
    atomicAdd(&byte_counts[(word >> 16u) & 0xffu], 1u);
    atomicAdd(&byte_counts[(word >> 24u) & 0xffu], 1u);
}

// --- Clear byte counts ---
@compute @workgroup_size(256)
fn clear_byte_counts(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < 256u) {
        atomicStore(&byte_counts[id.x], 0u);
    }
}
`;

// ============================================================
// RENDER SHADER
// Full-screen quad rendering the soup as colored tiles
// ============================================================
export const renderShader = /* wgsl */ `

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
    _pad1: u32,
    _pad2: u32,
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
    let word_idx = cell * 4u + (byte_idx >> 2u);
    let shift = (byte_idx & 3u) * 8u;
    return (soup[word_idx] >> shift) & 0xffu;
}

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
        for (var i = 0u; i < 16u; i++) {
            let byte_val = read_soup_byte(cell_idx, i);
            let col = unpack_color(colormap[byte_val]);
            acc += col.rgb * col.rgb;
        }
        var avg = sqrt(acc / 16.0);

        // Highlight hovered cell
        if (rparams.hover_cell >= 0 && cell_idx == u32(rparams.hover_cell)) {
            avg = avg * 1.4 + vec3f(0.05);
        }

        // Subtle grid lines when somewhat zoomed in
        if (pixel_cell_size < 0.15) {
            let edge = step(0.97, max(frac_x, frac_y));
            avg = mix(avg, vec3f(0.15), edge * 0.3);
        }

        return vec4f(avg, 1.0);
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

    // Subtle byte grid lines when very zoomed in
    if (pixel_cell_size < 0.05) {
        let byte_edge = step(0.92, max(fract(frac_x * f32(ts)), fract(frac_y * f32(ts))));
        color = mix(color, vec3f(0.08), byte_edge * 0.4);
    }

    return vec4f(color, 1.0);
}
`;
