"""Total table of the vendored WGSL's opcode classes (not a hardware ISA claim).

The trace classifier admits exact byte transfers and a small neutral inventory.
Other classes conservatively invalidate all GPR/segment tags. This is intentional:
unsupported arithmetic, flags and vector paths cannot preserve a memory-copy tag.
WGSL anchors are checked by the audit; both L16 and L32 have identical core code.
"""
CLASSES = {
    'alu': ('shared arithmetic phase', 'destination GPR/rm or flags; CMP/TEST no destination', 'conservative except memory-destination RMW'),
    'decimal_adjust': ('case 39u, 47u', 'AL or AX and flags', 'invalidate all'),
    'segment_push': ('case 6u, 14u', 'SP decrement; segment value unchanged', 'exact segment source; SP arithmetic'),
    'segment_pop': ('case 7u, 15u', 'SP increment; ES/CS/SS/DS overwrite', 'two ordered reads; SP arithmetic'),
    'inc_dec': ('op >= 64u && op <= 79u', 'named word GPR', 'literal stays literal, memory becomes unresolved'),
    'register_push': ('op >= 80u && op <= 87u', 'SP decrement; PUSH SP stores decremented SP', 'exact source except SP arithmetic'),
    'register_pop': ('op >= 88u && op <= 95u', 'SP increment then named GPR overwrite, including POP SP', 'two ordered reads; overwritten SP replaces increment tag'),
    'conditional_jump': ('op >= 96u && op <= 127u', 'IP only (60..6F aliases included)', 'preserve'),
    'xchg': ('op == 134u || op == 135u', 'reg plus rm; byte/word', 'exact register exchange; memory store conservatively unresolved'),
    'mov': ('op == 136u || op == 137u', 'reg/rm, byte or word', 'exact byte transfer'),
    'segment_mov': ('op == 140u', '8C word rm; 8E word segment reg&3', 'exact two-byte transfer (8C forces word)'),
    'lea': ('op == 141u', 'word GPR, register mode traps', 'invalidate all'),
    'pop_rm': ('op == 143u', 'SP increment then word rm', 'ordered stack reads; SP arithmetic'),
    'ax_exchange': ('op >= 144u && op <= 151u', 'AX and named word GPR', 'swap exact tags, including NOP'),
    'sign_extend': ('case 152u', 'CBW AX; CWD DX', 'invalidate all'),
    'immediate_far_call': ('case 154u', 'two SP decrements; CS immediate; IP immediate', 'invalidate all'),
    'flags_stack': ('case 156u', 'PUSHF SP; POPF SP and flags (TF possible)', 'invalidate all'),
    'flags_transfer': ('case 158u', 'SAHF flags; LAHF AH', 'invalidate all'),
    'accumulator_memory': ('op >= 160u && op <= 163u', 'AL/AX or memory', 'exact byte transfer'),
    'string': ('fn i8086_string', 'SI/DI; LODS AL/AX; REP CX only if CX nonzero', 'exact data, pointer/count arithmetic; zero REP preserves all tags'),
    'immediate': ('op >= 176u && op <= 191u', 'named byte/word GPR', 'literal overwrite'),
    'return': ('case 192u, 193u', 'SP pops+extra; far return CS overwrite', 'invalidate all'),
    'far_load': ('op == 196u || op == 197u', 'word reg then ES/DS; register mode traps', 'four ordered reads'),
    'immediate_rm': ('op == 198u || op == 199u', 'byte/word rm (all reg fields accepted)', 'literal overwrite'),
    'interrupt': ('case 204u', 'IVT reads, SP decrements, CS; IRET pops SP/CS/flags', 'invalidate all; TF after any opcode also invalidates'),
    'shift': ('op >= 208u && op <= 211u', 'byte/word rm and flags; count zero preserves numeric value', 'invalidate all; unchanged address-paired RMW ignored'),
    'aam_aad_salc': ('case 212u', 'AX/AL and flags; AAM zero can vector', 'invalidate all'),
    'xlat': ('case 215u', 'AL', 'invalidate all (not admitted)'),
    'escape': ('op >= 216u && op <= 223u', 'no GPR/segment writes; may read word memory', 'invalidate all (not admitted)'),
    'loop': ('case 224u', 'CX decrement except JCXZ; IP', 'CX arithmetic; JCXZ preserve'),
    'io': ('case 228u', 'IN AL/AX=FF/FFFF; OUT no GPR/segment writes', 'invalidate all (not admitted)'),
    'near_call': ('case 232u', 'SP decrement; IP', 'invalidate all'),
    'near_jump': ('case 233u', 'IP', 'preserve'),
    'immediate_far_jump': ('case 234u', 'CS immediate overwrite and IP', 'CS literal overwrite; GPRs and other segments preserve'),
    'muldiv_group': ('op == 246u || op == 247u', 'TEST flags; NOT/NEG rm; MUL/DIV AX and word DX; divide trap CS/SP', 'invalidate all'),
    'inc_transfer_push_group': ('op == 254u || op == 255u', 'INC/DEC rm; calls SP; far transfers CS; /6,/7 SP decrement', 'exact /6,/7 push; all other forms invalidate all'),
    'neutral': ('case 244u', 'flags, halt or IP only', 'preserve'),
    'unsupported': ('default: { i8086_raise(ZIL_TRAP_UNIMPLEMENTED', 'trap/halt', 'invalidate all'),
}
OPCODE_CLASS = ['unsupported'] * 256

def assign(name, ops):
    for op in ops: OPCODE_CLASS[op] = name

assign('alu', [o for o in range(64) if o & 7 < 6] + list(range(0x80,0x86)) + [0xa8,0xa9])
assign('decimal_adjust', [0x27,0x2f,0x37,0x3f])
assign('segment_push', [6,14,22,30]); assign('segment_pop',[7,15,23,31])
assign('inc_dec',range(0x40,0x50)); assign('register_push',range(0x50,0x58)); assign('register_pop',range(0x58,0x60))
assign('conditional_jump',range(0x60,0x80)); assign('xchg',[0x86,0x87]); assign('mov',range(0x88,0x8c))
assign('segment_mov',[0x8c,0x8e]); assign('lea',[0x8d]); assign('pop_rm',[0x8f]); assign('ax_exchange',range(0x90,0x98))
assign('sign_extend',[0x98,0x99]); assign('immediate_far_call',[0x9a]); assign('flags_stack',[0x9c,0x9d]); assign('flags_transfer',[0x9e,0x9f])
assign('accumulator_memory',range(0xa0,0xa4)); assign('string',[0xa4,0xa5,0xa6,0xa7,*range(0xaa,0xb0)])
assign('immediate',range(0xb0,0xc0)); assign('return',[0xc0,0xc1,0xc2,0xc3,0xc8,0xc9,0xca,0xcb])
assign('far_load',[0xc4,0xc5]); assign('immediate_rm',[0xc6,0xc7]); assign('interrupt',range(0xcc,0xd0))
assign('shift',range(0xd0,0xd4)); assign('aam_aad_salc',[0xd4,0xd5,0xd6]); assign('xlat',[0xd7]); assign('escape',range(0xd8,0xe0))
assign('loop',range(0xe0,0xe4)); assign('io',[0xe4,0xe5,0xe6,0xe7,0xec,0xed,0xee,0xef]); assign('near_call',[0xe8]); assign('near_jump',[0xe9,0xeb]); assign('immediate_far_jump',[0xea])
assign('muldiv_group',[0xf6,0xf7]); assign('inc_transfer_push_group',[0xfe,0xff]); assign('neutral',[0x9b,0xf4,0xf5,*range(0xf8,0xfe)])
OPCODE_CLASS = tuple(OPCODE_CLASS)
