# Traced mechanism classifier

Classification applies to the episode ending at the last partner write, not to
the opcode names in a periodic word. All 17 encounters receive separate labels.

Literal writers require every effective partner write in the episode to have
reset/immediate register provenance through the traced admitted forms. The
register inventory follows byte/word immediate and MOV loads, LODS, POP,
LES/LDS, XCHG and direct PUSH/STOS/MOV stores, plus segment transfers. Register-producing forms outside this conservative
inventory invalidate register provenance until a traced load restores it.
Modified memory-derived values are not called exact data copies. Every register byte carries an exact read-event identity and value. Byte
loads update only the named byte; an unresolved byte in a word store is other.

Memory-fed copy writes require the actual ordered source byte at each store
position (low then high for words), with exact read-event provenance, through
MOVS, memory PUSH or a tracked word load followed by register PUSH/STOS. A
copy loop additionally repeats full instruction-start addresses and fetches
no partner position; distinct full start addresses mark the admitted one-pass
forms. Ring positions alone must not convert straight execution into a loop.

Arithmetic/shift RMW stores that write back exactly the byte just read at the
same address add transactions and write coverage but no new value provenance;
they are ignored for mechanism classification. They remain fully present in
the traces and all coverage/inflow columns. This matters for B9 51 bootstrap's
zero-effect ADDs between immediate-loaded pushes. Changed or unresolved RMW
values are other. Mixed literal/memory writes and trace overflow are other.

These classes identify admitted writing mechanisms, not whole-tape exactness,
causal hereditary capacity, partner independence or a population route.
Copy scores, unique write coverage, original partner reads and empirical
partner-output differences are separate measurements. A label can describe
a partial writer; exact/near-copy thresholds must be read from their columns.

Correction 2026-10-12: instruction-start addresses come from the hook's pc
argument, including prefixes and retained REP. Implicit pointer/count/stack
arithmetic never preserves a memory-copy identity, even if a result coincides
with another read byte. PUSH SP uses this kernel's decremented stored value.
All push/pop helpers, segment stack operations, strings, REP and LOOP,
call/return/interrupt paths, and implicit accumulator/DX transformations were
audited; unsupported transfers poison provenance. See correction_labels.csv
and correction_provenance.json for the complete fresh census re-audit.

B6 correction 2026-10-12: immediate far JMP (EA) sets CS's two tags to
literal, independently of the old CS and of coincidental operand equality.
Register LES/LDS traps invalidate the bank; no absent read creates an empty
register tag. A zero-count REP leaves AX/SI/DI/CX untouched. TF vector traffic
following an admitted instruction invalidates the post-instruction bank.
The total opcode/ModRM table is b6_instruction_audit.csv in S2 prelaunch;
per-class regression tests and a separate flat value-tag oracle cover the
conservative contract. The differential compares ordered effective write
channels as well as final classes, over every archived census candidate and
seeded random fixed programs. Raw census outcomes and the registered R0–R2
counts and 07 A5 witness are unchanged; additional non-exact/random label
corrections are listed separately in b6_correction_labels.csv.


## Prospective descriptive correction — 2026-10-12

The preceding trace-only class implementation is historical. For S2 and the
final census correction, a writing claim must additionally pass both isolated
55/AA source substitutions in `prefix_causality8086.py`: operand-fetch returns,
memory-read returns, or the exact reset register/segment source. The actual
ordered instruction-start/fetch/data-read/write addresses must be identical
through each claimed write, and its value must follow the source expression.
Later structural divergence is permitted. Every partner store, including a
neutral RMW store, has to be supported and verified; neutral stores contribute
coverage but cannot supply a new writing channel. Mixed, unsupported, failed,
interrupt/vector and overflow cases are `other`, with reasons retained.

All four mechanism classes are descriptive. R2a/R2b are withdrawn as
confirmatory predictions before population data because causal verification
repeatedly changed their supporting labels. No label change stops the audit.
The latest dated correction and complete `descriptive_correction_labels.csv`
are authoritative for labels and counts; raw machine outputs, fetch-based
openness, R0/R1/kill and L1/L2 proof obligations are unchanged.
