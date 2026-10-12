# Traced mechanism classifier

Classification applies to the episode ending at the last partner write, not to
the opcode names in a periodic word. All 17 encounters receive separate labels.

Literal writers require every effective partner write in the episode to have
reset/immediate register provenance through the traced admitted forms. The
register inventory follows word immediate loads, word MOV loads, LODSW, POP,
XCHG and direct PUSH/STOS. Register-producing forms outside this conservative
inventory invalidate register provenance until a traced load restores it.
Modified memory-derived values are not called exact data copies. Partial byte
loads with unresolved high-byte provenance are other.

Memory-fed copy writes require paired actual same-value data reads through
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
