# Night of 9–10 October 2026: results

Each item gives the pre-registration (REVISION_PREREG.md), the outcome and the source table. Nothing here has been put
into the manuscript.

## N1 — the toxic payload (rejected)
*Hypothesis:* bytes a transmitter never runs are selected to halt intruders (0x00 under lethal tar, 0x76 under both).
*Outcome:* rejected. Payload zeros are inherited scars of stack damage: they rise towards the end of the payload (to
0.32–0.38 at the second-last byte at L = 32) with an even–odd pattern, are as common under benign as under lethal tar
(L = 32: 0.067 against 0.052), and are absent without mutation. 0x76 is not enriched. Removing the halting bytes lowers a
host's survival as an intact partner by about 0.001 per encounter (registered threshold 0.02). Sources:
`results/toxin/COMPOSITION.md`, `results/toxin/CAUSAL.md`.
*What it gives the paper:* transmitters inherit the record of the damage done to them; regenerators erase it.

## N2 — what differs between the tars (running)
N2-2 refuted the backup hypothesis directly: a regenerator with a zero anywhere in its first core never copies itself,
under either rule (0 of 4,096), because its LDIR with BC = 0 consumes the budget before execution can reach a backup core.
`results/n2/BACKUP.md`. The in-situ demography (N2-1) is running.

## N3 — the line of descent of the first self-confined replicators (running on Modal)
Validation: exact replay of 795,934 recorded encounters (0 mismatches), accounting within the mutation count, and all
64 sampled confined copiers in a seeded world traced to the seeded closer. `runs/lod/validate/validate.json`.
