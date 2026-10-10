# Research plan for the night of 9–10 October 2026

Two questions, both bigger than anything in revision 2:

- **A. Do the bytes a transmitter never runs have a job?** Hypothesis: they are a poison that only intruders run.
- **B. Does the closed organism descend from the open one, or replace it?** Exact genealogy of every copy.

Either one, if it holds, is a new headline result. A also answers the question the paper leaves open: why the environment picks transmitters. Nothing below has been run yet.

Spent so far: about $80 of the $300. Cap for tonight: **$140**, which keeps the total under $230.

---

## A. Toxic payload: inert bytes that kill intruders

**Why we think this.** Each step below is from code or results we already have.
1. Transmitters copy bytes they never execute themselves: with the core `XX 5e ed b0` and copy offset d = L, that is d − 4 bytes.
2. Under lethal tar, a zero fetched as the *first byte of an instruction* halts the pair for the rest of the encounter (`gen_lethal_shader.py`). Under benign tar, a zero is a NOP.
3. So a zero in the payload costs its owner nothing, because its own pointer never reaches it. But it halts any intruder whose pointer runs into it.
4. The exploratory census already found that junk bodies protect while tiled-code bodies are hijacked (`INTRUDER.md`).
5. Transmitters win more as tar gets more lethal: 0.55–0.87 against 0.03–0.37 at L = 32, and the share rises with p on the dial.

**If it holds:** the unexecuted payload is a compartment of the genome whose meaning is read only by others. That is a defence that evolved without being designed, and it explains the environment's choice.

**Rival explanation, to rule out:** zeros pile up in payloads by *damage*. Intruders write tar into a transmitter and then halt. That would also enrich zeros without any benefit. Only the causal tests (A1, A2) separate the two.

### A0. Composition of existing worlds (free, local, about 1 hour)

**Data.** No new runs needed.
- 110 saved final soups of the E runs (`runs/offset/offset/*_final.npy`), all four tar × length cells with mutation on, plus the mutation-off controls.
- 360 lethality-dial worlds at L = 16 (`runs/dial/dial/`: p = 0, 0.01, 0.03, 0.1, 0.3 and 1; snapshots at 2k, 20k and 100k steps).

**Transmitter.** A carrier of `XX 5e ed b0` with d = XX mod 2L = L, plus the `1e NN ed b0` family. The payload is positions 4 to L − 1 after the core. For a sample of 256 transmitters, confirm with `exectrace` that the payload really is never executed by its owner.

**Pre-registered predictions** (written into REVISION_PREREG.md before any number is computed):
- **A0-1:** under lethal tar, the payload zero frequency is at least 3× that under benign tar at the same length, in at least 8 of 10 mix50 worlds at L = 32 and at L = 16. The null from drift is 1/256 ≈ 0.4%, because the founding payloads were uniform random bytes and mutation draws uniform bytes.
- **A0-2 (selection on standing variation, no mutation).** In the mutation-off lethal L = 16 worlds, the surviving tails carry more zeros than the founding tails did; in the benign mutation-off worlds they do not.
- **A0-3 (dose response).** Across the dial, the payload zero frequency of transmitters at 100k steps rises with p (Spearman ρ > 0 over the 60 worlds).
- **A0-4 (discovery, FDR-controlled).** Enrichment of all 256 byte values in the payload against the founding distribution, separately by tar. Prediction: 0x00 ranks first under lethal tar. Under benign tar, the top bytes (if any) tell us what protects there.
- **A0-5 (position, which separates poison from damage).**
  - Poison predicts zeros enriched where intruders enter. I will measure entry positions from intruder traces in the same soups.
  - Damage predicts zeros where intruders write, and zeros in the core too, which kill those cells.

**Gate.**
- If A0-1 and A0-3 both fail, the poison idea is dead. A keeps only A0-4 and A0-5 as descriptive results, and B gets the night.
- If either holds, go on to A1.

### A1. Causal test in encounters (free, local, about 2 hours)

**Setup.**
- **Hosts:** 200 evolved transmitters sampled from the final lethal L = 32 and L = 16 soups.
- **Three variants of each:**
  - *original*;
  - *detoxified*, every payload zero replaced by a random non-zero byte;
  - *sham*, the same number of payload positions re-drawn among non-zero bytes, leaving the zeros in place.
- **Executors:** non-core cells and regenerators drawn from the same soups, as in `offset_intruder.py`.
- **Encounters:** 4,096 traced encounters per variant, under both tars.

**Outcomes per encounter:**
- the intruder's pointer enters the host;
- it halts on a host byte;
- the host body is intact afterwards;
- the host's copy reaches the intruder's half.

**Prediction A1:** under lethal tar, detoxified hosts stay intact less often than original or sham hosts, in both lengths. Under benign tar, no difference beyond ±0.02.

### A2. Causal test in populations (Modal, at most $30)

**Setup.**
- Start from the 10 final lethal L = 32 mix50 soups.
- Two arms per soup: *detoxify every transmitter's payload*, or *sham*. New seeds.
- Run 100,000 steps with snapshots as in E.

**Prediction A2:**
- The transmitter share in the detox arm falls below the sham arm by ≥ 0.10 at some snapshot in ≥ 7 of 10 soups.
- Payload zeros climb back toward the sham level over the run, which would be selection seen directly.

**Before spending.** I will run 1,000 steps locally, review the code with an independent agent, and estimate the cost.

### A3. A small model, as theory

If an intruder enters at a random point and runs straight ahead m bytes, the chance it halts on a payload zero is 1 − (1 − z)^m. Fit z from A0 and m from the traces. This predicts A1's protection gap, and A1 tests it. It is a model, not a theorem, and the paper should say so.

---

## B. Genealogy: descent or displacement

**The gap.** Reviewers asked for lineage tags. The title claims "precede"; it cannot say "give rise to" without this.

**Design.** An exact record of copying, with no shader change.
- Every step, the host draws pairs (`Soup.read_pairs`) and we read the soup before and after (`read_soup`).
- For each cell after the step, its **source** is whichever pre-step tape it best matches: itself or its partner, with the similarity taken at the best cyclic shift.
- Two arrays are carried forward each step:
  - **founder labels** for K start times (label ← label of source), started at t_open (the first time the open replicator holds most cells) and every 500 steps after it;
  - the **minimum similarity along the path** (min ← min(min of source, similarity)).
- The soup is saved at every label start time. The founder of any cell at any start time is then an actual genotype we can read: we will see the ancestor of the closer at 500-step resolution.
- Cost: one readback and an O(N · L²) comparison per step. That is about 20 ms per step at L = 16 in numpy, so 150,000 steps take about 50 minutes per world.

**Validation before any paid run.** Local runs, plus an independent code review.
1. **Unit tests** on built encounters: exact copy, shifted copy, point mutation, chimera, untouched. Each must give the right source and similarity.
2. **Ground truth.** The invasion world at L = 16: the evolved closer seeded at 1% into a pusher world, with labels from step 0. At 20,000 steps, ≥ 99% of confined heritable cells must trace to a seeded cell, and ≥ 95% with path similarity ≥ 0.75.
3. **Negative control:** the same with the pair list shuffled. Tracing must fall to chance.

**Main runs (Modal).**
- **Benign tar, zero registers, L = 16:** 20 new seeds, run to 150,000 steps. By that point about 97% of these worlds have closed (Extended Data Fig. 10g).
- **Contrast: lethal tar, L = 16:** 10 seeds. Replicators there are born closed, so the tracing should *not* lead through a pusher. This is a positive control for independent origin.
- **L = 32 (Stage K):** only if a GPU version of the source rule is built and matches the numpy one bit for bit on 2,000 steps. In numpy, 10⁶ steps would take about 11 hours per world.
- Estimated cost: L = 16 about $20; L = 32 about $50 with the GPU rule.

**Pre-registered decision rule**, on confined heritable cells at the closure snapshot (the paper's criterion):
- **Descent:** ≥ 90% have a founder of the pusher class (Hamming ≤ L/4 to the modal pusher) at t_open, with path similarity ≥ 0.75.
- **Descent with recombination:** a pusher founder, but path similarity < 0.75. Report which bytes came from where.
- **Independent origin:** most founders are not of the pusher class.
- **Chance level:** the pusher share at t_open, reported next to the result.
- **My honest prior is "descent with recombination".** The closers (`ad e3 21 e3 21 c0 ad c0`, `04 5e ed b0`) are many mutations away from the pusher, so point mutations alone are unlikely.

**What the figure would show:** the ancestral genotype of the winning closer, every 500 steps, from the pusher to the closer, with the bytes that changed marked. Plus a spatial map of the lineage's founder inside a pusher patch.

---

## Order of work, about 9 hours

| Time | Work | Cost |
|---|---|---|
| 0:00–0:30 | Pre-register A0, A1, A2 and B in REVISION_PREREG.md, with timestamps. Inventory the data. | $0 |
| 0:30–1:30 | A0, all five predictions. **Gate A.** | $0 |
| 0:30–3:30 | B: recorder, unit tests, ground-truth and shuffled-pair validation, independent code review. Runs alongside A, with absolute paths and no shared cwd. | $0 |
| 1:30–3:00 | A1, if the gate passes. | $0 |
| 3:30 | Launch B on Modal (L = 16 benign and lethal). | ≤ $30 |
| 3:30–5:00 | A2: local smoke test, review, launch. Optionally the GPU source rule for L = 32. | ≤ $30 |
| 5:00–5:30 | Launch B at L = 32, only if the GPU rule matches numpy exactly. | ≤ $50 |
| 5:00–8:00 | Analysis as results land. Draft figures through figcheck and an independent critic. Outcomes appended to the pre-registration. | $0 |
| 8:00 | Morning report: what held, what failed, what it means for the paper. | |

**Rules for the night:**
- Never wait idle: while Modal runs, analyse or build the next piece.
- A run that fails gets one fix and one relaunch, then I write it down and move on.
- No result goes into the manuscript tonight. Findings go into a results note and a draft figure, and you decide what enters the paper.
- Every number comes from a generated table. Pre-register before computing. Report failed predictions plainly.

**If both A and B finish early, in this order:**
1. The real 8080 with alternate opcodes.
2. The random-register worlds through the B recorder: do the self-initialising closers descend from anything?
3. Long lethal runs, to see whether the payload keeps evolving.

## What I need from you tonight

1. **Approve the spend:** up to $140 tonight, only after the internal review and tests above pass.
2. **Set the priority if you disagree.** As written, A's free tests run first because they take an hour and tell us whether to go big on A. B's build starts at the same time either way.
3. **Choose the title** (still open from this morning).
