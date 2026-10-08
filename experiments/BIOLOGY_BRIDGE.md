# Can the theory explain or predict anything in real origins-of-life and evolutionary biology? (2026-10-08)

Asked by the user: with an origins-of-life chemist's, molecular biologist's or astrobiologist's hat on, does the theory
(open literals first; closure as a control cycle; the window set by tar lethality; detection fooled by sterile order;
three load-bearing capabilities; tilings and the error threshold; heredity without clones) explain something not yet
explained, and is that worth pursuing? Below: what transfers, seven candidate bridges ranked by testability, and a
recommendation. Honest framing first.

## 0. What transfers and what does not

The theory is architectural. It applies to any system in which (i) a copying process runs on into adjacent material
by default, with no intrinsic boundary; (ii) there is a literal write channel, an operation whose output is its own
content; (iii) the process writes less than it reads per unit of activity. From these three it derives: the smallest
replicators are open (their process depends on the material around them); closure requires a cycle; the open phase is
a window whose length is set by whether the sterile by-products of the same chemistry stop the process; before the
first replicator there is abundant order without heredity, which complexity-based detectors cannot tell from life.
What does not transfer: bytes, pointers, instruction budgets, the ring. Every biological statement below is a
prediction or a reframing, not a result; the paper must say so.

## 1. Candidate bridges, ranked by how testable they are now

### A. Complexity-based biosignatures cannot see the beginning of life (testable on our data this week)

Claim. At the origin, the living thing is the lowest-complexity object the chemistry can make that copies (a two-byte
word repeated), and the sterile order made by the same chemistry (the zero flood, the return-address smears) has nearly
the same complexity and higher abundance. Any detector built from complexity, compressibility or complexity × abundance
therefore fires on the tar before life, and cannot separate the two; only an intervention that measures heredity can.
We have shown this for compression (high-order entropy, AUC 0.07–0.34 against the culture test at L ≥ 25; fires at the
first sample in the benign-tar BFF worlds and goes silent during the organism's reign).

Why it matters. Assembly theory (Sharma et al., Nature 2023; Marshall et al., Nature Comms 2021) proposes copy number ×
assembly index as a universal signature of selection, with "assembly index ≥ 15 only from life" for molecules. By
construction the first replicator in our worlds has an assembly index of about five (build `01 c5`, double three times)
and the tar about four; both are abundant. The prediction is that the assembly measure A would rank tar and the first
replicator together and above the random soup, and that complexity thresholds detect evolved life, not its beginning.
This is a falsifiable statement about a published Nature claim, with a ground truth (the culture test) that the
chemistry does not have.

Test. Compute a string assembly index (shortest construction from single bytes with reuse of built substrings; exact for
strings of 16–64 bytes by search, or the standard bound) for every class in the soup snapshots, and A over time, against
the culture test, for Z80 and BFF worlds. Cost: a day; data in hand. Risk: assembly theorists will answer that AT
describes selected objects, not first replicators; our scope is detection at the transition, and the answer is still
new. Astrobiology implication: planetary chemistry that produces abundant low-complexity order (asphalt) will look like
the first replicators to any complexity measure; detection at the origin needs a heredity test or a context-independence
test (bridge F).

### B. Individuality as zero information inflow, measured in the field's own currency (testable on our data)

Lemma 7 says closure is I(offspring; environment | parent) = 0. Krakauer et al. (2020) define individuality by the
information a system's past carries about its future relative to what its environment carries, and classify
"organismal", "colonial" and "environmentally determined" individuality. Our soups are the first system in which that
quantity can be computed through the transition: the first replicator should score as environmentally determined (its
offspring depend on the partner), the closed successor as organismal (zero inflow), with the jump at the closure event.
Test: estimate I(o; e | x) from the partner tests (offspring bytes against partner bytes, given the parent) for first
and final replicators across the 80 Stage G worlds and the BFF cells. Cost: a day. Value: turns "individuality made
measurable" from a phrase into a number in a published framework, and predicts that the same measure jumps at every
origin of a new replicating unit.

### C. The order of events in de novo replicator emergence (checkable in the literature)

Prediction for any experimental system where replicators arise from non-replicating material: first, sterile
amplifiable order (products that are made abundantly by the machinery but do not inherit: abortive transcripts, primer
dimers, polymerase-only products); then minimal, low-complexity, periodic replicators whose yield depends on the
sequences around them; then autonomy. Candidate systems: Jain et al. (Science 2020, novel RNA replicons emerging with
T7 RNA polymerase from non-replicating templates), the Ichihashi/Mizuuchi long-term RNA replicator evolution (parasites
first, then host–parasite networks; Mizuuchi et al. Nature Comms 2022 is in the library), Lehman's cooperative
networks (Vaidya et al. Nature 2012), and DNA tile replication (Wang et al. Nature 2011). What to check: are the first
replicons small and repetitive; do they depend on specific flanking sequences or partners (open); does a later variant
lose that dependence (closure)? If the pattern holds, the discussion can say the order was predicted; if not, the theory
has a boundary and we say where.

### D. Templates before closure: a mechanism for an old debate (a position, not data)

Metabolism-first (autocatalytic sets, organisational closure: Kauffman; Fontana & Buss) versus genetics-first
(templates). Our result says the cheapest replication is a template-like literal write, it comes first and it is open;
organisational closure is a later, rarer acquisition driven by selection for fidelity, and we give the reason: a closed
replicator needs a cycle, which is strictly larger and exponentially rarer in a random soup (Theorem 6, Proposition 4).
This is a mechanism for why closure follows rather than precedes replication at the origin. It explains the shape of the
debate rather than a dataset, but it is the kind of theoretical claim origins-of-life reviews ask for.

### E. Tar lethality as a habitability parameter (a new prediction for chemistry; untested)

The classification says an open beginning ends in closure only if the sterile by-products do not stop the open
replicator's process; inhibitory by-products end it in extinction. In chemistry: whether side products are inert (they
precipitate, or are skipped by the copying process) or inhibitory (they stall or terminate copying) decides whether
templating replicators persist long enough for autonomous replication to evolve. Prediction: prebiotic chemistries
differ in the lethality of their tar, and the ones compatible with the transition are those whose tar is inert to the
copying process. A screen of nonenzymatic replication systems for inert versus inhibitory by-products would test the
window model. Benner's asphalt problem is usually posed as a yield problem; this reframes it as a dynamics problem.
Caution: stalling after a mismatch (Rajamani et al. 2010) raises fidelity in template copying; our lethal tar is the
halting of the replicator's own process by material it runs into, a different effect, and the discussion must not
conflate them.

### F. The Darwinian threshold as a budget threshold (a quantitative reframing)

Woese's progenote: communal early evolution, horizontal transfer dominant, lineages ill defined; the Darwinian
threshold when vertical descent takes over. Our sub-clonal regime: when one encounter cannot copy a whole genome, heredity
is carried by the population with no clone (6 of 10 worlds at 64 steps and L = 100), and lineages appear once the
per-encounter budget exceeds the genome. Prediction: the threshold is crossed when per-event copying capacity
(processivity × event duration) exceeds genome length; below it, genomes are mosaics assembled from partial copies.
Experimental handle: polymerase ribozymes of increasing processivity (Cojocaru & Unrau 2021; Gianni et al. 2026, a
45-nt ribozyme that copies in pieces): clonal lineages should appear only when a single copying event spans the genome.

### G. Independence as the first act of evolution at every transition (a general claim)

Lemma 7 reads: selection for fidelity is selection against listening to the environment. If the architecture is general,
then at every origin of a new replicating unit (genes, cells, multicellular individuals, perhaps cultural replicators)
the first form is open and context-dependent and the evolution of individuality is the closing of information inflow;
horizontal and context-dependent inheritance should fall across each transition. This matches the known decline of
horizontal transfer with the rise of vertical inheritance but adds no new data; it belongs in the discussion as the
theory's widest reading, with its limits stated.

## 2. Recommendation

Do A and B now: both use data already on disk, cost about a day each, and connect the paper to two debates its referees
live in (biosignatures and individuality). Write C as a literature check for the discussion, with Jain 2020 and
Mizuuchi 2022 read closely. State D, E, F and G as predictions with explicit tests, in the measured register of the
exemplars ("if the architecture is general, then …"). Do not claim to have solved a chemistry problem; claim that the
theory changes what detection and individuality measures should look for at the origin, and that it predicts the order
of events in de novo replicator experiments.

## 3. What a genuine solved problem would look like

A simulation cannot settle a chemistry question. The nearest genuine contribution is to the theory of detection (A),
to the quantitative definition of individuality (B), and to the ordering problem of templates versus closure (D). If A
shows that an assembly-type measure fires on tar before life with the culture test as ground truth, that is a concrete,
falsifiable statement about a published Nature claim with consequences for astrobiological detection, and it is the
kind of result that gets a simulation paper read by people who do not usually read simulation papers.
