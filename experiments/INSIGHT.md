# The insight, stated so that it can be used (2026-10-08, after the user's reading of the one-pager)

The user's objection: the paper and the one-pager swim in bytes and never say the deep, usable thing. This document
states the explanation first, in the sense of a good explanation (hard to vary, with reach, answering questions we did
not ask), then what it predicts, then how a chemist, an evolutionary biologist or an astrobiologist would use it
tomorrow, then the experiments, ours and theirs, that would make it ironclad or kill it. Every quantitative claim is
from a generated table (NUMBERS_INDEX.md); every claim about chemistry is a prediction.

## 1. The explanation

**Any process that copies by writing must either run beyond what it has copied or come back to it.** A copier that
writes fewer symbols than it reads cannot finish a copy of itself in one pass over itself. So its first pass ends in one
of exactly two ways: execution leaves the copier and continues in whatever lies beyond it, or it returns to a place it has
already been. There is no third way, and nothing about this depends on the Z80: it is counting.

From this one fact the order of events at the beginning of life follows, and it is the order we observed in thousands of
worlds and in a second, unrelated machine:

1. **The first replicator is the smallest thing that writes itself, and it is open.** A channel in which content is its
   own instruction (a literal: the bytes an instruction carries are the bytes it writes) produces a self-copier of two
   symbols. Nothing simpler can copy; nothing this simple can contain a loop. So the first replicator leaves itself and
   runs on into its surroundings, which decide most of what its offspring look like. Measured: seven of the eight bits
   that describe a child belong to the neighbour.
2. **Individuality is a return, not a wall.** The step that makes an organism's future depend on itself is a cycle in
   control: execution comes back inside. It is not a membrane, not a boundary in space, not protection; the closed
   organism's bytes are as exposed as anyone's. Measured: zero bits from the neighbour, in sixty-three of sixty-seven
   worlds, with four different instructions inventing the same return.
3. **Order precedes life and mimics it.** Before anything copies, the same chemistry writes sterile order (zeros from
   empty registers). It is as simple as the first replicator (same minimal assembly index), more abundant, and it
   trips every detector that reads structure: compression and assembly both fire on it. Only intervention separates
   the two: lift the thing out, give it fresh surroundings, see whether its copies copy.
4. **Two properties of a substrate decide the beginning.** Whether it has a literal channel decides whether life begins
   open (BFF as published has none and is born closed; add one instruction and it begins open). Whether the sterile order
   halts the open copier's process decides how the open phase ends: in closure if a copyable return exists, for ever if
   none does, in extinction if the order is lethal.
5. **The boundary of the rule is itself a prediction.** When a copier writes at least as much as it reads, it can close
   by saturation, overwriting everything it touches, and needs no return; BFF's literal pusher under a wrapping pointer
   does exactly this (0.04 bits of inflow with its pointer entering the partner in every encounter). Overhead, meaning
   instructions longer than their output, is what forces loops.

Why this is a good explanation in Deutsch's sense: it has no adjustable part (the dichotomy is arithmetic; the literal
channel is a property you can read off a substrate; the lethality of by-products is measurable); it reaches beyond its
origin (any sequential copying process with overhead, in silicon or chemistry); and it answered questions we had not
asked: why the first replicators are sloppy and only partly heritable, why individuality can precede compartments, why
complexity-based biosignatures must fail at the origin, why spatial structure should matter to the open phase, and why
the first evolved innovations should be about control rather than function.

## 2. What it says about which beginnings are plausible

- The most probable beginning is the least complex object the chemistry can make that writes itself: a short repeat in a
  literal channel. Beginnings that require a copying machine, a code or a compartment are not where life starts; they
  are what life builds afterwards.
- Beginnings are inevitable where three things coexist: a literal channel, by-products that do not stop the copying
  process, and enough time. Where the by-products are inhibitory the beginning is a window that closes.
- Beginnings are communal. The open copier's offspring are partly the environment's; lineages, in the sense of vertical
  descent, do not exist until the return evolves. Woese's progenote and the Darwinian threshold are not a special early
  biology; they are what an open copier is.
- Beginnings do not need space. A pre-registered well-mixed control found the open phase exactly as heritable among
  strangers as among kin, and closure later rather than sooner. The mechanism of the kin idea is real (the first replicator
  meets its own kind in 57% of encounters on the lattice and 11% when mixed; kin encounters damage it in 0–2.5% of cases,
  strangers in 31–34%), but the open phase survives strangers all the same. What space seems to change is how reliably
  closure arrives (post hoc, to test), not whether.

## 3. The insight as a tool

**For a chemist working on template replication.** Base pairing is a literal channel: the template specifies its
complement by physical fit, with no interpreter. Expect the first replicators to be the shortest self-templating repeats,
to depend on their surroundings, and to produce mostly partial copies. Therefore: (i) stop looking for a clean
self-replicating species and start measuring *heritable propagation among sloppy copies*: transfer products into fresh
material and ask whether the products of the products carry the pattern (the culture test); (ii) measure openness
directly as *information inflow*, the entropy of the product distribution across varied contexts, which falls to zero
when a replicator becomes an individual; (iii) treat by-products as a design variable and screen chemistries for whether
their tar is inert or inhibitory to the copying process, because that, not yield, decides whether an open phase lasts
long enough for control to evolve; (iv) do not require surfaces or compartments for the open phase: in our control it is as heritable when well mixed; but
expect spatial structure to make the arrival of individuality reliable rather than hit-or-miss.

**For an evolutionary biologist.** The first selected innovations after replication should be about *control*, where
copying starts, stops and returns (circular templates, rolling-circle replication, terminators, primers), not about
new functions or codes. In laboratory evolution of RNA replicators, the earliest fixed variants should be of this kind.
Vertical lineages should appear exactly when a single copying event spans the genome: processivity is the chemical
write ratio, and below it heredity is carried by the population without clones, as our 64-step soups show.

**For an astrobiologist.** Expect abundant low-complexity order before and around the first replicators, and expect
every structure-reading biosignature (compression, assembly, disequilibrium of composition) to fire on it. Two
consequences: look for *order without heredity* as a precursor signature, which is more abundant than early life; and
design detection as intervention, not inspection: perturb a sample with fresh substrate and look for propagation that
depends on the inoculum. A first living thing may be the simplest ordered object in the sample, not the most complex.

**For a theorist.** The write ratio sorts substrates: below one, closure needs a cycle and the first replicator is open;
at or above one, closure by saturation is possible and an open phase may never exist. This gives a one-number
classification of candidate origin chemistries and of artificial-life substrates, and it predicts which of them will show
the open-then-closed order and which will not.

## 4. Experiments

**Ours, cheap, decisive (pre-registered where launched).** Stage H, well-mixed pairing: does the open phase need kin? (Done: no; the order of events is independent of space.)
Stage I, lethal zeros: does tar lethality shorten the window in the first machine as in the second? A write-ratio sweep:
machines whose literal instruction writes one, two or four bytes per cell executed; the theorem predicts the open phase
disappears at ratio one. A processivity sweep at fixed genome length (budget per encounter), predicting the step at which
lineages appear. A definition grid over our own thresholds, to show the order of events does not depend on them.

**Theirs, proposed.** (i) In vitro evolution of RNA replicators (Mizuuchi–Ichihashi, Jain): score the earliest fixed
mutations as control versus function; measure product entropy across contexts before and after. (ii) Nonenzymatic
template copying (Szostak-type): add inert versus inhibitory by-products at matched concentration and measure how long
template propagation persists, the window prediction. (iii) Surface-bound replication (mineral, droplet, pore): compare
propagation at patch interiors and frontiers. (iv) Planetary analogue samples: apply the culture-test protocol alongside
assembly and compression measures, and report what each detects on known sterile order (asphalt, tholins).

## 5. What would kill it

A substrate with a literal channel whose first replicator is closed; an open first replicator with zero inflow; a well-mixed soup indistinguishable from the lattice at every measure (this happened for the open phase: the spatial
reading is dead, the order of events stands); a lethal-tar
machine in which closure arrives as readily as with benign tar; a chemistry in which the first heritable products are
complex. Each is a measurement, not an argument.
