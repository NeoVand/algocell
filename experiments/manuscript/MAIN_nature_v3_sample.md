# Open replicators evolve closure in a digital primordial soup

*Sample of the rewritten register, 2026-10-08: summary, introduction and the first results section, written to the
ladder in `REWRITE_PLAN.md` and the observations in `STYLE_NOTES.md`. Bracketed names mark references to be numbered;
every figure in `NUMBERS_INDEX.md`. For calibration with the user before the full rewrite.*

## Summary

Theories of the origin of life disagree about what came first but agree about what the transition was: a
self-propagating pattern became an individual, something whose persistence depends on its own organisation rather than
on its surroundings [Maturana & Varela; Rosen; Szathmáry; Krakauer et al.]. That step has never been watched. Soups of
random computer programs produce self-replicators without any selection [Agüera y Arcas et al.], but what their first
replicators are and what becomes of them is unknown. Here we show, in thousands of simulated worlds of random machine
code, that the first replicator is a two-byte literal that copies itself without a loop and is open: execution runs on
into whatever program it meets, so its success depends on its neighbours. It is replaced, convergently, by a descendant
carrying one control-flow cycle that keeps execution inside the organism and copies into every partner. We prove that
closure requires such a cycle and that an instruction writing its own operand forces life to begin open; in a second
machine, adding one such instruction switches the open phase on, and the lethality of the sterile order that precedes
life decides whether the phase ends in closure, extinction or permanence. The first replicators in any such substrate
are predicted to be environment-dependent and only partly heritable; evolution's first act is to make them independent.

## Main

What distinguishes a living thing from the chemistry around it has been answered in two ways. One tradition lists
properties, metabolism, compartment and heredity, and origin-of-life research has largely been organised around which
property came first [Gilbert; Wächtershäuser; Szostak]. The other locates life in a relation rather than a list: a
living system is one whose operations produce the organisation that produces them, so that its future depends on its
own past rather than on its environment [Maturana & Varela; Rosen; Fontana & Buss; Mossio & Moreno]. This second view
has recently been made quantitative, as the information an entity carries forward about itself relative to what its
environment carries for it [Krakauer et al.], and it coincides with the moment that population genetics identifies as
the beginning of Darwinian evolution proper, when vertical descent comes to dominate the communal exchange of an early
world [Woese; Nowak & Ohtsuki; Szathmáry]. Whatever the first replicating molecules were, this is the transition that
made them organisms. No experimental system has yet allowed it to be observed as it happens, repeated under
controlled variation, and taken apart.

Soups of random programs come close. When random byte strings for a simple machine are made to execute one another in
pairs, with no fitness function and no replacement, self-replicating programs arise and take over [Agüera y Arcas et
al.], as they do in related settings with engineered starting points [Ray; Adami; Fontana]. The phenomenon is robust
across languages and instruction sets, including the Z80 processor [Cicala et al.], and it reproduces the gross
signature of a transition to life: a sudden collapse of diversity and a rise in compressibility. What has not been
asked is what the first replicator is, why that program and not another, what happens to it after it has filled the
soup, and which properties of the machine decide these outcomes. Nor has heredity been measured directly; the
transition has been inferred from compression statistics of the whole soup, which, we will show, respond to a sterile
order that precedes life and fall silent when a one-byte organism takes over.

We treated such a soup as an experiment. Twenty thousand programs of a fixed length, written for the Z80, interact in
random pairs for a fixed number of instructions, and nothing is selected except what persists. We ran 3,560 worlds on
graphics processors across thirteen deletions of instruction families, program lengths from 3 to 100 bytes, execution
budgets, mutation rates and memory geometries, with confirmatory stages pre-registered with predictions and kill
criteria (Methods). Heredity is measured by intervention: a program is lifted out of the soup, executed against fresh
random partners, and its offspring are tested in turn, so that a pattern that merely spreads is not mistaken for one
that inherits. We then repeated the central experiment on a second, minimal machine with a controllable instruction
set, and proved what could be proved.

Here we show that life in these worlds begins in a fixed order. A sterile tar of self-written zero bytes forms first.
Then a two-byte word that loads itself into a register and pushes the register into memory becomes the first heritable
replicator in almost every world; it needs no loop and never reads its surroundings, but execution runs on from its
last byte into its partner, and its success depends on what it finds there. Within a few hundred thousand interactions
it is displaced, convergently and down to the byte, by a descendant that carries one control-flow cycle and copies into
every partner it meets. Deleting instruction families shows that three of them are load-bearing. A theorem shows that
closure requires a cycle whenever a machine writes fewer bytes than it executes, and that an instruction which writes
its own operand forces the first replicator to be open; a second machine confirms that adding such an instruction
switches the open phase on, and that the lethality of its tar decides how the phase ends.

### Sterile order precedes life

In every unablated world the soup first fills with order that does not inherit. At the moment the most common pattern
first occupies a tenth of the soup, between 24% and 34% of all bytes are zero, at every program length from 9 to 100
bytes (Extended Data Fig. 1; [results/stageB; results/stageG]). The zeros are self-written: push, call and reset
instructions executed with empty registers write zeros wherever the stack pointer points, and zero is itself an
instruction that does nothing, so the pattern spreads by being written and costs nothing to execute. Lifted out of the
soup and cultured against random partners, its carriers produce no copies that copy in turn (second-generation
heritability 0.00). We call this zero flood tar, after the asphalt that prebiotic chemistry produces from the same
reactions that produce sugars [Benner et al.]: order without heredity, made by the chemistry that will later make life.

The tar is what detectors see. The high-order entropy used to detect the transition to life in random-program soups
[Agüera y Arcas et al.] fires on the flood: in worlds of 25 to 81 bytes it ranks the soups that produce life below those
that do not (area under the curve 0.07–0.34 against the culture test; Methods), and in the second machine it fires at
the first sample, before any replicator exists, and falls to zero while a one-byte organism holds the entire soup. Only
the intervention separates order from heredity, and we use it as the ground truth throughout: a program is a
replicator when at least three quarters of its random partners become copies and those copies copy in turn (Methods).
