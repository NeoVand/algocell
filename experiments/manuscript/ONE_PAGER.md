# How life gets started, watched inside a computer

*A one-page explanation for a curious high-school student. Everything here comes from experiments we ran; where we say
what it might mean for real chemistry, we say so.*

## What did we find that nobody knew before?

Take twenty thousand tiny computer programs made of random bytes, lay them out on a grid, and let neighbours run each
other's code, over and over, with no goal, no scoring, nothing telling them what to do. Within a few hundred rounds,
the same thing happens in almost every world: a program appears that copies itself. It is two bytes long. Read as an
instruction it says *load this number into a register, then write the register into memory*, and the number it loads
is itself. It copies itself by accident of being executed: the instruction's own bytes are what it writes. It has no
loop, no "copy" command, and no idea where it ends. The processor runs straight out of it into the neighbouring
program and keeps going.

That last fact matters. Because it runs into its neighbour, what its children look like depends on the neighbour. We
measured it: of the eight bits that describe a child, about seven are decided by the neighbour, not by the parent. The
first living thing in these worlds is not an individual. It is something more like a colony that works best among its
own copies.

Then, in world after world, evolution does the same thing next. A descendant appears that carries one extra instruction,
a jump that sends execution back inside its own code. From then on the processor never leaves the organism, every
neighbour becomes an exact copy, and the children carry zero bits from the neighbourhood. We proved a theorem about why:
anything that writes fewer bytes than it reads cannot finish a copy of itself in one pass, so it must either run on into
its neighbour or come back on itself. A loop is the only way to become independent.

## How does this change the way we think about the origin of life?

Three ways. First, the first living thing was simple, not complex. In our worlds it is the simplest object of its size
that exists; by a standard complexity measure it scores the same as the sterile junk around it, and lower than the random
programs it replaced. If you look for life by looking for complexity, you look in the wrong place at the moment that
matters most. Second, being an individual is a later invention than reproducing. Life began open and dependent, and
independence had to evolve. Third, independence is not a wall or a membrane. It is a property of control, of where the
process goes, and we can measure it as a number of bits.

## Why is this significant?

Origin-of-life theories agree that at some point a pattern that merely spread became a thing with a self, whose future
depended on itself rather than on its surroundings. Nobody had been able to watch that step, repeat it, and take it
apart. Now there is a place where it happens thousands of times, with the dials exposed. We also found what controls it:
whether the machine has an instruction that writes its own content decides whether life begins open; whether the junk
that forms first is harmless or deadly decides whether the open phase ends in independence, in extinction, or never.

## What new questions does it raise, and what new tools does it give?

Questions: Does the open beginning survive only because neighbours on the grid are copies of itself? What happens if the
junk kills? Does chemistry's template copying, where a strand writes its own complement, behave like our two-byte word,
with sloppy, environment-dependent first copiers and by-products deciding their fate? Would a life detector sent to
another planet miss the first living things because they are too simple?

Tools: the **culture test**, which lifts a candidate out, gives it fresh random surroundings and asks whether its copies
also copy, and the **inflow measurement**, which counts how many bits of a child belong to the surroundings. Together
they turn "is it alive?" and "is it an individual?" from arguments into measurements.

## How does it change the way we look at origins?

Instead of asking which complex thing came first, ask when a spreading pattern stopped listening to its environment,
and what it took. In our worlds the answer was one instruction. In chemistry it is a prediction waiting for a test.
