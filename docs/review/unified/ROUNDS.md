# Unified application review

The design was committed as `d767d64` before UI changes. The complete initial
inventory, user paths and 55-state baseline were committed as `4e6088e` against
frozen source `6484a24`. Additional baseline captures, where listed, use that same
frozen source. Work stays on `feat/app-unified`; the reviewer server on 5175 is
outside the capture/test infrastructure.

This is an instrument review, not a claim of a human A/B study. The comparisons
with Figma, Linear, Blender, Ableton and Observable mean spatial consistency,
visual hierarchy, stable workspace, uninterrupted transport and explicitly
scoped evidence. Clicks and pointer distances are repeatable task-path proxies.

## Pre-build critique

The evolution screenshots show the underlying lattice remains distinctive and
fast, but its interface has three competing corners plus independent inspector
and help windows. Colours, scientific evidence and tools are already valuable;
their placement and state transitions make them feel disconnected.

The chosen shell places World, Measure, Experiments, Guide and About in one
right sidebar; direct tools align with its canvas edge; Look remains over the
field's header and Run occupies a fixed bottom strip. The old chart widget
becomes the Charts section in Measure. No new rendering identity is introduced.

Static review caught several issues before the first visual round:

- A live core change must update the next-world draft's unchanged live fields
  without overwriting the user's staged seed or geometry.
- A preset selected for preview must not label observations from the current
  applied world, and a started preset must no longer show as unapplied.
- An inspector refresh must never select its sidebar tab again; users must be
  free to read Guide while keeping that target and its evidence.
- Assay results must stay mounted across sidebar tabs and describe their sampled
  tape, even as the actual population continues.
- The measurement colour keys must survive removal of the old observation box.
- Form limits must match the engine's supported tape, ring and lattice ranges.
- The former automatic 55-second product pause conflicts with a continuous
  instrument. Test and benchmark workloads remain bounded independently; a
  hidden document suspends work while preserving the user's playback choice.

## Review rounds

Round-by-round captures, critique, corrections and measured results follow below
as each built shell is exercised. Raw paths and state inventories are retained
in each round's `metrics.json`.

### Round 1 — one workspace, then remove friction inside it

The frozen first shell completed all twelve paths and 135 successful screenshots
across the three requested sizes. No horizontal overflow, prohibited visible
wording, incidental pause, incidental reset or engine replacement was observed.
Totals changed from 60 to 57 clicks, 26,580 to 18,597 pointer pixels (−30.0%),
26 to 19 summed surfaces, ten pauses to one deliberate pause, and two resets to
one explicitly requested preset start. The full table is in `TASKS.md`.

**Critique.** Compared with the evolution images, the stable canvas/sidebar edge
now provides the spatial continuity expected of Figma or Blender. World versus
Next world makes the destructive boundary visible. Transport stays present as
in Ableton; measurement evidence retains its source as it should in Observable.
The consistent copper selection and 32px controls have the restraint of Linear.
These comparisons are design criteria, not claims of equivalent product scope.

The review still found specific defects:

- Export required visiting Measure and World. Its task rose from three to five
  clicks despite the shared sidebar. App-level image, video and checkpoint work
  belongs together; result-specific JSON remains beside the result it describes.
- World used generous divider/intro spacing while Seed fell near the bottom at
  900px height. Dense configuration needs a clearer, shorter visual rhythm.
- The parent L32 tape preview used four widely spaced columns and eight rows,
  pushing the encounter's main action below the fold. Eight filled columns match
  the inspector's byte grid and halve its height without shrinking controls.
- Status text was too small. An 11px minimum makes applied-world context useful.
- The frequent Look path grew from 1,002 to 1,492px: view and appearance sat at
  the left of the header, while zoom/fit occupied the opposite end. They should
  be adjacent. Long encounter results also scrolled their context out of view.
- Guide→transport and live-control paths increased slightly. Their stable global
  transport location remains a deliberate tradeoff; Space and the visible speed
  control keep repeated run work predictable. The regressions remain in the table.

**Corrections for round 2.** Consolidated PNG/WebM/checkpoint/import under
Measure→Export, with the persistent footer leading there. Tightened World
spacing, raised status typography, and changed the sampled tape preview to eight
columns with 32px cells and 4px gaps. Svelte checking, focused lint, TypeScript
checking and a fresh production build passed. Header grouping and sticky assay
context are the next round's focused review items.

Harness corrections (waiting for asynchronous confinement selection, exact
palette labels and genuinely different staged tape lengths) were rerun against
the same immutable source snapshot. Superseded harness-error images were removed
from the successful-state manifest. No failed path was assigned guessed metrics.

### Round 2 — make nearby actions and long evidence work together

The second immutable snapshot completed all twelve paths and 135 screenshots,
with no capture errors, horizontal overflow or prohibited visible wording.
Totals were 55 clicks, 17,764 pointer pixels (−33.2% versus baseline), 19 summed
surfaces, one requested pause and one explicit preset reset. The combined Export
section now supports a direct footer→PNG→checkpoint path: three clicks and
1,482px, compared with five clicks/2,189px in round 1 and three/2,167px before.
The alternative Measure→Export path is retained separately in raw results.

**Critique.** World now exposes the beginning of Next world more clearly at
900px. The compact tape preview brings Run culture test into view without
reducing its hit target. Checkpoints and image/video controls read as a single
destination, and the higher status type size is easier to read. The source
caption correctly distinguishes selection time from assay launch time.

The long-result screenshot still began with a clipped Culture tab: the primary
sidebar tabs remained stable, but the local encounter identity disappeared.
First-class editors retain selection context during long inspection; this bench
needs the same treatment. The Look controls remained unnecessarily separated.
Static narrow-screen inspection also found an inherited mobile rule hiding the
three chart tabs, and a child-initiated History action could leave Export selected.

The dedicated continuity suite exposed two real input problems beyond the task
paths. A tape-length blur handler could change the ring field while the user was
typing into it, producing 4848 instead of 48. Modified Save/Import shortcuts
were rejected by the editable-field guard. These are recorded as defects,
not attributed to the task harness.

**Corrections for round 3.** Grouped zoom/fit with view/Look; made the encounter's
return/title/source row sticky; restored mobile chart tabs; synchronized
child-initiated tool navigation without allowing periodic probe refreshes to
pull users away from Guide. Moved linked ring adjustment to the tape input event
and handled Cmd/Ctrl+S/O before the plain-hotkey editable guard. The unified
export heading now says Images & video and points to the checkpoint controls
below. Added focused browser regressions for these interactions, including long
results and 390px chart navigation. Static Svelte/TypeScript/lint checks passed;
the complete browser suite and final captured round follow.

The complete 77-test run then passed, including the 22 preset contracts,
desktop/mobile tools and all continuity checks. Its 1.8-second control observer
recorded zero nonnumeric mutations, 13 numeric changes and stable control-node
identities. This run also verified the fixes for mobile rail overlap and the
embedded History panel's inherited positioning offsets. Panel changes now use
a 160ms opacity transition with reduced-motion support; canvas keyboard focus
uses the same restrained one-pixel copper treatment as the controls.

A final specification audit before the last capture found three remaining
clarity issues: Guide forgot its selected subsection after About; core changes
did not disclose changed reset/dial/ablation defaults; and the World-side start
summary omitted a staged preset's core and budgets. Guide now remembers its
section, a compact inline notice names only the defaults actually changed, and
both start actions describe the complete target condition. Three regression
tests cover these corrections. The README now describes the actual shell and
continuous playback behavior.

### Round 3 — verify the complete shell, then inspect the pixels

The committed `3f54bcb` shell completed all 80 browser tests, followed by twelve
successful task paths and 147 state captures from an immutable source snapshot.
There were no capture errors, horizontal overflow or prohibited visible wording.
Task totals were 55 clicks, 17,232 pointer pixels (−35.2%), 19 summed surfaces,
one requested pause and one explicit preset reset. No incidental pause/reset or
engine replacement occurred. Grouping Look controls reduced that frequent path
from 1,492px in round 2 to 903px, below the original 1,002px. The complete staged
preset summary added 90px of travel while retaining three clicks; the visible
replacement contract warrants that cost.

**Critique.** The encounter's cell identity now remains visible through long
results, and the full target world is clear at both start actions. Guide stays
in the reader's chosen section; core swaps explain changed execution defaults
without obscuring the lattice. The hierarchy, spacing and tool placement remain
consistent at the requested desktop sizes. The earlier mobile overlap and
hidden chart-tab defects are resolved.

Pixel review still found three defects that the functional assertions did not
catch. A checkbox was checked in the DOM while its checked paint was hidden by
the parent's background shorthand overriding the forms styling. This affected
Mutation off, Random registers and Grid lines. The initial Measure empty state
repeated its instruction in parent and child blocks. Finally, the mobile canvas
was 310px wide but its first lattice occupied only about 75px: initial geometry
was measured before the collapsed sidebar's layout had settled.

The 1280px footer also hid chemistry too early despite having room for it.
Move that visibility breakpoint to the narrow layout so every required status
item remains present at all three specified desktop sizes.

**Corrections for round 4.** Restore visible copper checkbox states, retain one
teaching block with its Probe action, and settle the initial responsive layout
before measuring the world and initial camera. Later sidebar resizing must
continue to preserve the world and camera. Add regressions for these visual and
initialization conditions and required status visibility, then repeat the complete task/state atlas against a
new committed source. Round 3 remains intact as evidence of the review findings.

### Round 4 — close the visual defects and freeze the instrument

The final shell is committed as `192a926`. All twelve paths pass with the same
55 clicks, 17,232 pointer pixels and 19 summed surfaces as round 3. Its 147
captures have no errors, horizontal overflow or prohibited visible wording.
The complete archive contains 114 before images and 564 after images across
four rounds. Every measured path retains the engine; the only remaining pause
is the requested transport pause and the only reset is the explicit preset start.

**Final visual review.** The initial 390px world now fills the 310px canvas before
the user presses Fit. Mutation off visibly displays its copper check mark,
matching the other controls. The empty Measure section has one explanation and
one Probe action. At 1280px the chemistry status remains visible alongside core,
seed and geometry. These were checked in the final images, not inferred only
from passing DOM assertions. The lattice keeps its original rendering and colour
identity; stable edges, restrained copper and consistent controls now give its
configuration and evidence one coherent frame. No further correction was needed.

The complete 83-test browser run passed after the runtime and visual fixes.
The last CSS-only chemistry breakpoint then passed three focused desktop
layout/visibility rechecks. Svelte checking reported zero errors and warnings;
lint and the production build passed. The final playback observer again recorded
zero nonnumeric mutations, 13 numeric changes and stable control identities.
Exact test names and validation scope are retained in `validation.json`.

The final before/after table for every task is in `TASKS.md`. It retains the
tradeoffs: first-visit Guide travel and the live-control path are a little longer,
the complete preset review takes more travel, and history adds one explicit
sidebar-section click. The most frequent Look and probe paths improve, and
transport, state and evidence stay predictable across all paths. These scripted
measurements and design reviews do not claim a human A/B study.

Final performance measurements, the original noisy result and its control/
confirmation runs are documented in `PERFORMANCE.md`. No third-party dependency
was added. No push or deployment is part of this work.
All owned test, capture and benchmark servers are closed; the reviewer server
on port 5175 remains running.
