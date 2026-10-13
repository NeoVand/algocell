# Dedicated paper repository: inventory and migration plan

Status: **planning only**. Prepared for the coordinator on 2026-10-12 (America/Chicago). No files were moved or deleted, no repository was created, and no experiment, Modal function, upload, or publication was started. The only project file authored by this task is this plan.

## Recommendation

Create **`NeoVand/algocell-paper`** in a later, separately authorized migration. Make Algocell the real-time Svelte web application, Zilion the processor/machine library, and `algocell-paper` the frozen scientific pipelines, registrations, evidence, figures, manuscripts, and reproduction instructions. Keep all superseded work, with explicit archive status. Put bulk raw histories and large generated evidence in versioned Zenodo deposits, with downloadable, hash-verified manifests in the paper repository. Keep the existing Modal volume as an operational mirror, rather than the sole public accession.

Recommend a **fresh repository with precise provenance**, rather than rewritten extracted history: the current tracked experiments payload alone is about 1.55 GB, and the evidence contains original commit identities and frozen source hashes. A fresh scientific release can have a small checkout without importing every historical output blob; preserve the complete original Git history separately as described below. This is a recommendation, not an instruction to create it now.

## Scope and inventory method

The committed baseline is [`96e81124697c79f336ebc9d93df75121f98a6640`](https://github.com/NeoVand/algocell/commit/96e81124697c79f336ebc9d93df75121f98a6640). There are **375 commits reachable from HEAD touching `experiments/`**. The working tree is active: uncommitted v8/SI/reconciliation text and legend edits were present, and other manuscript work continued during this inventory. They are inventoried as working-tree material, never silently attributed to that commit. Freeze a new complete snapshot at migration time. Sizes below are a local inventory snapshot during 2026-10-12 20:08–20:12 CDT; Modal was sampled a few minutes earlier. A few concurrently edited text bytes can change after this plan.

The prescribed `/Users/neo/repos/algocell/experiments/.venv/bin/python` was used. The inventory did not import scientific pipelines to infer dependencies. It used:

1. `os.walk`/`stat` for every local file, including ignored `runs/`, `.venv`, literature, figure references, videos, and archives; `git ls-files -z` for tracked bytes. Logical sizes are sums of `st_size`; allocated sizes use `st_blocks * 512`. `.venv`, caches, and run data are excluded from Python source parsing, not from size accounting.
2. Source brackets from `MAIN_v7.md`, the working `MAIN_v8.md`, `SI.md`, `SI_v8.md`, and the registration documents. Resolve paths from `experiments/`; expand explicitly cited directories, and resolve abbreviated sibling filenames within a bracket. Read `audit_v7.csv`'s `source` and `evidence_source` fields, `AUDIT_v7.md`'s fingerprints, and the new `AUDIT_v7_RESOLUTION.md`. Code named by a registration is retained even if its filename sounds exploratory.
3. AST parsing of **321 Python files**, including archived/provenance Python sources; local `import`/`from` resolution, literal input/code paths, and transitive dependency closure. The initial combined graph had **1,929 import/static-path edges**. Also inspect shell dependencies, producer output roots, literal/glob input patterns, and hash/input/provenance manifests (27 manifest nodes with 5,502 recorded local dependency edges, including aliases/repetitions). Explicit SI table/map paths outside brackets are additional roots. Retain the producers and underlying inputs of cited tables; merely reading a produced CSV is insufficient. Directory citations and the old SI deliberately make this a conservative union of v7/v8/SI requirements.
4. Figure manifests: `out/v8/data/sources_v8.csv`, `hashes.csv`, earlier `out/v6/data/sources.csv`, frozen v6/v7 inputs, and `prepare_*`/`build_*` imports. Active v8 still depends on v6/v6.2/v7 preparation and the legacy `out/fig1.pdf`; an old version number does not by itself make an asset dispensable.
5. TypeScript import traversal starting at `scripts/export-sim.ts` and `src/lib/dev/z80difftest_ring.node.ts`, plus searches for app references to experiments. Eight TS files form the relevant closure (listed below).
6. A **read-only recursive metadata listing** through `modal.Volume.from_name('algocell-atlas-runs', create_if_missing=False).iterdir('/', recursive=True)`. This uses the client API, not a deployed function or volume-writing probe. The available volume list also contained unrelated `mindview-distill`, which was excluded. All experiment Modal wrappers refer to the one `algocell-atlas-runs` volume.

The final appendix classifies every top-level experiment item; package modules, manuscript/planning/test sources, every results cohort, every local run item, and every Modal prefix have explicit dispositions. For cohorts containing many files, named essential scopes and the remainder rule classify individual descendants without printing 64,000 raw filenames. Static reachability cannot prove that every branch executes. Retain uncertain sibling evidence as supporting provenance; confirm runtime inputs with manifests during migration before narrowing the release.

Classification keys throughout:

| Key | Classification | Disposition |
|---|---|---|
| E | **paper-essential**: cited by v7/v8, SI, a figure or registration, or a dependency/producer/verification input | Active paper tree; bulk bytes may live in Zenodo with a required manifest |
| S | **paper-supporting provenance**: registrations' operational records, audits, corrections, archived logs, cohort siblings not proven essential | `results/provenance/`, `registrations/`, or `archive/provenance/`; retained and indexed |
| A | **superseded or exploratory-only** with no established current dependency | `archive/`; retain code, evidence and original names |
| W | **web-app material** | Stays in Algocell; a frozen copy of a shared historical dependency may also be needed by the paper |

These labels describe paper relevance, not confirmation status. **Cited exploratory crystal/ladder/S2 follow-ups are E and must remain labelled exploratory/post hoc.** An operational diagnostic can be S even when its result is important provenance. OS metadata and caches are A/nonresearch; preserve their inventory, not a requirement to ship caches in the scientific release.

## Measured storage and release implications

Local `experiments/` contains **77,337 files, 29,638,189,745 bytes (29.64 GB / 27.60 GiB)**. Of these, **1,552,511,546 bytes** belong to tracked paths at inventory time. Local file counts include the virtual environment and ignored files; these are not Git object database sizes. Directory rows below overlap their descendants and must not be added twice.

| Local scope | Files | Logical bytes | Meaning |
|---|---:|---:|---|
| `runs/` | 64,220 | 26,838,077,366 | 26.84 GB / 25.00 GiB; entirely outside Git; 56 directories plus 218 root files |
| `results/` | 2,586 | 1,361,779,720 | 1.36 GB; includes 287.57 MB of S2 downloads and other archived artifacts |
| `manuscript/` | 1,693 | 1,091,387,770 | 1.09 GB, mostly figure outputs/reviews/video frames |
| `algocell_exp/` | 981 | 28,882,768 | Historical shaders plus vendored machines and research adapters |
| `algocell_exp/machines/` | 717 | 19,461,063 | 352 shader/manifest pairs, six specs, provenance and builder |
| `algocell_exp/shader/` | 222 | 8,850,289 | Historical exported executors/worlds, metadata and traced derivatives |
| `.venv/` | 7,496 | 244,171,989 | Recreate from a lock; do not copy the interpreter/environment |
| `library/` | 49 | 68,341,598 | Ignored literature cache; release citations/manifests and authorized material |
| `manuscript/figures/refs/` | 40 | 17,106,372 | Ignored design/reference inputs; actual build inputs must be preserved |
| `/tmp/figs-v62-c1` (outside repo) | 80 | 2,971,902 | Required C1 E trajectories and completion markers, also retrievable from Modal |
| `/private/tmp/algocell-s1audit` (outside repo) | 5 | 7,448,743 | Four selected S1 logs and one final soup; audit subset, not all 80 worlds |

Modal **`algocell-atlas-runs` contains 60,714 files, 489 directory entries, and 27,709,689,964 bytes (27.71 GB / 25.81 GiB)**. The complete prefix inventory is below. This is logical file content size, not billing or compressed physical storage. The listing does not establish a provider-side immutable snapshot; take a timestamped manifest once writers are quiescent at migration.

Do **not** sum local and Modal size as a unique dataset. Their gross raw total is **54,547,767,330 bytes**. A basename-plus-size comparison finds **54,939 possible matching remote files / 25,196,562,830 bytes**; **5,775 remote files / 2,513,127,134 bytes** have no such local match. This is only a discovery heuristic: repeated names, alternate directory nesting and changed content require SHA-256 matching. The heuristic raw-union estimate is 29,351,204,500 bytes, **not a verified release size**. S2 downloads under `results/` and external `/tmp` copies overlap the volume again. No remote contents were downloaded just to establish these totals.

Important bulk groups: local Stage E 12.39 GB, Stage C 3.36 GB, BFF Modal copies 2.63 GB, Stage F 1.54 GB, Stage D 1.50 GB; remote `impact_i1` 1.58 GB and `tier2` 503.03 MB are additional release-critical evidence. Large generated tables include `census8086/law_serial.csv.gz` (39,448,913 bytes), four 8086 census CSVs (29.76–31.38 MB each), `evolved6502/geometry_traces.csv.gz` (24,796,151), `evolved6502/exact_serial.csv.gz` (21,461,607), `crystal/cells_with_offspring.csv` (25,412,324), and S2 raw fate/prelaunch tables. Keep their manifests in Git; the release can store these byte-identical files in Zenodo rather than growing Git history.

## Proposed repository layout and boundaries

```text
algocell-paper/
  README.md                    # paper identity, status, DOI, route to reproduction
  REPRODUCE.md                 # exact command DAG, expected outputs and checks
  CITATION.cff
  LICENSE                      # chosen code licence; separate data/third-party notices
  pyproject.toml               # installable algocell_exp package and CLI/test config
  env/
    requirements.in
    requirements-macos-arm64.lock.txt
    requirements-linux-gpu.lock.txt
    requirements-py65.txt
    package.json + package-lock.json  # minimal TS exporter/validator tools
    zilion-lock.json            # per-spec commit/tag, package, compat and hashes
    toolchain.json              # Python/Node/Poppler/TeX/ffmpeg/GPU details
    modal-images.json           # frozen image IDs and image dependency manifests
  src/
    algocell_exp/               # executor/soup/assay/provenance research adapters
      machines/                # specs + vendored WGSL/manifests + provenance
      shader/                  # frozen historical C0 kernels and derivatives
    paths.py                   # explicit root/data/output resolution
    *.py                       # initially preserve entrypoint names; group later
    legacy/                    # frozen TS export/validation host code
  registrations/
    originals/                 # byte-preserved PLAN/REVISION_PREREG/BIOLOGY_PREREG
    chronology.csv             # dated section/amendment -> original path/commit/hash
    amendments/2026-10-*/
    freezes/                   # eligibility, prediction, source and launch freezes
  results/
    branching/ capacity/ evolved6502/ census6502/ census8086/
    s6502/ s8086/ impact/ tier2/ ...   # small canonical tables + reports
    provenance/                # source maps, audits, corrections, command manifests
  figures/
    *.py                       # active builders + required predecessor producers
    inputs/                    # designer panel, required reference/build inputs
    out/v8/                    # main PDF/SVG/PNG + plotting CSVs + QA hashes
    out/extended-data/         # final numbered displays, not just a map
  manuscript/
    MAIN_v8.md SI_v8.md LEGENDS_V8.md
    references.bib             # extracted verified reference apparatus
    audits/                    # v7 audit, CSV and v8 per-row reconciliation
    builds/                    # final PDFs; no transient TeX/cache in Git
  data/
    README.md
    manifests/                 # remote/local original path, bytes, SHA-256, DOI URL
    validation/                # pinned vector/baseline ledgers and input manifests
    raw/                       # ignored verified download/cache tree
  tests/
  archive/
    manuscript/ exploratory/ planning/ provenance/
    INDEX.csv                  # original paths, category, status, hash, source commit
    history/README.md          # where the original Git bundle/history can be obtained
```

Keep biological analyses (`biology/`), conditions (`conds/`), and required `micro/` helpers with `src/` or immutable input/config areas, according to their dependency roles. Keep all planned/review documents; move their inactive versions to `archive/planning/`. Preserve original names in a one-to-one `old_path -> new_path` inventory, including archive members. Avoid an immediate wholesale pipeline rewrite: copy, establish equivalence, then reorganize in a separate change.

There is **no production Svelte/web UI inside `experiments/`**. The research kernels are derived from app code, but are scientific execution contracts and belong to the paper. Algocell keeps its `src/`, static/UI assets, SvelteKit configuration and npm application lock. Eight shared historical TS files need frozen paper copies or archived source access; the live application keeps its own versions. Retire Algocell's research export command only after the paper exporter works independently. Do not remove a shared app module merely because the paper also needs it.

### History preservation

For the recommended fresh repository:

1. At the accepted migration snapshot, record the Algocell remote, complete original SHA, branch/refs, timestamps, status, and a file manifest. Resolve which concurrent v8 edits constitute the accepted manuscript; distinguish committed content, pending edits and ignored generated inputs.
2. Preserve the original repository history unchanged. Create a verified Git bundle **during migration**, from a dedicated read-only copy, covering the baseline and relevant original refs. Archive it alongside a manifest or in a separate historical deposit. Keep the original public Algocell commits addressable after cleanup; no force-push or rewrite of Algocell history.
3. Start the paper repository with a provenance note linking the original full SHAs; record for each copied source the last original source commit, original path and SHA-256. Record uncommitted sources as working-tree snapshots with their own hashes until they have an accepted original commit. Copy chronological registration text byte-for-byte. Preserve the full originals even if dated amendments are also split into readable files.
4. Keep existing embedded original commit references and historical manifests intact. Add a sidecar mapping for changed paths/source wrappers. Changing a pathname or embedding a new `source_commit` must never masquerade as an original execution identity.
5. Use a paper release tag and its Zenodo **version DOI** for the accepted code/data snapshot; list the upstream historical commit chain in `PROVENANCE.md`/`CITATION.cff`.

Concrete alternative if full active Git history is preferred: in a **new clone only**, run `git filter-repo --path experiments/ --path-rename experiments/:`; include the eight TS closure paths explicitly if they should also carry history. Audit historical renames first because path selection does not automatically follow them. Retain the original repository bundle and `.git/filter-repo/commit-map`; extraction changes commit identities and still requires a provenance mapping. Subsequently reorganize in ordinary commits. Do not execute this command in Algocell. [git-filter-repo documentation](https://github.com/newren/git-filter-repo/blob/main/Documentation/git-filter-repo.txt) describes path selection/renaming and the commit map.

### Raw data and Zenodo release

Use two related release objects initially: **paper code/derived evidence** and **raw histories/provenance**. If the verified union exceeds the quota, split the raw release into scientifically meaningful linked cohort records (historical Z80/BFF, S1/S2, I1/Tier-2), rather than omit failed attempts or silently downsample. The current local and remote copies each fit under 50 GB separately, but their undeduplicated gross raw sum does not. Zenodo's default is **50,000,000,000 bytes and 100 uploaded files per record**; extra quota may be available up to 200 GB. Bundle the thousands of members into a small number of archives, retaining per-member hashes and uncompressed byte counts. These current limits were checked in [Zenodo file guidance](https://help.zenodo.org/docs/deposit/manage-files/) and [quota guidance](https://help.zenodo.org/docs/deposit/manage-quota/).

| Destination | Contents | Size basis / release rule |
|---|---|---|
| Paper Git | Source, pinned locks, original registrations, specs, small canonical CSV/MD, plotting inputs, final figures, QA/manifests | Exact total to be measured from the E/S release selection; do not treat all 1.55 GB tracked data as required Git bytes |
| Zenodo raw histories | `.jsonl`, summary/condition records, Brotli `.u8.br` snapshots, completion markers and source identities | Local `runs/` 26.84 GB; remote volume 27.71 GB; deduplicate by hash and record all original path aliases |
| Zenodo derived evidence | Large gzip/CSV intervention, trace, census, judgment and source-map tables | Entire current `results/` 1.36 GB is an upper bound, not an additional independent raw dataset |
| Zenodo provenance/history | Attempts, amendments, launch/watchdog logs, saved billing/lifecycle records, corrected and superseded labels, Git bundle | Retain provenance without treating attempts as additional biological worlds |
| Modal operational mirror | Existing `algocell-atlas-runs` namespaces | Preserve existing names and controls; public reproduction must also work without Modal credentials |
| Local ignored caches | Downloads, rendered video frames, venv and build intermediates | Recreate; archive actual scientific input originals elsewhere before discarding a cache |

Every raw manifest must include logical cohort/world/arm/seed, original volume/path and local aliases, bytes, compressed SHA-256, compression, decompressed shape/endianness and optional decompressed SHA-256, execution contract/core/world/WGSL hashes, source/registration freeze, completion state, data DOI/version and exact download URL. Distinguish canonical complete worlds from failed/partial/recovered attempts. Preserve file order where JSONL chronology matters. Existing starting manifests: S2 `download_hashes.csv` (450 files), exploratory `input_hashes.csv` (120 inputs), `stageG_input_hashes.csv` (200 inputs), C2 `coverage.csv` (80 raw logs), C1 `c1_download_audit.json`, I1/Tier-2 source/dispatch/completion records. Store original and migrated manifests together, not one rewritten in place.

Reserve a DOI in a Zenodo draft once the coordinator authorizes release preparation; reservation is not publication. Replace availability placeholders with the released accessions only when the deposits are valid. [Zenodo DOI guidance](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/) explains reservation and registration on publication. Use new versions for scientific data corrections and pin the cited version; link code, data and paper records to one another.

For anonymised review, generate a **separate review snapshot** with neutral identifiers and stripped author-identifying Git history, account names, local paths, PDF metadata, billing identifiers and provenance links that reveal authors. Keep a private reversible mapping back to canonical filenames/hashes. Do not change scientific bytes, denominators or status labels. A preview-only expiring secret link to an unpublished review deposit may be suitable; it does not automatically anonymise creators/metadata. Inspect the reviewer-visible landing page before sharing. [Zenodo link-sharing guidance](https://help.zenodo.org/docs/share/link-sharing/) supports draft previews and restricted-file access. Preserve canonical unredacted records separately, and release full provenance when anonymity is no longer required. No review link or deposit was created by this task.

### Environment and Zilion pinning

The local interpreter is **CPython 3.12.12**, macOS arm64, created with uv 0.10.2. `requirements.txt` currently has ranges, not a lock: `wgpu>=0.32,<0.40`, `numpy>=2`, `pandas>=2`, `matplotlib>=3.8`, `brotli>=1.1`, `modal>=1.0`. `requirements-py65.txt` pins `py65==1.2.0`. The installed freeze has 54 packages; the complete observed freeze is appended. Critical installed versions include wgpu 0.32.0, NumPy 2.5.3, pandas 3.0.6, matplotlib 3.11.2, Brotli 1.2.0, Modal 1.6.1, Pillow 12.3.0, PyMuPDF 1.28.2 and pytest 9.1.1. Figure/PDF/test dependencies are missing from the range-based requirements. Record the whole observed freeze, then create validated platform-specific hashed locks; a macOS-only dependency such as `rubicon-objc` must not be blindly imposed on Linux Modal images.

Observed tools: Node v24.21.0, npm 11.19.0, Poppler `pdftoppm` 26.03.0, Tectonic 0.16.9, ffmpeg 9.0.1. Record their build/OS versions, font inventory, font files/licences, and GPU adapter/driver/backend in the release environment. Pin the minimal exporter toolchain independently of SvelteKit; do not copy the application's entire npm dependency set. Current application lock records `@neovand/zilion` 0.2.0, `z80-emulator` 2.3.0 and TypeScript 5.9.3. `export:sim` invokes `tsx`; ensure that executable is explicitly pinned in the paper environment rather than assumed globally available.

The paper needs **multiple frozen Zilion identities**, not one mutable `next` checkout:

| Spec family | Exact recorded Zilion commit | Package / execution distinction |
|---|---|---|
| `worlds.json`, `worlds_6502.json`, `worlds_assays.json`, `worlds_impact.json`, `worlds_tier2.json` | `183ff5c7429403439c664487980a40bf62f49b7b` | `@neovand/zilion` 0.3.0-dev; explicitly retain historical Z80 compat 0.2 and corrected compat 0.3 distinctions |
| `worlds_8086.json` | `39a60429a9b16c77cd845f6087062c4e2cc12c53` | 0.3.0-dev; NMOS 8086 compat 0.3, coherent fetch, zero-EA hidden latch, SP=1 frozen S2 contract |
| Historical `shader/` | Zilion **0.2.0** plus the frozen Algocell TS host/export sources | `meta.json` records version and truncated export digests; preserve full file SHA-256 in the release inventory |

Both full vendoring commits exist locally. The current sibling Zilion HEAD was `3c3534d76f228ce48c766df4aed98970f5dd4bd2`, so rebuilding everything from today's sibling checkout would not reproduce the recorded provenance. After Zilion is released, pin an **exact tag resolving to the intended commit**, plus package version/integrity; retain the original per-study commits even if a later tag is used for new experiments. Never replace historical results with regenerated modern-core simulations.

Future rebuild/check recipe, one clean checkout/build per recorded commit, using its own `package-lock.json`:

```sh
# In the separately prepared Zilion checkout at the recorded SHA:
npm ci
npm run build
# From the paper root after path adaptation, at SHA 183ff5c...:
node src/algocell_exp/machines/build_machines.mjs --zilion "$PAPER_ZILION_DIR" --spec worlds.json --check
node src/algocell_exp/machines/build_machines.mjs --zilion "$PAPER_ZILION_DIR" --spec worlds_6502.json --check
node src/algocell_exp/machines/build_machines.mjs --zilion "$PAPER_ZILION_DIR" --spec worlds_assays.json --check
node src/algocell_exp/machines/build_machines.mjs --zilion "$PAPER_ZILION_DIR" --spec worlds_impact.json --check
node src/algocell_exp/machines/build_machines.mjs --zilion "$PAPER_ZILION_DIR" --spec worlds_tier2.json --check
# Repeat npm ci/build in the SHA 39a60429... checkout, then:
node src/algocell_exp/machines/build_machines.mjs --zilion "$PAPER_ZILION_DIR" --spec worlds_8086.json --check
```

`PAPER_ZILION_DIR` must point to the applicable clean checkout. The builder currently checks shader/manifest bytes, recorded provenance commit and core/world/WGSL hashes; `--check` writes nothing. To regenerate, omit `--check` **only in a scratch output copy**, compare every byte, and verify with `zmachine.load_machine`. Do not use `--all` to overwrite all study families from one checkout. The present builder has Algocell-specific root/path assumptions and provenance text; adapt those without changing specs or emitted shader/manifest bytes. Preserve original provenance separately because new path wording changes its text hash.

Historical `shader/` regeneration additionally needs the eight-file TS closure at the actual export source revision and Zilion 0.2.0. `npm run export:sim` currently writes into Algocell's `experiments/` and includes timestamps; replace its output-root handling in the paper copy. Compare WGSL and ISA JSON bytes/full hashes; treat export timestamp metadata explicitly. Rebuild traced/lethal/dial/mixed/convolution variants using the `gen_*_shader.py` producers with frozen source hashes, and test that instrumentation leaves base encounter execution unchanged. In this inventory **all 352 vendored WGSL files matched `manifest.sha256.wgsl`**; rebuilding from source was not performed.

## Reproduction DAG and required command coverage

`REPRODUCE.md` must distinguish **reanalysis of the exact archived worlds**, **re-execution of deterministic fixed-input encounters**, and **new evolutionary simulations**. The soup's collision/mutation races mean the same seed does not guarantee byte-identical world replay. The migration gate is exact reproduction of published numbers from the original raw histories and frozen contracts; new worlds are a separate robustness check, not a replacement for the archive.

Initially reproduce in a scratch copy preserving the existing `experiments/` shape. Only then adapt `src/`, `figures/`, and `registrations/` paths through `paths.py` and explicit CLI options. Use `PAPER_ROOT`, `PAPER_DATA_ROOT`, `PAPER_OUTPUT_ROOT` and `PYTHONPATH=src`; outputs must go to a new scratch tree. Several current entrypoints write under `Path(__file__).parent / 'results'` and cannot yet run safely in the proposed layout. The future commands below are a required CLI contract after that adaptation, not commands executed or certified by this planning task.

| Evidence / display | Archived inputs and producers | Required checks |
|---|---|---|
| Figure 1; validation and early atlas | Stage E `assays.csv`, seed-2002 video JSONL, I1/Tier-2 tables; legacy `make_figures.py`/`concept.py`; external 8086 validation baseline | KM cohorts, illustrative-world identity, independent-vector provenance, core C0/C1/C2 and experiment C1/C2 distinguished |
| Figure 2; serial/branching ED | `serial_retention.py`, `branching_transfer.py`, `pe_matrix.py`, `archive_erasure_v6.py`, corresponding raw assay tables | all starting-population denominators, control carriage, surviving erasure, actual ordered ancestry; seed-6003/6004 specimen distinction |
| Figure 3; overlap/C1/C2 ED | `crystal_check.py`, `capacity_assays.py`, `evolved6502_retention.py`, geometry/case/serial archives | eligibility frozen before assay, joint substitutions, exact source maps, 101 settings/42 roles, 41 distinct 6502 tapes/11 worlds, 8086 registered law remains untested |
| Figure 4; environmental/causal ED | raw `offset/`, C1 `tier2/E*`, `dial/`, `offset_switch.py`, `dial_soups.py`, `tier2_report.py`, toxin/N2/intruder evidence | raw trajectories, median definition, missing R+T=0 ratio retained, all 22 claim/36 component verdicts and negative results |
| Figure 5; machine-route/S2 ED | Stage G, corrected S1, canonical S2 raw+scored downloads, `score6502.py`, `correction_s1.py`, `score8086.py`, `law8086.py`, S2 `generate_report.py` | 80/20/20 route partitions; 14 detected, 2 open, 12 confined, 6 censored; KM medians; ten physical-phase first tapes |
| Cross-machine census/literal ED/SI | `census6502.py report`, `census8086_report.py`, literal-channel verification scripts and corrected labels | final SP=1 phase, earlier SP=0 inventory archived, finite grammar limits, constructed/evolved distinction, exact table hashes |
| BFF / biological metrics / genealogy ED/SI | BFF cohorts, `biology/`, `lod*`, `lod_n6`, registered and exploratory ledgers | fetch/inflow/carriage distinct; prospective versus discovery genealogy; failed components retained |
| Post hoc S2 numbers | `exploratory8086.py` stages and verifier; canonical S2+Stage G raw inputs and frozen predictions | 11,700 sample records versus 35,100 ranked tapes; 2,000,000 checkpoint cells; 600 admissions, 32-genotype panel, 2,224,640 encounters; no registered verdict overwrite |
| Every remaining main/SI number | v7 audit CSV, per-row resolution, new v8 claim-to-output ledger; registration constants and exact manuscript reference/figure paths | definition versus observation, named filters, units, denominator, rounding, evidence hash and original row ID; no “same numeral occurs” checks |

Known exact baseline commands (run **later, in a scratch copy**, from its `experiments/` directory; `PAPER_PYTHON` points to the locked interpreter):

```sh
# CPU table/report regeneration from already verified source evidence
"$PAPER_PYTHON" census6502.py report
"$PAPER_PYTHON" census8086_report.py
"$PAPER_PYTHON" capacity_assays.py report
"$PAPER_PYTHON" evolved6502_retention.py report
"$PAPER_PYTHON" correction_s1.py
"$PAPER_PYTHON" tier2_report.py --data results/tier2
"$PAPER_PYTHON" -I results/s8086/provenance/generate_report.py
"$PAPER_PYTHON" -I exploratory8086_verify.py
"$PAPER_PYTHON" -I exploratory8086.py report

# Rebuild the full plotting-input chain, not only the final copied CSVs
"$PAPER_PYTHON" manuscript/figures/fetch_c1_v62.py --destination "$PAPER_C1_CACHE"
"$PAPER_PYTHON" manuscript/figures/prepare_v6.py --runs "$PAPER_RAW_ROOT" --c1-runs "$PAPER_C1_CACHE"
"$PAPER_PYTHON" manuscript/figures/prepare_v62.py
"$PAPER_PYTHON" manuscript/figures/prepare_v7.py
"$PAPER_PYTHON" manuscript/figures/prepare_v8.py
"$PAPER_PYTHON" manuscript/figures/build_v8.py
"$PAPER_PYTHON" manuscript/figures/legends_v8.py
"$PAPER_PYTHON" manuscript/figures/verify_v8.py --technical-only
"$PAPER_PYTHON" manuscript/refcheck.py --md manuscript/MAIN_v8.md
```

The fetch script currently hardcodes Algocell's Modal binary; replace that with the locked client and add a credential-free Zenodo retrieval path. If the needed C1 raw logs are already hash-verified, use them without fetching. `prepare_v6.py` also hardcodes an external 8086 baseline; make that a pinned `data/validation/` input. Build/recover the legacy vector panel `manuscript/figures/out/fig1.pdf` before the final figure build. Recover any immutable designer input needed for that panel; do not replace it with an unrecorded image. `archive_erasure_v6.py` is the exact fixed-input GPU trace producer if the ordered trace is rebuilt; compare the regenerated events to the saved archive.

Current full-assay command contracts that must be ported and exercised against verified raw inputs:

```sh
# These perform foreground local GPU work; run later in the scratch reproduction tree.
"$PAPER_PYTHON" capacity_assays.py qualify
"$PAPER_PYTHON" capacity_assays.py assay
"$PAPER_PYTHON" evolved6502_retention.py qualify --archive "$PAPER_S1_RAW"
"$PAPER_PYTHON" evolved6502_retention.py assay
"$PAPER_PYTHON" score8086.py --input "$PAPER_S2_WORLD" --out "$PAPER_SCORE_OUTPUT"
"$PAPER_PYTHON" -I exploratory8086.py scan
"$PAPER_PYTHON" -I exploratory8086.py reextract
"$PAPER_PYTHON" -I exploratory8086.py encounters
"$PAPER_PYTHON" -I exploratory8086.py transmission
"$PAPER_PYTHON" -I exploratory8086.py world_family
"$PAPER_PYTHON" -I exploratory8086.py law
"$PAPER_PYTHON" -I exploratory8086_verify.py
```

Do not refreeze eligibility after observing outcomes: compare regenerated qualification to the original ledger, then assay the frozen panel. The current S2 law code enforces original source hashes and committed freezes; changing its path/root resolver changes source bytes. Preserve its original executable source or provide a documented compatibility loader and a separate migrated-source identity; do not patch historical hash fields to pass checks. Record every concrete world/arm/machine command, including filenames, in `REPRODUCE.md`/a machine-readable job manifest. The exact successful Tier-2 scoring commands already exist in `results/tier2/T2_COMMANDS.csv`; translate all 15 recorded invocations' raw/output roots, not their scientific flags. Historical stage shell pipelines and I1 saved command/manifests provide the other job definitions. Generate the expanded commands from those frozen jobs, so no required cohort hides behind “run analysis”.

For the final layout, the CPU figure commands become `python figures/prepare_v6.py ...`, then `prepare_v62.py`, `prepare_v7.py`, `prepare_v8.py`, `build_v8.py`, `legends_v8.py`, `verify_v8.py --technical-only` with `PYTHONPATH=src` and explicit roots. `python src/capacity_assays.py report`, `python src/tier2_report.py --data results/tier2`, and the corresponding S1/S2 commands must write to the configured output root. Finish `REPRODUCE.md` with **fully expanded tested paths**, versions, wall time/GPU requirements and expected SHA-256/numeric results. This plan intentionally does not claim these relocated commands already work.

## Findings that must be reconciled before cleanup

- `AUDIT_v7.md`/CSV reports **682 numeric occurrences: 528 verified, 150 not established by the nearest source, 3 without a source bracket, and 1 mismatch**. The mismatch is the period-four/offset-eight example: **six total sites, four unexecuted sites**. The new working `AUDIT_v7_RESOLUTION.md` supplies a per-row reconciliation and v8 corrects the wording. A read-only ID comparison confirmed **154 unique entries for exactly the 154 originally flagged IDs, with no missing or extra IDs**. Keep both the original audit and resolution; independently verify the decisions and the final v8 number ledger before calling the numerical audit closed. Source reconciliation is not a new experiment.
- V7 has eight distinct missing author–year reference entries and nine mapped-but-not-rendered ED packages. The new v8 text/changes describe a reference apparatus and ED targets. Test the final actual files; a map or handoff does not establish that a numbered display has been rendered. `build_pdf.py` still uses older v3/v4 figure maps and Tectonic; update and verify a v8-specific main/SI/ED mapping before rendering the new manuscript.
- A read-only `verify_v8.py --technical-only` run **failed at missing `figures/out/v8/fig1.svg`**. Input/scientific assertions preceding that point passed, but the complete check did not. All **43 `sources_v8.csv` rows and 40 `hashes.csv` rows** independently matched current bytes; the v6 `sources.csv` chain also matched. Restore/rebuild the missing export and rerun the full technical check in the later figure work. Existing QA/critique manifests do not certify newly changed artwork.
- The audit fingerprints for external Zilion `docs/ARCHITECTURE.md` and `docs/I8086.md` differ from the files in today's sibling checkout. Recover the exact historical doc bytes by hash/commit and archive them, or explicitly qualify the updated text. Current docs were 61,983 and 10,734 bytes; the separate 8086 validation baseline exists (137,350 bytes; SHA-256 `d86a55d47a6e9a580dcae7b6c48a5df2a594f4349257c01b9ebaab3047707916`) and is part of the figure source chain.
- Current code has absolute paths to Algocell, an S2 worktree, a Zilion worktree, `/private/tmp/algocell-s1audit`, `/tmp/figs-v62-c1`, and `/runs`; frozen reports sometimes require `git diff HEAD`/tracked-file checks. Refactor resolution without losing original evidence identity. The C2 raw corpus is **80 logs in `runs/s6502_raw/`**, not the five-file S1 audit cache. S2's canonical 20 worlds/checkpoints are under `results/s8086/downloads/worlds_A/A/`, **84,553,109 bytes**; failed attempts are distinct provenance.
- Availability/licence/hardware/toolchain details remain release tasks. Hash inventory and a Modal volume name are not public data accessions. Figure sources, independent test vectors, literature PDFs and design references need correctly scoped licences/citations; do not assume all ignored local files can be redistributed.

## Migration sequence and acceptance gate

1. **Freeze and reconcile.** Coordinator selects the accepted v8/SI/figure state after concurrent work finishes. Preserve v7, v8, audit, exception ledger, all registrations, original and corrected outcomes, and current source/output hashes. Capture tracked/untracked/ignored state and full Git references. Complete a file-level category/destination manifest from this inventory; mark exclusions explicitly.
2. **Protect history and bulk data.** Produce and verify the history bundle in a separate workspace. Obtain a quiescent Modal metadata inventory; download remote-only required inputs and all cited source archives. Match hashes, establish canonical namespaces, retain alias mappings and failed-attempt archives. Measure the deduplicated release and archive packaging overhead. Do not delete any local or remote copy.
3. **Prepare release artifacts.** Build Zenodo archives/manifests and the neutral review snapshot if needed, without publishing until authorized. Verify every compressed member hash and decompressed shape; reproduce lookup by DOI/URL and by Modal path. Keep raw archives, generated evidence and operational logs identifiable separately.
4. **Create the paper repository later.** Use the recommended fresh repository, add provenance/history access and original registrations first, then source/contracts, canonical small results, figure inputs and manuscripts. Put uncited older material under indexed `archive/`; bulky members can remain in the archival deposit with their manifests. Record every original-to-new path and byte hash.
5. **Lock environments and external inputs.** Create validated macOS/Linux locks and the minimal Node lock; pin per-study Zilion commits, export source revision, baseline/vector manifests, fonts and native tools. Recreate the environment on a clean machine/container. Verify/rebuild every required machine family; compare all WGSL/manifest bytes and test source-map instrumentation.
6. **Establish the baseline in isolation.** In the initial compatibility tree, rederive all paper tables from exact raw histories/frozen judgments; re-execute required fixed-input assays from raw snapshots. Do not launch new paid worlds as a substitute. Run the recorded successful commands and required regression suites; preserve known failures/untested verdicts and nested denominators. Store command, input, source, environment, output and timing hashes in a reproduction manifest.
7. **Adapt paths and layout incrementally.** Introduce explicit input/output roots, remove absolute worktree assumptions, relocate sources/builders with an import-compatible package, and preserve frozen executables where hash-enforced contracts require it. Repeat the same baseline jobs. Store original scientific manifests unchanged plus migrated path/source identities. Never write reproduced evidence over the sole archived original.
8. **Verify all final scientific artifacts.** Require a row-by-row `claim_id -> exact input/filter/aggregate -> output -> main/SI/figure location` ledger for every number. Counts/denominators/verdicts/eligibility/seeds match exactly; float tolerances and display rounding are declared per field. Compare CSV/GZip parsed content when serialization timestamps differ; retain canonical byte hashes too. Rebuild all five main figures and every numbered ED target in PDF/SVG/PNG, verify plotted coordinates, text, specimen identities, vector text, page dimensions, resolution and clipping, and visually review final exports. Normalize timestamp metadata for reproducible byte checks or document the exceptions. Rebuild main/SI PDFs with the v8 figure map and references. Verify all local links and public accession placeholders.
9. **Test independence.** In a second fresh clone using only the published/downloadable manifest inputs, run documented reproduction without Algocell, adjacent Zilion worktrees, old `/tmp` caches or private Modal credentials. The reviewer must be able to obtain every required byte and execute the documented command DAG. Run tests for executors, ring/wrap/phase/source maps, fixed raw replay, qualification and report aggregation; run the technical figure checker and a strict final manuscript audit. CPU saved-table success alone does not prove raw assay reproduction.
10. **Release, then clean Algocell.** Record the successful reproduction manifest, code tag and data version DOI. **Only after steps 1–9 pass** may a separate Algocell cleanup remove paper-only current-tree material and retire paper entrypoints. Keep application TS modules/runtime assets, extract only the paper-specific dev harness if appropriate, and run the web application's required check/build/Playwright checks. Leave a small paper link/provenance pointer. Preserve original history and data backups; do not rewrite Algocell history or remove the Modal mirror as part of routine cleanup.

Acceptance is binary: **no Algocell cleanup until every retained number, figure and raw-input dependency reproduces in the new repository**, and every archived item is recoverable by manifest. Missing SVGs, unresolved v8 claims, unrendered ED, changed source identities or private-only inputs are reasons to keep the original tree intact, not reasons to delete or omit evidence.

## Inventory appendices

The following tables are mechanically assembled from the read-only file/graph inventory. Paths are relative to `experiments/` except where stated. E is conservative: explicit directory citations, old SI build inputs, registration pipelines and their import closure remain included until the accepted manuscript and runtime manifests justify a narrower release. S sibling files are retained rather than asserted to be essential. Generated caches and OS metadata are A/nonresearch. No table authorizes a move.




### A. Every top-level experiment item

The size snapshot excludes this newly written plan itself. Counts include ignored/cache files. Mixed directories have their descendant dispositions in later tables. `.venv` is E as an environment specification, with the installed directory treated as a reproducible cache.

| Item | Category | Files | Bytes | Tracked-path bytes | Basis |
|---|---|---|---|---|---|
| `.DS_Store` | A | 1 | 6,148 | 0 | no established active paper dependency; retain in archive |
| `.gitignore` | A | 1 | 81 | 81 | no established active paper dependency; retain in archive |
| `.pytest_cache` | A | 5 | 49,115 | 0 | nonresearch cache |
| `.venv` | E (recreate) | 7496 | 244,171,989 | 0 | 54-package freeze, Python 3.12.12; do not copy environment |
| `BIOLOGY_BRIDGE.md` | A | 1 | 10,935 | 10,935 | no established active paper dependency; retain in archive |
| `BIOLOGY_PREREG.md` | E | 1 | 8,990 | 8,990 | submission/registration root |
| `DEEP_RESEARCH_PROMPT.md` | A | 1 | 16,563 | 16,563 | no established active paper dependency; retain in archive |
| `DISCOVERY_PRINCIPLES.md` | A | 1 | 7,860 | 7,860 | no established active paper dependency; retain in archive |
| `HANDOFF_2026-10-09.md` | S | 1 | 21,385 | 21,385 | audit/correction/review/operational provenance |
| `INSIGHT.md` | A | 1 | 10,445 | 10,445 | no established active paper dependency; retain in archive |
| `LITERATURE.md` | E | 1 | 20,587 | 20,587 | shell dependency of manuscript/build_si.sh |
| `NATURE_PLAN.md` | A | 1 | 6,357 | 6,357 | no established active paper dependency; retain in archive |
| `NIGHT_PLAN_2026-10-09.md` | A | 1 | 11,211 | 11,211 | no established active paper dependency; retain in archive |
| `PLAN.md` | E | 1 | 76,381 | 76,381 | source bracket manuscript/SI.md |
| `RESEARCH_DIRECTIONS.md` | A | 1 | 9,317 | 9,317 | no established active paper dependency; retain in archive |
| `REVIEW.md` | E | 1 | 24,451 | 24,451 | source bracket manuscript/SI.md |
| `REVIEW_1_ASSESSMENT.md` | S | 1 | 9,977 | 9,977 | audit/correction/review/operational provenance |
| `REVIEW_2_ASSESSMENT.md` | S | 1 | 7,215 | 7,215 | audit/correction/review/operational provenance |
| `REVIEW_R2_1_ASSESSMENT.md` | S | 1 | 5,791 | 5,791 | audit/correction/review/operational provenance |
| `REVIEW_R2_2_ASSESSMENT.md` | S | 1 | 4,026 | 4,026 | audit/correction/review/operational provenance |
| `REVISION_PREREG.md` | E | 1 | 224,895 | 224,895 | v7 exception resolution |
| `STATE_2026-10-07.md` | S | 1 | 13,789 | 13,789 | audit/correction/review/operational provenance |
| `STATE_2026-10-08.md` | S | 1 | 16,883 | 16,883 | audit/correction/review/operational provenance |
| `STORY.md` | A | 1 | 17,645 | 17,645 | no established active paper dependency; retain in archive |
| `STORY_FOR_NON_EXPERTS.md` | A | 1 | 67,551 | 67,551 | no established active paper dependency; retain in archive |
| `SYNTHESIS.md` | A | 1 | 19,950 | 19,950 | no established active paper dependency; retain in archive |
| `THEOREMS.md` | E | 1 | 24,071 | 24,071 | shell dependency of manuscript/build_si.sh |
| `THEORY.md` | E | 1 | 28,798 | 28,798 | shell dependency of manuscript/build_si.sh |
| `__pycache__` | A | 7 | 110,615 | 0 | nonresearch cache |
| `algocell_exp` | E:971 | 981 | 28,882,768 | 28,250,629 | see module/cohort/archive scope below |
| `analyze.py` | E | 1 | 12,755 | 12,755 | shell dependency of analyze_stage.sh |
| `analyze_stage.sh` | E | 1 | 755 | 755 | environment or registered pipeline dependency |
| `assay_batch.py` | E | 1 | 13,638 | 13,638 | shell dependency of analyze_stage.sh |
| `assays_any_isa.py` | E | 1 | 57,270 | 57,270 | producer/control dependency of cited assays_any_isa |
| `audit_prefix8086.py` | E | 1 | 14,218 | 14,218 | transitive import/static input or frozen machine contract |
| `audit_reset8086.py` | E | 1 | 6,230 | 6,230 | producer/control dependency of cited census8086 |
| `audit_s1.py` | E | 1 | 26,392 | 26,392 | producer/control dependency of cited s6502 |
| `b6audit8086.py` | E | 1 | 8,655 | 8,655 | producer/control dependency of cited census8086 |
| `benchmark8086.py` | E | 1 | 16,055 | 16,055 | pipeline named in REVISION_PREREG.md |
| `bff_window.py` | E | 1 | 2,266 | 2,266 | producer/control dependency of cited bff |
| `biology` | E:3, A:1 | 4 | 83,566 | 83,566 | see module/cohort/archive scope below |
| `branching_table.py` | E | 1 | 1,819 | 1,819 | transitive import/static input or frozen machine contract |
| `branching_transfer.py` | E | 1 | 25,202 | 25,202 | source bracket manuscript/SI_v8.md |
| `c4_functional.py` | E | 1 | 6,054 | 6,054 | shell dependency of stage_pipeline.sh |
| `capacity_assays.py` | E | 1 | 14,099 | 14,099 | pipeline named in manuscript/MAIN_v7.md |
| `capacity_common.py` | E | 1 | 7,454 | 7,454 | source bracket manuscript/MAIN_v8.md |
| `capacity_over_time.py` | E | 1 | 5,033 | 5,033 | transitive import/static input or frozen machine contract |
| `census6502.py` | E | 1 | 31,963 | 31,963 | transitive import/static input or frozen machine contract |
| `census6502_report.py` | E | 1 | 50,166 | 50,166 | source bracket manuscript/SI_v8.md |
| `census8086.py` | E | 1 | 10,195 | 10,195 | pipeline named in REVISION_PREREG.md |
| `census8086_law.py` | E | 1 | 9,584 | 9,584 | pipeline named in REVISION_PREREG.md |
| `census8086_report.py` | E | 1 | 23,059 | 23,059 | producer/control dependency of cited census8086 |
| `census_dynamics.py` | A | 1 | 6,713 | 6,713 | no established active paper dependency; retain in archive |
| `census_period.py` | E | 1 | 4,271 | 4,271 | pipeline named in REVISION_PREREG.md |
| `check_8080_closer.py` | E | 1 | 2,127 | 2,127 | pipeline named in REVISION_PREREG.md |
| `check_py65_difftest.py` | S | 1 | 6,569 | 6,569 | audit/correction/review/operational provenance |
| `check_s6502_smoke.py` | E | 1 | 8,138 | 8,138 | transitive import/static input or frozen machine contract |
| `closer_census.py` | A | 1 | 4,326 | 4,326 | no established active paper dependency; retain in archive |
| `closure.py` | E | 1 | 8,525 | 8,525 | source bracket manuscript/SI.md |
| `conds` | E:15 | 15 | 1,593,398 | 1,593,398 | see module/cohort/archive scope below |
| `conv_soups.py` | E | 1 | 8,292 | 8,292 | pipeline named in REVISION_PREREG.md |
| `core_families.py` | E | 1 | 1,728 | 1,728 | producer/control dependency of cited offset |
| `correction_s1.py` | E | 1 | 24,587 | 24,587 | producer/control dependency of cited s6502 |
| `crystal_check.py` | E | 1 | 45,886 | 45,886 | producer/control dependency of cited crystal |
| `crystal_roles.py` | E | 1 | 4,232 | 4,232 | producer/control dependency of cited crystal |
| `descriptive8086.py` | E | 1 | 6,474 | 6,474 | producer/control dependency of cited census8086 |
| `descriptive_report8086.py` | E | 1 | 7,662 | 7,662 | producer/control dependency of cited census8086 |
| `detectors.py` | E | 1 | 8,081 | 8,081 | pipeline named in BIOLOGY_PREREG.md |
| `diagnose_reset_stop8086.py` | S | 1 | 7,089 | 7,089 | audit/correction/review/operational provenance |
| `dial_soups.py` | E | 1 | 10,500 | 10,500 | v7 exception resolution |
| `dials_score.py` | E | 1 | 6,598 | 6,598 | pipeline named in REVISION_PREREG.md |
| `dispatch8086.py` | E | 1 | 27,808 | 27,808 | pipeline named in REVISION_PREREG.md |
| `emulator_report.py` | E | 1 | 2,563 | 2,563 | producer/control dependency of cited review_r2 |
| `evolved6502_retention.py` | E | 1 | 23,091 | 23,091 | v7 exception resolution |
| `evolved6502_trace.py` | E | 1 | 5,831 | 5,831 | transitive import/static input or frozen machine contract |
| `exact_copy.py` | E | 1 | 3,157 | 3,157 | transitive import/static input or frozen machine contract |
| `exec_trace.py` | E | 1 | 6,150 | 6,150 | transitive import/static input or frozen machine contract |
| `exploratory8086.py` | E | 1 | 17,811 | 17,811 | pipeline named in manuscript/MAIN_v7.md |
| `exploratory8086_compute.py` | E | 1 | 15,858 | 15,858 | transitive import/static input or frozen machine contract |
| `exploratory8086_law.py` | E | 1 | 13,441 | 13,441 | pipeline named in manuscript/MAIN_v7.md |
| `exploratory8086_report.py` | E | 1 | 25,511 | 25,511 | transitive import/static input or frozen machine contract |
| `exploratory8086_verify.py` | E | 1 | 5,305 | 5,305 | pipeline named in manuscript/MAIN_v7.md |
| `explore_s1_6502.py` | E | 1 | 30,980 | 30,980 | producer/control dependency of cited s6502 |
| `fetch.sh` | E | 1 | 492 | 492 | shell dependency of analyze_stage.sh; preserve historical downloader/publisher |
| `fig2_numbers.py` | A | 1 | 3,830 | 3,830 | no established active paper dependency; retain in archive |
| `figstyle.py` | E | 1 | 12,393 | 12,393 | transitive import/static input or frozen machine contract |
| `findings.py` | E | 1 | 9,229 | 9,229 | pipeline named in manuscript/SI.md |
| `first_closure.py` | E | 1 | 6,177 | 6,177 | pipeline named in REVISION_PREREG.md |
| `i8086_value_tags.py` | E | 1 | 6,318 | 6,318 | pipeline named in REVISION_PREREG.md |
| `impact_assay_supplement.py` | E | 1 | 5,391 | 5,391 | transitive import/static input or frozen machine contract |
| `impact_confirm.py` | E | 1 | 8,857 | 8,857 | transitive import/static input or frozen machine contract |
| `impact_i1_report.py` | E | 1 | 29,768 | 29,768 | transitive import/static input or frozen machine contract |
| `impact_modal.py` | E | 1 | 17,308 | 17,308 | transitive import/static input or frozen machine contract |
| `impact_report.py` | E | 1 | 23,826 | 23,826 | transitive import/static input or frozen machine contract |
| `impact_study.py` | E | 1 | 26,615 | 26,615 | transitive import/static input or frozen machine contract |
| `inflow_insoup.py` | E | 1 | 19,088 | 19,088 | pipeline named in REVISION_PREREG.md |
| `intervention_stop8086.py` | S | 1 | 12,384 | 12,384 | audit/correction/review/operational provenance |
| `invasion.py` | E | 1 | 7,336 | 7,336 | transitive import/static input or frozen machine contract |
| `invasion_aligned.py` | E | 1 | 6,577 | 6,577 | pipeline named in REVISION_PREREG.md |
| `invasion_closer.py` | E | 1 | 8,520 | 8,520 | pipeline named in REVISION_PREREG.md |
| `invasion_generic.py` | E | 1 | 4,017 | 4,017 | transitive import/static input or frozen machine contract |
| `invasion_pair.py` | E | 1 | 6,725 | 6,725 | pipeline named in REVISION_PREREG.md |
| `kin_census.py` | E | 1 | 4,354 | 4,354 | pipeline named in REVISION_PREREG.md |
| `l50_lineage.py` | E | 1 | 2,195 | 2,195 | pipeline named in REVISION_PREREG.md |
| `landscape.py` | E | 1 | 9,555 | 9,555 | transitive import/static input or frozen machine contract |
| `law8086.py` | E | 1 | 28,693 | 28,693 | pipeline named in REVISION_PREREG.md |
| `library` | S/A | 49 | 68,341,598 | 0 | ignored literature; citation/rights review before release |
| `lifecycle8086.py` | E | 1 | 32,384 | 32,384 | transitive import/static input or frozen machine contract |
| `lineage_examples.py` | E | 1 | 2,696 | 2,696 | transitive import/static input or frozen machine contract |
| `lit_review` | S:5 | 5 | 401,046 | 401,046 | see module/cohort/archive scope below |
| `lod.py` | E | 1 | 21,273 | 21,273 | pipeline named in REVISION_PREREG.md |
| `lod_analyse.py` | E | 1 | 9,613 | 9,613 | transitive import/static input or frozen machine contract |
| `lod_chain.py` | E | 1 | 7,002 | 7,002 | shell dependency of n6_analyse.sh |
| `lod_found.py` | A | 1 | 8,793 | 8,793 | no established active paper dependency; retain in archive |
| `lod_founder_entry.py` | A | 1 | 1,806 | 1,806 | no established active paper dependency; retain in archive |
| `lod_founders2.py` | E | 1 | 3,727 | 3,727 | shell dependency of n6_analyse.sh |
| `lod_founding.py` | E | 1 | 13,049 | 13,049 | shell dependency of n6_analyse.sh |
| `lod_founding_summary.py` | E | 1 | 2,010 | 2,010 | shell dependency of n6_analyse.sh |
| `lod_neighbours.py` | A | 1 | 4,043 | 4,043 | no established active paper dependency; retain in archive |
| `lod_openfirst.py` | E | 1 | 14,933 | 14,933 | pipeline named in REVISION_PREREG.md |
| `lod_origin.py` | E | 1 | 6,982 | 6,982 | transitive import/static input or frozen machine contract |
| `lod_parts.py` | E | 1 | 5,267 | 5,267 | shell dependency of n6_analyse.sh |
| `lod_parts_first.py` | E | 1 | 2,815 | 2,815 | shell dependency of n6_analyse.sh |
| `lod_reach.py` | E | 1 | 4,813 | 4,813 | shell dependency of n6_analyse.sh |
| `lod_traj.py` | E | 1 | 8,227 | 8,227 | pipeline named in REVISION_PREREG.md |
| `lod_v1.py` | E | 1 | 15,914 | 15,914 | transitive import/static input or frozen machine contract |
| `make_conds.py` | E | 1 | 17,978 | 17,978 | transitive import/static input or frozen machine contract |
| `manuscript` | E:197, S:413, A:1080 | 1693 | 1,091,387,770 | 179,965,625 | see module/cohort/archive scope below |
| `marker_check.py` | E | 1 | 6,407 | 6,407 | pipeline named in REVISION_PREREG.md |
| `meter8086.py` | E | 1 | 5,320 | 5,320 | pipeline named in REVISION_PREREG.md |
| `micro` | E:5, A:3 | 8 | 65,390 | 65,390 | see module/cohort/archive scope below |
| `modal_app.py` | E | 1 | 8,172 | 8,172 | transitive import/static input or frozen machine contract |
| `modal_bff.py` | E | 1 | 4,877 | 4,877 | pipeline named in manuscript/SI.md |
| `modal_census.py` | E | 1 | 5,291 | 5,291 | producer/control dependency of cited census6502 |
| `modal_conv.py` | A | 1 | 2,584 | 2,584 | no established active paper dependency; retain in archive |
| `modal_dial.py` | E | 1 | 3,875 | 3,875 | pipeline named in REVISION_PREREG.md |
| `modal_impact.py` | E | 1 | 4,756 | 4,756 | transitive import/static input or frozen machine contract |
| `modal_impact_confirm.py` | E | 1 | 2,375 | 2,375 | transitive import/static input or frozen machine contract |
| `modal_lod.py` | E | 1 | 1,671 | 1,671 | pipeline named in REVISION_PREREG.md |
| `modal_offset.py` | E | 1 | 2,384 | 2,384 | pipeline named in REVISION_PREREG.md |
| `modal_pilot6502.py` | A | 1 | 1,968 | 1,968 | no established active paper dependency; retain in archive |
| `modal_probe.py` | A | 1 | 2,074 | 2,074 | no established active paper dependency; retain in archive |
| `modal_r4.py` | E | 1 | 2,766 | 2,766 | transitive import/static input or frozen machine contract |
| `modal_s6502.py` | E | 1 | 4,723 | 4,723 | pipeline named in REVISION_PREREG.md |
| `modal_s8086.py` | E | 1 | 5,257 | 5,257 | pipeline named in REVISION_PREREG.md |
| `modal_score6502.py` | E | 1 | 11,626 | 11,626 | pipeline named in REVISION_PREREG.md |
| `modal_score8086.py` | E | 1 | 5,263 | 5,263 | pipeline named in REVISION_PREREG.md |
| `modal_tier2.py` | E | 1 | 12,528 | 12,528 | producer/control dependency of cited tier2 |
| `modal_tier2_dispatch.py` | E | 1 | 10,887 | 10,887 | pipeline named in REVISION_PREREG.md |
| `modal_tier2_score.py` | E | 1 | 11,339 | 11,339 | producer/control dependency of cited tier2 |
| `mutscan.py` | E | 1 | 11,045 | 11,045 | v7 exception resolution |
| `n2_background.py` | E | 1 | 3,460 | 3,460 | producer/control dependency of cited n2 |
| `n2_composition.py` | E | 1 | 1,818 | 1,818 | producer/control dependency of cited n2 |
| `n2_convert.py` | E | 1 | 1,980 | 1,980 | producer/control dependency of cited n2 |
| `n2_demography.py` | E | 1 | 9,860 | 9,860 | pipeline named in REVISION_PREREG.md |
| `n2_mutual.py` | E | 1 | 1,725 | 1,725 | producer/control dependency of cited n2 |
| `n4_scars.py` | E | 1 | 3,549 | 3,549 | pipeline named in REVISION_PREREG.md |
| `n5_path.py` | E | 1 | 7,174 | 7,174 | pipeline named in REVISION_PREREG.md |
| `n6_analyse.sh` | E | 1 | 642 | 642 | pipeline named in REVISION_PREREG.md |
| `n6_score.py` | E | 1 | 3,059 | 3,059 | shell dependency of n6_analyse.sh |
| `nascent.py` | A | 1 | 4,882 | 4,882 | no established active paper dependency; retain in archive |
| `offset_census.py` | E | 1 | 2,834 | 2,834 | pipeline named in REVISION_PREREG.md |
| `offset_census_enc.py` | E | 1 | 3,858 | 3,858 | producer/control dependency of cited offset |
| `offset_encounter.py` | E | 1 | 2,783 | 2,783 | pipeline named in REVISION_PREREG.md |
| `offset_flow.py` | E | 1 | 7,281 | 7,281 | pipeline named in REVISION_PREREG.md |
| `offset_intruder.py` | E | 1 | 3,206 | 3,206 | pipeline named in REVISION_PREREG.md |
| `offset_law.py` | E | 1 | 1,543 | 1,543 | producer/control dependency of cited offset |
| `offset_population.py` | E | 1 | 4,429 | 4,429 | producer/control dependency of cited offset |
| `offset_switch.py` | E | 1 | 8,925 | 8,925 | v7 exception resolution |
| `offset_switch_times.py` | E | 1 | 1,954 | 1,954 | pipeline named in REVISION_PREREG.md |
| `payload_toxin.py` | E | 1 | 14,532 | 14,532 | source bracket manuscript/SI_v8.md |
| `pe_matrix.py` | E | 1 | 5,262 | 5,262 | pipeline named in REVISION_PREREG.md |
| `pilot6502.py` | A | 1 | 7,698 | 7,698 | no established active paper dependency; retain in archive |
| `pilot6502_summary.py` | A | 1 | 2,277 | 2,277 | no established active paper dependency; retain in archive |
| `planning` | E:3, S:7, A:2 | 12 | 377,279 | 377,279 | see module/cohort/archive scope below |
| `population_classes.py` | E | 1 | 11,739 | 11,739 | producer/control dependency of cited population |
| `population_sample.py` | E | 1 | 8,775 | 8,775 | pipeline named in REVISION_PREREG.md |
| `prebuild8086.py` | E | 1 | 3,096 | 3,096 | transitive import/static input or frozen machine contract |
| `prefix_causality8086.py` | E | 1 | 14,399 | 14,399 | transitive import/static input or frozen machine contract |
| `preflight.py` | E | 1 | 9,295 | 9,295 | source bracket manuscript/SI.md |
| `publish_results.sh` | E | 1 | 1,137 | 1,137 | shell dependency of analyze_stage.sh; preserve historical downloader/publisher |
| `py65_difftest.py` | E | 1 | 28,401 | 28,401 | transitive import/static input or frozen machine contract |
| `r4_analyse.py` | E | 1 | 3,129 | 3,129 | transitive import/static input or frozen machine contract |
| `r4_closers.py` | E | 1 | 3,980 | 3,980 | pipeline named in REVISION_PREREG.md |
| `r4_long.py` | A | 1 | 3,953 | 3,953 | no established active paper dependency; retain in archive |
| `reaudit8086.py` | E | 1 | 4,576 | 4,576 | producer/control dependency of cited census8086 |
| `recovery8086.py` | E | 1 | 3,668 | 3,668 | transitive import/static input or frozen machine contract |
| `reforecast8086.py` | E | 1 | 4,366 | 4,366 | transitive import/static input or frozen machine contract |
| `refreeze8086.py` | E | 1 | 14,515 | 14,515 | producer/control dependency of cited census8086 |
| `remaining.py` | A | 1 | 1,233 | 1,233 | no established active paper dependency; retain in archive |
| `report.py` | E | 1 | 12,576 | 12,576 | shell dependency of analyze_stage.sh |
| `requirements-py65.txt` | E | 1 | 81 | 81 | environment or registered pipeline dependency |
| `requirements.txt` | E | 1 | 75 | 75 | environment or registered pipeline dependency |
| `results` | E:1332, S:1169, A:85 | 2586 | 1,361,779,720 | 1,338,930,219 | see module/cohort/archive scope below |
| `ring_sweep.py` | A | 1 | 5,188 | 5,188 | no established active paper dependency; retain in archive |
| `robust_ref22.py` | E | 1 | 6,258 | 6,258 | pipeline named in REVISION_PREREG.md |
| `round2_numbers.py` | E | 1 | 7,644 | 7,644 | producer/control dependency of cited review_r2 |
| `run6502.py` | E | 1 | 8,304 | 8,304 | pipeline named in REVISION_PREREG.md |
| `run8086.py` | E | 1 | 8,120 | 8,120 | pipeline named in REVISION_PREREG.md |
| `run_table.py` | E | 1 | 2,183 | 2,183 | producer/control dependency of cited review_r2 |
| `runs` | E/S/A | 64220 | 26,838,077,366 | 0 | ignored raw/history archive; see every item below |
| `s8086.py` | E | 1 | 9,053 | 9,053 | v7 exception resolution |
| `score6502.py` | E | 1 | 23,821 | 23,821 | pipeline named in REVISION_PREREG.md |
| `score8086.py` | E | 1 | 16,502 | 16,502 | v7 exception resolution |
| `score_L.py` | E | 1 | 8,371 | 8,371 | pipeline named in REVISION_PREREG.md |
| `serial_retention.py` | E | 1 | 15,819 | 15,819 | v7 exception resolution |
| `size_axis.py` | E | 1 | 11,212 | 11,212 | producer/control dependency of cited stageE |
| `spend_sim8086.py` | E | 1 | 21,977 | 21,977 | transitive import/static input or frozen machine contract |
| `sr_kin.py` | E | 1 | 2,538 | 2,538 | transitive import/static input or frozen machine contract |
| `sr_posthoc.py` | E | 1 | 5,005 | 5,005 | transitive import/static input or frozen machine contract |
| `stage_c.py` | E | 1 | 21,381 | 21,381 | producer/control dependency of cited stageC |
| `stage_d.py` | E | 1 | 7,287 | 7,287 | producer/control dependency of cited stageD |
| `stage_e.py` | E | 1 | 20,113 | 20,113 | transitive import/static input or frozen machine contract |
| `stage_f.py` | E | 1 | 12,422 | 12,422 | transitive import/static input or frozen machine contract |
| `stage_g.py` | E | 1 | 13,934 | 13,934 | source bracket manuscript/SI.md |
| `stage_h_analysis.py` | E | 1 | 10,029 | 10,029 | shell dependency of stage_h_pipeline.sh |
| `stage_h_local.py` | E | 1 | 7,244 | 7,244 | transitive import/static input or frozen machine contract |
| `stage_h_pipeline.sh` | E | 1 | 1,191 | 1,191 | environment or registered pipeline dependency |
| `stage_i_analysis.py` | E | 1 | 7,629 | 7,629 | shell dependency of stage_i_pipeline.sh |
| `stage_i_pipeline.sh` | E | 1 | 1,046 | 1,046 | environment or registered pipeline dependency |
| `stage_k_pipeline.sh` | E | 1 | 2,403 | 2,403 | environment or registered pipeline dependency |
| `stage_pipeline.sh` | E | 1 | 754 | 754 | environment or registered pipeline dependency |
| `stats_ci.py` | E | 1 | 5,426 | 5,426 | producer/control dependency of cited review_r2 |
| `stats_tests.py` | E | 1 | 1,825 | 1,825 | transitive import/static input or frozen machine contract |
| `succession.py` | E | 1 | 7,541 | 7,541 | shell dependency of analyze_stage.sh |
| `tests` | E:31 | 32 | 303,132 | 285,549 | see module/cohort/archive scope below |
| `threshold_grid.py` | E | 1 | 8,264 | 8,264 | transitive import/static input or frozen machine contract |
| `tier2_dispatch.py` | E | 1 | 8,751 | 8,751 | pipeline named in REVISION_PREREG.md |
| `tier2_prepare.py` | E | 1 | 23,128 | 23,128 | transitive import/static input or frozen machine contract |
| `tier2_report.py` | E | 1 | 24,236 | 24,236 | source bracket manuscript/SI_v8.md |
| `tier2_status.py` | E | 1 | 3,512 | 3,512 | producer/control dependency of cited tier2 |
| `timing_heritable.py` | A | 1 | 2,436 | 2,436 | no established active paper dependency; retain in archive |
| `trace_z80.py` | A | 1 | 3,661 | 3,661 | no established active paper dependency; retain in archive |
| `two_byte_census.py` | E | 1 | 5,945 | 5,945 | source bracket manuscript/SI.md |
| `volume_probe8086.py` | S | 1 | 8,998 | 8,998 | audit/correction/review/operational provenance |
| `z80_ring_gpu.py` | E | 1 | 5,461 | 5,461 | transitive import/static input or frozen machine contract |
| `zoo.py` | E | 1 | 6,866 | 6,866 | shell dependency of analyze_stage.sh |
| `zsoup_validate.py` | E | 1 | 24,161 | 24,161 | producer/control dependency of cited zsoup |

Root scripts include **175 E, 4 S and 17 A** Python/shell entrypoints. `fetch.sh` and `publish_results.sh` are E dependencies of `analyze_stage.sh`; “publish” here means copying analysis into the tracked results tree, not web deployment. Preserve these originals, but replace overwrite/delete behavior with scratch-output publication in the future pipeline.

### B. Package, manuscript, figure, planning and test sources

This table covers each source/document outside generated output trees. For `machines/`, **all six `worlds*.json` specs, 352 WGSL/manifest pairs, seven builder/provenance files** are E as the frozen contract family. For `shader/`, **all 222 files** are E as historical execution/ISA/export evidence and instrumented derivatives; ignored newly generated trace files need hashes and source origins. Cache files are A/nonresearch. Conditions and tests not proven essential are S; keep their validation provenance.

| Path | Category | Bytes | Basis |
|---|---|---|---|
| `algocell_exp/ASSAYS_ANY_ISA.md` | E | 16,148 | environment or registered pipeline dependency |
| `algocell_exp/ZSOUP.md` | E | 20,407 | source bracket manuscript/SI_v8.md |
| `algocell_exp/__init__.py` | E | 363 | transitive import/static input or frozen machine contract |
| `algocell_exp/assay.py` | E | 20,171 | v7 exception resolution |
| `algocell_exp/batch.py` | E | 7,071 | transitive import/static input or frozen machine contract |
| `algocell_exp/census8086_trace.py` | E | 14,103 | pipeline named in REVISION_PREREG.md |
| `algocell_exp/conv.py` | E | 5,980 | transitive import/static input or frozen machine contract |
| `algocell_exp/exectrace.py` | E | 4,909 | pipeline named in REVISION_PREREG.md |
| `algocell_exp/executor.py` | E | 22,062 | pipeline named in REVISION_PREREG.md |
| `algocell_exp/gen_conv_shader.py` | E | 4,675 | transitive import/static input or frozen machine contract |
| `algocell_exp/gen_dial_shader.py` | E | 4,525 | v7 exception resolution |
| `algocell_exp/gen_lethal_shader.py` | E | 6,089 | transitive import/static input or frozen machine contract |
| `algocell_exp/gen_mixed_shader.py` | E | 4,925 | transitive import/static input or frozen machine contract |
| `algocell_exp/gen_trace_shader.py` | E | 2,650 | transitive import/static input or frozen machine contract |
| `algocell_exp/i8086_provenance_spec.py` | E | 6,145 | transitive import/static input or frozen machine contract |
| `algocell_exp/impact.py` | E | 49,169 | transitive import/static input or frozen machine contract |
| `algocell_exp/impact_assays.py` | E | 12,629 | transitive import/static input or frozen machine contract |
| `algocell_exp/impact_corpus.py` | E | 16,901 | transitive import/static input or frozen machine contract |
| `algocell_exp/impact_worlds.py` | E | 10,835 | transitive import/static input or frozen machine contract |
| `algocell_exp/isa.py` | E | 8,566 | pipeline named in REVISION_PREREG.md |
| `algocell_exp/m6502_instructions.json` | E | 36,207 | transitive import/static input or frozen machine contract |
| `algocell_exp/metrics.py` | E | 9,415 | v7 exception resolution |
| `algocell_exp/optrace.py` | E | 5,417 | transitive import/static input or frozen machine contract |
| `algocell_exp/prng.py` | E | 580 | pipeline named in REVISION_PREREG.md |
| `algocell_exp/py65_ring.py` | E | 4,770 | transitive import/static input or frozen machine contract |
| `algocell_exp/run.py` | E | 20,514 | transitive import/static input or frozen machine contract |
| `algocell_exp/soup.py` | E | 17,119 | v7 exception resolution |
| `algocell_exp/tier2.py` | E | 3,365 | transitive import/static input or frozen machine contract |
| `algocell_exp/zcensus.py` | E | 37,027 | pipeline named in REVISION_PREREG.md |
| `algocell_exp/zmachine.py` | E | 19,797 | pipeline named in REVISION_PREREG.md |
| `algocell_exp/zsoup.py` | E | 24,651 | pipeline named in REVISION_PREREG.md |
| `algocell_exp/zsoup_check.py` | E | 31,972 | transitive import/static input or frozen machine contract |
| `biology/__init__.py` | A | 0 | no established active paper dependency; retain in archive |
| `biology/assembly.py` | E | 41,239 | transitive import/static input or frozen machine contract |
| `biology/individuality.py` | E | 39,244 | transitive import/static input or frozen machine contract |
| `biology/test_assembly.py` | E | 3,083 | test of paper pipeline |
| `conds/pilot_modal.json` | E | 649 | transitive import/static input or frozen machine contract |
| `conds/ring_sweep.json` | E | 64,181 | transitive import/static input or frozen machine contract |
| `conds/stageA.json` | E | 133,775 | transitive import/static input or frozen machine contract |
| `conds/stageA_remaining.json` | E | 97,724 | transitive import/static input or frozen machine contract |
| `conds/stageB.json` | E | 132,146 | transitive import/static input or frozen machine contract |
| `conds/stageC.json` | E | 236,642 | source bracket manuscript/SI.md |
| `conds/stageD.json` | E | 130,142 | source bracket manuscript/SI.md |
| `conds/stageE.json` | E | 588,865 | source bracket manuscript/SI.md |
| `conds/stageF.json` | E | 101,102 | source bracket manuscript/SI.md |
| `conds/stageG.json` | E | 37,242 | source bracket manuscript/SI.md |
| `conds/stageH.json` | E | 4,692 | transitive import/static input or frozen machine contract |
| `conds/stageI.json` | E | 4,732 | transitive import/static input or frozen machine contract |
| `conds/stageK.json` | E | 9,602 | transitive import/static input or frozen machine contract |
| `conds/stageL.json` | E | 10,562 | transitive import/static input or frozen machine contract |
| `conds/stageM.json` | E | 41,342 | transitive import/static input or frozen machine contract |
| `lit_review/PROMPT.md` | S | 16,563 | audit/correction/review/operational provenance |
| `lit_review/VERIFICATION.md` | S | 32,394 | audit/correction/review/operational provenance |
| `lit_review/dr1_alife.md` | S | 80,516 | audit/correction/review/operational provenance |
| `lit_review/dr2_alife.md` | S | 53,416 | audit/correction/review/operational provenance |
| `lit_review/dr3_artificial_life_literature_review.md` | S | 218,157 | audit/correction/review/operational provenance |
| `manuscript/AUDIT_v7.md` | E | 147,345 | number/source audit |
| `manuscript/AUDIT_v7_RESOLUTION.md` | E | 64,833 | source bracket manuscript/SI_v8.md |
| `manuscript/CONCEPTUAL_FIGURES.md` | A | 8,009 | no established active paper dependency; retain in archive |
| `manuscript/CONCEPT_ABSTRACT.md` | A | 2,955 | no established active paper dependency; retain in archive |
| `manuscript/DESIGNER_ROUND5.md` | A | 2,609 | no established active paper dependency; retain in archive |
| `manuscript/DESIGNER_ROUND6.md` | A | 3,653 | no established active paper dependency; retain in archive |
| `manuscript/FIGURES_V6.md` | S | 12,412 | audit/correction/review/operational provenance |
| `manuscript/FIGURES_V7.md` | S | 19,923 | audit/correction/review/operational provenance |
| `manuscript/FIGURES_V8.md` | E | 20,099 | submission reconciliation/current figure manifest |
| `manuscript/LEGENDS_V7.md` | A | 17,106 | no established active paper dependency; retain in archive |
| `manuscript/LEGENDS_V8.md` | E | 8,944 | transitive import/static input or frozen machine contract |
| `manuscript/MAIN.md` | A | 25,901 | no established active paper dependency; retain in archive |
| `manuscript/MAIN_nature.md` | E | 187,277 | v7 exception resolution |
| `manuscript/MAIN_nature_v3_sample.md` | A | 7,584 | no established active paper dependency; retain in archive |
| `manuscript/MAIN_v5.md` | A | 67,058 | no established active paper dependency; retain in archive |
| `manuscript/MAIN_v5_CHANGES.md` | A | 9,970 | no established active paper dependency; retain in archive |
| `manuscript/MAIN_v6.md` | A | 62,788 | no established active paper dependency; retain in archive |
| `manuscript/MAIN_v6_REVIEW.md` | S | 47,913 | audit/correction/review/operational provenance |
| `manuscript/MAIN_v7.md` | E | 92,296 | submission/registration root |
| `manuscript/MAIN_v7_CHANGES.md` | A | 18,227 | no established active paper dependency; retain in archive |
| `manuscript/MAIN_v7_REVIEW.md` | S | 60,252 | audit/correction/review/operational provenance |
| `manuscript/MAIN_v8.md` | E | 81,220 | submission/registration root |
| `manuscript/MAIN_v8_CHANGES.md` | S | 13,475 | audit/correction/review/operational provenance |
| `manuscript/NUMBERS_INDEX.md` | E | 94,537 | transitive import/static input or frozen machine contract |
| `manuscript/ONE_PAGER.md` | A | 5,254 | no established active paper dependency; retain in archive |
| `manuscript/PAPER_PLAN.md` | E | 11,451 | transitive import/static input or frozen machine contract |
| `manuscript/RESPONSE_TO_REVIEWS.md` | S | 8,323 | audit/correction/review/operational provenance |
| `manuscript/RESPONSE_TO_REVIEWS_R2.md` | S | 26,913 | audit/correction/review/operational provenance |
| `manuscript/REWRITE_PLAN.md` | A | 9,985 | no established active paper dependency; retain in archive |
| `manuscript/SI.md` | E | 269,571 | submission/registration root |
| `manuscript/SI_v8.md` | E | 32,468 | submission/registration root |
| `manuscript/STYLE_NOTES.md` | A | 7,901 | no established active paper dependency; retain in archive |
| `manuscript/V4_OUTLINE.md` | A | 5,687 | no established active paper dependency; retain in archive |
| `manuscript/archive/MAIN_nature_v3.md` | E | 81,224 | transitive import/static input or frozen machine contract |
| `manuscript/audit_v7.csv` | E | 381,551 | number/source audit |
| `manuscript/audit_v7.py` | E | 87,168 | number/source audit |
| `manuscript/build_pdf.py` | E | 18,619 | transitive import/static input or frozen machine contract |
| `manuscript/build_si.sh` | E | 1,488 | environment or registered pipeline dependency |
| `manuscript/build_story_pdf.py` | A | 8,278 | no established active paper dependency; retain in archive |
| `manuscript/ed_tables.md` | E | 2,404 | transitive import/static input or frozen machine contract |
| `manuscript/ed_tables.py` | E | 3,019 | transitive import/static input or frozen machine contract |
| `manuscript/figures/CONCEPT_PROMPTS.md` | A | 13,530 | no established active paper dependency; retain in archive |
| `manuscript/figures/DESIGN_BRIEF_REPLY.md` | A | 10,904 | no established active paper dependency; retain in archive |
| `manuscript/figures/DESIGN_NOTES_FOR_3D.md` | A | 12,158 | no established active paper dependency; retain in archive |
| `manuscript/figures/LAYOUT_V6.md` | A | 2,055 | no established active paper dependency; retain in archive |
| `manuscript/figures/archive_erasure_v6.py` | E | 3,848 | current figure/ED producer |
| `manuscript/figures/build_v6.py` | A | 26,399 | no established active paper dependency; retain in archive |
| `manuscript/figures/build_v62.py` | A | 27,073 | no established active paper dependency; retain in archive |
| `manuscript/figures/build_v7.py` | A | 35,202 | no established active paper dependency; retain in archive |
| `manuscript/figures/build_v8.py` | E | 38,202 | v7 exception resolution |
| `manuscript/figures/captions_v6.py` | A | 13,426 | no established active paper dependency; retain in archive |
| `manuscript/figures/concept.py` | E | 22,642 | transitive import/static input or frozen machine contract |
| `manuscript/figures/critique_v6.py` | A | 8,561 | no established active paper dependency; retain in archive |
| `manuscript/figures/critique_v7.py` | A | 4,799 | no established active paper dependency; retain in archive |
| `manuscript/figures/critique_v8.py` | E | 5,123 | transitive import/static input or frozen machine contract |
| `manuscript/figures/fetch_c1_v62.py` | E | 5,828 | current figure/ED producer |
| `manuscript/figures/figcheck.py` | E | 11,230 | transitive import/static input or frozen machine contract |
| `manuscript/figures/figs_n3.py` | E | 24,937 | current figure/ED producer |
| `manuscript/figures/figs_r3.py` | E | 26,744 | current figure/ED producer |
| `manuscript/figures/legends_v62.py` | A | 22,512 | no established active paper dependency; retain in archive |
| `manuscript/figures/legends_v7.py` | A | 19,030 | no established active paper dependency; retain in archive |
| `manuscript/figures/legends_v8.py` | E | 1,320 | current figure/ED producer |
| `manuscript/figures/make_figures.py` | E | 126,385 | current figure/ED producer |
| `manuscript/figures/prepare_v6.py` | E | 9,588 | v7 exception resolution |
| `manuscript/figures/prepare_v62.py` | E | 7,308 | v7 exception resolution |
| `manuscript/figures/prepare_v7.py` | E | 6,737 | v7 exception resolution |
| `manuscript/figures/prepare_v8.py` | E | 5,099 | v7 exception resolution |
| `manuscript/figures/soup_stills.py` | E | 10,000 | transitive import/static input or frozen machine contract |
| `manuscript/figures/soup_video.py` | A | 12,781 | no established active paper dependency; retain in archive |
| `manuscript/figures/verify_v6.py` | A | 2,533 | no established active paper dependency; retain in archive |
| `manuscript/figures/verify_v62.py` | A | 5,912 | no established active paper dependency; retain in archive |
| `manuscript/figures/verify_v7.py` | A | 16,482 | no established active paper dependency; retain in archive |
| `manuscript/figures/verify_v8.py` | E | 11,436 | current figure/ED producer |
| `manuscript/figures/video_run.py` | A | 2,510 | no established active paper dependency; retain in archive |
| `manuscript/figures/write_v62_manifest.py` | A | 12,478 | no established active paper dependency; retain in archive |
| `manuscript/figures/write_v6_manifest.py` | A | 10,265 | no established active paper dependency; retain in archive |
| `manuscript/refcheck.py` | E | 3,050 | transitive import/static input or frozen machine contract |
| `micro/__init__.py` | A | 0 | no established active paper dependency; retain in archive |
| `micro/analyze_bff.sh` | A | 1,828 | no established active paper dependency; retain in archive |
| `micro/bff.py` | E | 17,285 | pipeline named in BIOLOGY_PREREG.md |
| `micro/bff_analysis.py` | E | 23,636 | pipeline named in REVISION_PREREG.md |
| `micro/bff_closed_search.py` | E | 4,838 | pipeline named in manuscript/SI.md |
| `micro/bff_search.py` | E | 6,346 | pipeline named in manuscript/SI.md |
| `micro/bff_soup.py` | E | 10,703 | transitive import/static input or frozen machine contract |
| `micro/launch_bff.sh` | A | 754 | no established active paper dependency; retain in archive |
| `planning/IMPACT_STUDY_DRAFT.md` | E | 6,281 | transitive import/static input or frozen machine contract |
| `planning/ISA_RESEARCH_2026-10-10.md` | E | 21,971 | transitive import/static input or frozen machine contract |
| `planning/LITERAL_CHANNEL.md` | A | 8,101 | no established active paper dependency; retain in archive |
| `planning/PREREG_6502_DRAFT.md` | E | 7,403 | transitive import/static input or frozen machine contract |
| `planning/PREREG_8086_DRAFT.md` | S | 55,232 | audit/correction/review/operational provenance |
| `planning/ZILION_DESIGN_2026-10-10.md` | A | 76,845 | no established active paper dependency; retain in archive |
| `planning/narrative/ASTRA_R1.md` | S | 114,921 | audit/correction/review/operational provenance |
| `planning/narrative/CLAUDE_R1.md` | S | 11,203 | audit/correction/review/operational provenance |
| `planning/narrative/CLAUDE_R1b.md` | S | 4,016 | audit/correction/review/operational provenance |
| `planning/narrative/CLAUDE_R2.md` | S | 7,014 | audit/correction/review/operational provenance |
| `planning/narrative/CLAUDE_R3.md` | S | 13,128 | audit/correction/review/operational provenance |
| `planning/narrative/JOINT_CHARTER.md` | S | 51,164 | audit/correction/review/operational provenance |
| `tests/test_assays_any_isa.py` | E | 11,180 | test of paper pipeline |
| `tests/test_b5_enforcement.py` | E | 17,623 | test of paper pipeline |
| `tests/test_b6_provenance.py` | E | 3,773 | test of paper pipeline |
| `tests/test_bff.py` | E | 9,644 | test of paper pipeline |
| `tests/test_capacity_provenance.py` | E | 2,281 | test of paper pipeline |
| `tests/test_exploratory8086.py` | E | 2,841 | test of paper pipeline |
| `tests/test_impact.py` | E | 22,910 | test of paper pipeline |
| `tests/test_impact_assay_supplement.py` | E | 943 | test of paper pipeline |
| `tests/test_impact_confirm.py` | E | 3,219 | test of paper pipeline |
| `tests/test_impact_i1_report.py` | E | 2,802 | test of paper pipeline |
| `tests/test_impact_q.py` | E | 3,169 | test of paper pipeline |
| `tests/test_impact_registry.py` | E | 3,475 | test of paper pipeline |
| `tests/test_impact_report.py` | E | 2,899 | test of paper pipeline |
| `tests/test_lethal.py` | E | 20,146 | test of paper pipeline |
| `tests/test_meter8086.py` | E | 12,715 | test of paper pipeline |
| `tests/test_mixed.py` | E | 10,881 | test of paper pipeline |
| `tests/test_modal_noninteractive8086.py` | E | 4,140 | test of paper pipeline |
| `tests/test_pipeline.py` | E | 22,957 | test of paper pipeline |
| `tests/test_port.py` | E | 1,795 | test of paper pipeline |
| `tests/test_prefix_causality8086.py` | E | 4,009 | test of paper pipeline |
| `tests/test_py65_ring.py` | E | 7,250 | test of paper pipeline |
| `tests/test_reset_source_stop.py` | E | 1,994 | test of paper pipeline |
| `tests/test_run6502.py` | E | 6,217 | test of paper pipeline |
| `tests/test_s2_final.py` | E | 6,155 | test of paper pipeline |
| `tests/test_s8086.py` | E | 29,686 | test of paper pipeline |
| `tests/test_s8086_repairs.py` | E | 17,499 | test of paper pipeline |
| `tests/test_score6502.py` | E | 11,941 | test of paper pipeline |
| `tests/test_tier2.py` | E | 4,121 | test of paper pipeline |
| `tests/test_volume_parallel8086.py` | E | 11,914 | test of paper pipeline |
| `tests/test_zcensus.py` | E | 5,336 | test of paper pipeline |
| `tests/test_zsoup.py` | E | 20,034 | test of paper pipeline |

Required output/input scopes: current `figures/out/v8/data/**`, main PDF/SVG/PNG and QA manifests are E; its historical `critique/**` is S. Required v6/v7 **data/** and their producers are E. Older rendered images, reviews and layouts are A/S unless referenced by the active builder or SI; `figures/out/fig1.pdf` is specifically E because `build_v8.py::fig1` reads it. Required ignored designer assets (including any `refs/designer_fig1_round3.png` read by the producer) are E; unrelated reference images are S/A, subject to rights. Old manuscript versions/exports are A; v7 audit/reference text remains E or S according to its current source role. `MAIN_nature.md` remains E as a theorem/proof source even when its narrative is superseded.

### B2. Generated manuscript assets and literature cache

Exact compressed scopes below complete the per-item classification of generated manuscript assets. These do not substitute for checking final export existence. Cache/frame directories may be regenerated after their required inputs are preserved.

| Scope | Files | Bytes | Categories |
|---|---|---|---|
| `manuscript/figures/out/` | 1525 | 1,064,737,866 | A:963, E:161, S:401 |
| `manuscript/figures/refs/` | 40 | 17,106,372 | A:37, E:1, S:2 |
| `manuscript/out/` | 47 | 6,642,576 | A:44, S:3 |
| `manuscript/archive/` | 1 | 81,224 | E:1 |
| `library/` | 49 | 68,341,598 | S:49 |

<details>
<summary>Exact asset scopes</summary>

**`manuscript/figures/out/`**

E: `ed_assembly.pdf`, `ed_bffinflow.pdf`, `ed_census.pdf`, `ed_closure.pdf`, `ed_confine.pdf`, `ed_dial.pdf`, `ed_invasions.pdf`, `ed_invasions2.pdf`, `ed_switch.pdf`, `fig1.pdf`, `v6/data/**`, `v6/ed_v6_competition.pdf`, `v6/ed_v6_erasure.pdf`, `v6/ed_v6_joint_table.pdf`, `v6/ed_v6_law.pdf`, `v7/data/**`, `v8/FIGURES_V8.md`, `v8/LEGENDS_V8.md`, `v8/critique/fig1/round3/critique.md`, `v8/critique/fig2/round3/critique.md`, `v8/critique/fig3/round3/critique.md`, `v8/critique/fig4/round3/critique.md`, `v8/critique/fig5/round3/critique.md`, `v8/data/**`, `v8/fig1.pdf`, `v8/fig1.png`, `v8/fig2.pdf`, `v8/fig2.png`, `v8/fig2.svg`, `v8/fig3.pdf`, `v8/fig3.png`, `v8/fig3.svg`, `v8/fig4.pdf`, `v8/fig4.png`, `v8/fig4.svg`, `v8/fig5.pdf`, `v8/fig5.png`, `v8/fig5.svg`, `v8/figcheck.json`, `v8/qa_manifest.json`.

S: `stills/video_L16_st128_k4_s2001.edited_preview.mp4`, `stills/video_L16_st128_k4_s2002.edited_preview.mp4`, `v6/critique/**`, `v6/critique_v62/ed_v6_competition/round1/figcheck.json`, `v6/critique_v62/ed_v6_competition/round2/figcheck.json`, `v6/critique_v62/ed_v6_confinement/round1/figcheck.json`, `v6/critique_v62/ed_v6_confinement/round2/figcheck.json`, `v6/critique_v62/ed_v6_erasure/round1/figcheck.json`, `v6/critique_v62/ed_v6_joint_table/round1/figcheck.json`, `v6/critique_v62/ed_v6_law/round1/figcheck.json`, `v6/critique_v62/ed_v6_reference/round1/figcheck.json`, `v6/critique_v62/ed_v6_reference/round2/figcheck.json`, `v6/critique_v62/fig1/round1/figcheck.json`, `v6/critique_v62/fig1/round2/figcheck.json`, `v6/critique_v62/fig2/round1/figcheck.json`, `v6/critique_v62/fig3/round1/figcheck.json`, `v6/critique_v62/fig3/round2/figcheck.json`, `v6/critique_v62/fig3/round3/figcheck.json`, `v6/critique_v62/fig4/round1/figcheck.json`, `v6/critique_v62/fig4/round2/figcheck.json`, `v6/figcheck.json`, `v6/figcheck_v62.json`, `v7/critique/**`, `v7/figcheck.json`, `v8/critique/fig1/round1/**`, `v8/critique/fig1/round2/**`, `v8/critique/fig1/round3/fig1.png`, `v8/critique/fig1/round3/figcheck.json`, `v8/critique/fig1/round3/manifest.json`, `v8/critique/fig1/round3/prompt.txt`, `v8/critique/fig1/round3/reference1_fig1.png`, `v8/critique/fig1/round3/reference2_fig4v5.png`, `v8/critique/fig1/round3/stderr.txt`, `v8/critique/fig1/round3/stdout.txt`, `v8/critique/fig2/round1/**`, `v8/critique/fig2/round2/**`, `v8/critique/fig2/round3/fig2.png`, `v8/critique/fig2/round3/figcheck.json`, `v8/critique/fig2/round3/manifest.json`, `v8/critique/fig2/round3/prompt.txt`, `v8/critique/fig2/round3/reference1_fig1.png`, `v8/critique/fig2/round3/reference2_fig4v5.png`, `v8/critique/fig2/round3/stderr.txt`, `v8/critique/fig2/round3/stdout.txt`, `v8/critique/fig3/round1/**`, `v8/critique/fig3/round2/**`, `v8/critique/fig3/round3/fig3.png`, `v8/critique/fig3/round3/figcheck.json`, `v8/critique/fig3/round3/manifest.json`, `v8/critique/fig3/round3/prompt.txt`, `v8/critique/fig3/round3/reference1_fig1.png`, `v8/critique/fig3/round3/reference2_fig4v5.png`, `v8/critique/fig3/round3/stderr.txt`, `v8/critique/fig3/round3/stdout.txt`, `v8/critique/fig4/round1/**`, `v8/critique/fig4/round2/**`, `v8/critique/fig4/round3/fig4.png`, `v8/critique/fig4/round3/figcheck.json`, `v8/critique/fig4/round3/manifest.json`, `v8/critique/fig4/round3/prompt.txt`, `v8/critique/fig4/round3/reference1_fig1.png`, `v8/critique/fig4/round3/reference2_fig4v5.png`, `v8/critique/fig4/round3/stderr.txt`, `v8/critique/fig4/round3/stdout.txt`, `v8/critique/fig5/round1/**`, `v8/critique/fig5/round2/**`, `v8/critique/fig5/round3/fig5.png`, `v8/critique/fig5/round3/figcheck.json`, `v8/critique/fig5/round3/manifest.json`, `v8/critique/fig5/round3/prompt.txt`, `v8/critique/fig5/round3/reference1_fig1.png`, `v8/critique/fig5/round3/reference2_fig4v5.png`, `v8/critique/fig5/round3/stderr.txt`, `v8/critique/fig5/round3/stdout.txt`.

A: `ed12.pdf`, `ed12.png`, `ed12.svg`, `ed13.pdf`, `ed13.png`, `ed13.svg`, `ed14.pdf`, `ed14.png`, `ed14.svg`, `ed_assembly.png`, `ed_assembly.svg`, `ed_bffinflow.png`, `ed_bffinflow.svg`, `ed_census.png`, `ed_census.svg`, `ed_closure.png`, `ed_closure.svg`, `ed_confine.png`, `ed_confine.svg`, `ed_dial.png`, `ed_dial.svg`, `ed_invasions.png`, `ed_invasions.svg`, `ed_invasions2.png`, `ed_invasions2.svg`, `ed_switch.png`, `ed_switch.svg`, `fig1.png`, `fig1_km.pdf`, `fig1a_design.pdf`, `fig2.pdf`, `fig2.png`, `fig2.svg`, `fig3.pdf`, `fig3.png`, `fig3.svg`, `fig4.pdf`, `fig4.png`, `fig4.svg`, `fig4v4.pdf`, `fig4v4.png`, `fig4v4.svg`, `fig4v5.pdf`, `fig4v5.png`, `fig4v5.svg`, `fig5.pdf`, `fig5.png`, `fig5.svg`, `fig5v4.pdf`, `fig5v4.png`, `fig5v4.svg`, `fig6.pdf`, `fig6.png`, `fig6.svg`, `fig6v4.pdf`, `fig6v4.png`, `fig6v4.svg`, `fig6v5.pdf`, `fig6v5.png`, `fig6v5.svg`, `fig_assembly.pdf`, `fig_assembly.png`, `fig_assembly.svg`, `figvar.pdf`, `figvar.png`, `figvar.svg`, `stills/frames_bytes_none@closure_L16_st128_k4_s2001/**`, `stills/frames_bytes_video_L16_st128_k4_s2001/**`, `stills/frames_classes_none@closure_L16_st128_k4_s2001/**`, `stills/frames_classes_video_L16_st128_k4_s2001/**`, `stills/none@closure_L16_st128_k4_s2001.bytes.emergence.png`, `stills/none@closure_L16_st128_k4_s2001.bytes.final.png`, `stills/none@closure_L16_st128_k4_s2001.bytes.mp4`, `stills/none@closure_L16_st128_k4_s2001.bytes.t20000.png`, `stills/none@closure_L16_st128_k4_s2001.bytes.t500.png`, `stills/none@closure_L16_st128_k4_s2001.bytes.t5000.png`, `stills/none@closure_L16_st128_k4_s2001.bytes.t50000.png`, `stills/none@closure_L16_st128_k4_s2001.bytes.t75000.png`, `stills/none@closure_L16_st128_k4_s2001.classes.emergence.png`, `stills/none@closure_L16_st128_k4_s2001.classes.final.png`, `stills/none@closure_L16_st128_k4_s2001.classes.mp4`, `stills/none@closure_L16_st128_k4_s2001.classes.t20000.png`, `stills/none@closure_L16_st128_k4_s2001.classes.t500.png`, `stills/none@closure_L16_st128_k4_s2001.classes.t5000.png`, `stills/none@closure_L16_st128_k4_s2001.classes.t50000.png`, `stills/none@closure_L16_st128_k4_s2001.classes.t75000.png`, `stills/sheet_bytes_none@closure_L16_st128_k4_s2001.png`, `stills/sheet_classes_none@closure_L16_st128_k4_s2001.png`, `stills/video_L16_st128_k4_s2001.bytes.emergence.png`, `stills/video_L16_st128_k4_s2001.bytes.final.png`, `stills/video_L16_st128_k4_s2001.bytes.mp4`, `stills/video_L16_st128_k4_s2001.bytes_small.mp4`, `stills/video_L16_st128_k4_s2001.classes.emergence.png`, `stills/video_L16_st128_k4_s2001.classes.final.png`, `stills/video_L16_st128_k4_s2001.classes.mp4`, `stills/video_L16_st128_k4_s2001.edited.mp4`, `stills/video_L16_st128_k4_s2001.edited_720p.mp4`, `stills/video_L16_st128_k4_s2002.edited.mp4`, `stills/video_L16_st128_k4_s2003.edited.mp4`, `v6/LEGENDS_V62.md`, `v6/critique_v62/ed_v6_competition/round1/critique.md`, `v6/critique_v62/ed_v6_competition/round1/ed_v6_competition.png`, `v6/critique_v62/ed_v6_competition/round1/ed_v6_competition_legend.png`, `v6/critique_v62/ed_v6_competition/round1/manifest.json`, `v6/critique_v62/ed_v6_competition/round1/prompt.txt`, `v6/critique_v62/ed_v6_competition/round1/stderr.txt`, `v6/critique_v62/ed_v6_competition/round1/stdout.txt`, `v6/critique_v62/ed_v6_competition/round2/critique.md`, `v6/critique_v62/ed_v6_competition/round2/ed_v6_competition.png`, `v6/critique_v62/ed_v6_competition/round2/ed_v6_competition_legend.png`, `v6/critique_v62/ed_v6_competition/round2/manifest.json`, `v6/critique_v62/ed_v6_competition/round2/prompt.txt`, `v6/critique_v62/ed_v6_competition/round2/stderr.txt`, `v6/critique_v62/ed_v6_competition/round2/stdout.txt`, `v6/critique_v62/ed_v6_confinement/round1/critique.md`, `v6/critique_v62/ed_v6_confinement/round1/ed_v6_confinement.png`, `v6/critique_v62/ed_v6_confinement/round1/ed_v6_confinement_legend.png`, `v6/critique_v62/ed_v6_confinement/round1/manifest.json`, `v6/critique_v62/ed_v6_confinement/round1/prompt.txt`, `v6/critique_v62/ed_v6_confinement/round1/stderr.txt`, `v6/critique_v62/ed_v6_confinement/round1/stdout.txt`, `v6/critique_v62/ed_v6_confinement/round2/critique.md`, `v6/critique_v62/ed_v6_confinement/round2/ed_v6_confinement.png`, `v6/critique_v62/ed_v6_confinement/round2/ed_v6_confinement_legend.png`, `v6/critique_v62/ed_v6_confinement/round2/manifest.json`, `v6/critique_v62/ed_v6_confinement/round2/prompt.txt`, `v6/critique_v62/ed_v6_confinement/round2/stderr.txt`, `v6/critique_v62/ed_v6_confinement/round2/stdout.txt`, `v6/critique_v62/ed_v6_erasure/round1/critique.md`, `v6/critique_v62/ed_v6_erasure/round1/ed_v6_erasure.png`, `v6/critique_v62/ed_v6_erasure/round1/ed_v6_erasure_legend.png`, `v6/critique_v62/ed_v6_erasure/round1/manifest.json`, `v6/critique_v62/ed_v6_erasure/round1/prompt.txt`, `v6/critique_v62/ed_v6_erasure/round1/stderr.txt`, `v6/critique_v62/ed_v6_erasure/round1/stdout.txt`, `v6/critique_v62/ed_v6_joint_table/round1/critique.md`, `v6/critique_v62/ed_v6_joint_table/round1/ed_v6_joint_table.png`, `v6/critique_v62/ed_v6_joint_table/round1/ed_v6_joint_table_legend.png`, `v6/critique_v62/ed_v6_joint_table/round1/manifest.json`, `v6/critique_v62/ed_v6_joint_table/round1/prompt.txt`, `v6/critique_v62/ed_v6_joint_table/round1/stderr.txt`, `v6/critique_v62/ed_v6_joint_table/round1/stdout.txt`, `v6/critique_v62/ed_v6_law/round1/critique.md`, `v6/critique_v62/ed_v6_law/round1/ed_v6_law.png`, `v6/critique_v62/ed_v6_law/round1/ed_v6_law_legend.png`, `v6/critique_v62/ed_v6_law/round1/manifest.json`, `v6/critique_v62/ed_v6_law/round1/prompt.txt`, `v6/critique_v62/ed_v6_law/round1/stderr.txt`, `v6/critique_v62/ed_v6_law/round1/stdout.txt`, `v6/critique_v62/ed_v6_reference/round1/critique.md`, `v6/critique_v62/ed_v6_reference/round1/ed_v6_reference.png`, `v6/critique_v62/ed_v6_reference/round1/ed_v6_reference_legend.png`, `v6/critique_v62/ed_v6_reference/round1/manifest.json`, `v6/critique_v62/ed_v6_reference/round1/prompt.txt`, `v6/critique_v62/ed_v6_reference/round1/stderr.txt`, `v6/critique_v62/ed_v6_reference/round1/stdout.txt`, `v6/critique_v62/ed_v6_reference/round2/critique.md`, `v6/critique_v62/ed_v6_reference/round2/ed_v6_reference.png`, `v6/critique_v62/ed_v6_reference/round2/ed_v6_reference_legend.png`, `v6/critique_v62/ed_v6_reference/round2/manifest.json`, `v6/critique_v62/ed_v6_reference/round2/prompt.txt`, `v6/critique_v62/ed_v6_reference/round2/stderr.txt`, `v6/critique_v62/ed_v6_reference/round2/stdout.txt`, `v6/critique_v62/fig1/round1/critique.md`, `v6/critique_v62/fig1/round1/fig1.png`, `v6/critique_v62/fig1/round1/fig1_legend.png`, `v6/critique_v62/fig1/round1/fig1c.png`, `v6/critique_v62/fig1/round1/fig1d.png`, `v6/critique_v62/fig1/round1/manifest.json`, `v6/critique_v62/fig1/round1/prompt.txt`, `v6/critique_v62/fig1/round1/stderr.txt`, `v6/critique_v62/fig1/round1/stdout.txt`, `v6/critique_v62/fig1/round2/critique.md`, `v6/critique_v62/fig1/round2/fig1.png`, `v6/critique_v62/fig1/round2/fig1_legend.png`, `v6/critique_v62/fig1/round2/fig1c.png`, `v6/critique_v62/fig1/round2/fig1d.png`, `v6/critique_v62/fig1/round2/manifest.json`, `v6/critique_v62/fig1/round2/prompt.txt`, `v6/critique_v62/fig1/round2/stderr.txt`, `v6/critique_v62/fig1/round2/stdout.txt`, `v6/critique_v62/fig2/round1/critique.md`, `v6/critique_v62/fig2/round1/fig2.png`, `v6/critique_v62/fig2/round1/fig2_legend.png`, `v6/critique_v62/fig2/round1/fig2a.png`, `v6/critique_v62/fig2/round1/fig2b.png`, `v6/critique_v62/fig2/round1/fig2c.png`, `v6/critique_v62/fig2/round1/fig2d.png`, `v6/critique_v62/fig2/round1/manifest.json`, `v6/critique_v62/fig2/round1/prompt.txt`, `v6/critique_v62/fig2/round1/stderr.txt`, `v6/critique_v62/fig2/round1/stdout.txt`, `v6/critique_v62/fig3/round1/critique.md`, `v6/critique_v62/fig3/round1/fig3.png`, `v6/critique_v62/fig3/round1/fig3_legend.png`, `v6/critique_v62/fig3/round1/fig3b.png`, `v6/critique_v62/fig3/round1/fig3c.png`, `v6/critique_v62/fig3/round1/fig3d.png`, `v6/critique_v62/fig3/round1/manifest.json`, `v6/critique_v62/fig3/round1/prompt.txt`, `v6/critique_v62/fig3/round1/stderr.txt`, `v6/critique_v62/fig3/round1/stdout.txt`, `v6/critique_v62/fig3/round2/critique.md`, `v6/critique_v62/fig3/round2/fig3.png`, `v6/critique_v62/fig3/round2/fig3_legend.png`, `v6/critique_v62/fig3/round2/fig3b.png`, `v6/critique_v62/fig3/round2/fig3c.png`, `v6/critique_v62/fig3/round2/fig3d.png`, `v6/critique_v62/fig3/round2/manifest.json`, `v6/critique_v62/fig3/round2/prompt.txt`, `v6/critique_v62/fig3/round2/stderr.txt`, `v6/critique_v62/fig3/round2/stdout.txt`, `v6/critique_v62/fig3/round3/critique.md`, `v6/critique_v62/fig3/round3/fig3.png`, `v6/critique_v62/fig3/round3/fig3_legend.png`, `v6/critique_v62/fig3/round3/fig3b.png`, `v6/critique_v62/fig3/round3/fig3c.png`, `v6/critique_v62/fig3/round3/fig3d.png`, `v6/critique_v62/fig3/round3/manifest.json`, `v6/critique_v62/fig3/round3/prompt.txt`, `v6/critique_v62/fig3/round3/stderr.txt`, `v6/critique_v62/fig3/round3/stdout.txt`, `v6/critique_v62/fig4/round1/critique.md`, `v6/critique_v62/fig4/round1/fig4.png`, `v6/critique_v62/fig4/round1/fig4_legend.png`, `v6/critique_v62/fig4/round1/fig4a.png`, `v6/critique_v62/fig4/round1/fig4b.png`, `v6/critique_v62/fig4/round1/fig4c.png`, `v6/critique_v62/fig4/round1/manifest.json`, `v6/critique_v62/fig4/round1/prompt.txt`, `v6/critique_v62/fig4/round1/stderr.txt`, `v6/critique_v62/fig4/round1/stdout.txt`, `v6/critique_v62/fig4/round2/critique.md`, `v6/critique_v62/fig4/round2/fig4.png`, `v6/critique_v62/fig4/round2/fig4_legend.png`, `v6/critique_v62/fig4/round2/fig4a.png`, `v6/critique_v62/fig4/round2/fig4b.png`, `v6/critique_v62/fig4/round2/fig4c.png`, `v6/critique_v62/fig4/round2/manifest.json`, `v6/critique_v62/fig4/round2/prompt.txt`, `v6/critique_v62/fig4/round2/stderr.txt`, `v6/critique_v62/fig4/round2/stdout.txt`, `v6/ed_v6_competition.png`, `v6/ed_v6_competition.svg`, `v6/ed_v6_competition_legend.png`, `v6/ed_v6_confinement.pdf`, `v6/ed_v6_confinement.png`, `v6/ed_v6_confinement.svg`, `v6/ed_v6_confinement_legend.png`, `v6/ed_v6_erasure.png`, `v6/ed_v6_erasure.svg`, `v6/ed_v6_erasure_legend.png`, `v6/ed_v6_joint_table.png`, `v6/ed_v6_joint_table.svg`, `v6/ed_v6_joint_table_legend.png`, `v6/ed_v6_law.png`, `v6/ed_v6_law.svg`, `v6/ed_v6_law_legend.png`, `v6/ed_v6_reference.pdf`, `v6/ed_v6_reference.png`, `v6/ed_v6_reference.svg`, `v6/ed_v6_reference_legend.png`, `v6/fig1.pdf`, `v6/fig1.png`, `v6/fig1.svg`, `v6/fig1_legend.png`, `v6/fig1c.pdf`, `v6/fig1c.png`, `v6/fig1c.svg`, `v6/fig1d.pdf`, `v6/fig1d.png`, `v6/fig1d.svg`, `v6/fig2.pdf`, `v6/fig2.png`, `v6/fig2.svg`, `v6/fig2_legend.png`, `v6/fig2a.pdf`, `v6/fig2a.png`, `v6/fig2a.svg`, `v6/fig2b.pdf`, `v6/fig2b.png`, `v6/fig2b.svg`, `v6/fig2c.pdf`, `v6/fig2c.png`, `v6/fig2c.svg`, `v6/fig2d.pdf`, `v6/fig2d.png`, `v6/fig2d.svg`, `v6/fig3.pdf`, `v6/fig3.png`, `v6/fig3.svg`, `v6/fig3_legend.png`, `v6/fig3b.pdf`, `v6/fig3b.png`, `v6/fig3b.svg`, `v6/fig3c.pdf`, `v6/fig3c.png`, `v6/fig3c.svg`, `v6/fig3d.pdf`, `v6/fig3d.png`, `v6/fig3d.svg`, `v6/fig4.pdf`, `v6/fig4.png`, `v6/fig4.svg`, `v6/fig4_legend.png`, `v6/fig4a.pdf`, `v6/fig4a.png`, `v6/fig4a.svg`, `v6/fig4b.pdf`, `v6/fig4b.png`, `v6/fig4b.svg`, `v6/fig4c.pdf`, `v6/fig4c.png`, `v6/fig4c.svg`, `v6/post_critique_fixes.json`, `v6/qa_manifest.json`, `v6/qa_manifest_v62.json`, `v7/FIGURES_V7.md`, `v7/LEGENDS_V7.md`, `v7/fig1.pdf`, `v7/fig1.png`, `v7/fig2.pdf`, `v7/fig2.png`, `v7/fig2.svg`, `v7/fig3.pdf`, `v7/fig3.png`, `v7/fig3.svg`, `v7/fig4.pdf`, `v7/fig4.png`, `v7/fig4.svg`, `v7/qa_manifest.json`.

**`manuscript/figures/refs/`**

E: `designer_fig1_round3.png`.

S: `compact-figures-v2/compact-v2/DESIGN-AND-REVIEW.txt`, `journal-figures-v1/journal-set/REVIEW-NOTES.txt`.

A: `compact-figures-v2/compact-v2/fig-1a-memory-cleanup.json`, `compact-figures-v2/compact-v2/fig-1a-memory-prompt.json`, `compact-figures-v2/compact-v2/fig-1a-memory.png`, `compact-figures-v2/compact-v2/fig-1b-final-cleanup.json`, `compact-figures-v2/compact-v2/fig-1b-replicator-cleanup.json`, `compact-figures-v2/compact-v2/fig-1b-replicator-prompt.json`, `compact-figures-v2/compact-v2/fig-1b-replicator.png`, `compact-figures-v2/compact-v2/fig-2d-control-flow.png`, `compact-figures-v2/compact-v2/fig-2d-prompt.json`, `compact-figures-v2/compact-v2/fig-2d-qa.json`, `compact-figures-v2/compact-v2/fig-2d-repair-prompt.json`, `compact-figures-v2/compact-v2/fig-4d-prompt.json`, `compact-figures-v2/compact-v2/fig-4d-qa.json`, `compact-figures-v2/compact-v2/fig-4d-worlds.png`, `compact-figures-v2/compact-v2/fig-5a-cycle-proof.png`, `compact-figures-v2/compact-v2/fig-5a-prompt.json`, `compact-figures-v2/compact-v2/fig-5a-qa.json`, `designer_fig1_round4.pdf`, `journal-figures-v1/journal-set/fig-1a-memory-prompt.json`, `journal-figures-v1/journal-set/fig-1a-memory-refinement.json`, `journal-figures-v1/journal-set/fig-1a-memory.png`, `journal-figures-v1/journal-set/fig-1a-refinement2.json`, `journal-figures-v1/journal-set/fig-1b-replicator-prompt.json`, `journal-figures-v1/journal-set/fig-1b-replicator-refinement.json`, `journal-figures-v1/journal-set/fig-1b-replicator.png`, `journal-figures-v1/journal-set/fig-2d-control-flow-draft.png`, `journal-figures-v1/journal-set/fig-2d-fresh-prompt.json`, `journal-figures-v1/journal-set/fig-2d-prompt.json`, `journal-figures-v1/journal-set/fig-2d-qa.json`, `journal-figures-v1/journal-set/fig-2d-repair-prompt.json`, `journal-figures-v1/journal-set/fig-2d-repair2-prompt.json`, `journal-figures-v1/journal-set/fig-4d-prompt.json`, `journal-figures-v1/journal-set/fig-4d-worlds.png`, `journal-figures-v1/journal-set/fig-5a-cycle-theorem.png`, `journal-figures-v1/journal-set/fig-5a-prompt.json`, `threejs-core/**`.

**`manuscript/out/`**

S: `RESPONSE_TO_REVIEWS.log`, `RESPONSE_TO_REVIEWS.pdf`, `RESPONSE_TO_REVIEWS.tex`.

A: `MAIN_nature.log`, `MAIN_nature.pdf`, `MAIN_nature.tex`, `MAIN_nature_reading.log`, `MAIN_nature_reading.pdf`, `MAIN_nature_reading.tex`, `MAIN_v6.log`, `MAIN_v6.pdf`, `MAIN_v6.tex`, `MAIN_v6_read.log`, `MAIN_v6_read.pdf`, `MAIN_v6_read.tex`, `ONE_PAGER.log`, `ONE_PAGER.pdf`, `ONE_PAGER.tex`, `STORY_FOR_NON_EXPERTS.log`, `STORY_FOR_NON_EXPERTS.pdf`, `STORY_FOR_NON_EXPERTS.tex`, `ed12.pdf`, `ed13.pdf`, `ed14.pdf`, `ed_assembly.pdf`, `ed_bffinflow.pdf`, `ed_census.pdf`, `ed_closure.pdf`, `ed_confine.pdf`, `ed_dial.pdf`, `ed_invasions.pdf`, `ed_invasions2.pdf`, `ed_switch.pdf`, `fig1.pdf`, `fig2.pdf`, `fig2_ab.pdf`, `fig2_cde.pdf`, `fig3.pdf`, `fig4.pdf`, `fig4v4.pdf`, `fig4v5.pdf`, `fig5.pdf`, `fig5v4.pdf`, `fig6.pdf`, `fig6v4.pdf`, `fig6v5.pdf`, `fig_bff_slope.pdf`.

**`manuscript/archive/`**

E: `**`.

**`library/`**

S: `**` (49 ignored literature files; retain local originals and a bibliography/hash inventory; redistribute only rights-cleared items).

</details>

### C. Every results cohort and leaf disposition

Counts refer to research files, excluding `__pycache__`. Each row gives E/S/A counts and bytes for the whole item. The subsequent compressed scope inventory explicitly assigns all descendants: `prefix/**` means all files under that directory; otherwise an exact relative filename. These are classifications, not instructions to move files. S retains uncited siblings of essential evidence; A covers disconnected/superseded cohorts.

| Results item | Files | Bytes | Categories |
|---|---|---|---|
| `REPORT_stageA.md` | 1 | 58,615 | A:1 |
| `REPORT_stageB.md` | 1 | 58,174 | A:1 |
| `REPORT_stageC.md` | 1 | 56,849 | A:1 |
| `REPORT_stageD.md` | 1 | 14,862 | A:1 |
| `REPORT_stageE.md` | 1 | 160,198 | A:1 |
| `REPORT_stageF.md` | 1 | 24,211 | A:1 |
| `REPORT_stageG.md` | 1 | 6,792 | A:1 |
| `REPORT_stageH.md` | 1 | 2,626 | A:1 |
| `REPORT_stageI.md` | 1 | 3,322 | A:1 |
| `assays_any_isa` | 8 | 105,161 | S:8 |
| `bff` | 11 | 8,937,991 | E:11 |
| `bff_closed_search` | 3 | 5,281 | E:1, S:2 |
| `bff_dial_hp0.01` | 8 | 759,510 | E:8 |
| `bff_dial_hp0.03` | 8 | 747,643 | E:8 |
| `bff_dial_hp0.1` | 8 | 709,059 | E:8 |
| `bff_dial_hp0.3` | 8 | 1,761,995 | E:8 |
| `bff_dial_r2` | 8 | 537,612 | E:8 |
| `bff_dial_r3` | 8 | 539,641 | E:8 |
| `bff_dials` | 4 | 10,450 | E:4 |
| `bff_search` | 3 | 46,603 | E:1, S:2 |
| `bff_stdnh` | 9 | 806,324 | E:9 |
| `biology` | 26 | 35,496,677 | E:12, S:14 |
| `branching` | 7 | 437,576 | E:6, S:1 |
| `capacity` | 10 | 20,962,526 | E:10 |
| `capacity_time` | 1 | 407,430 | A:1 |
| `capacity_time_L` | 1 | 68,075 | E:1 |
| `census2` | 3 | 32,442 | E:2, S:1 |
| `census6502` | 46 | 9,676,375 | E:45, S:1 |
| `census8086` | 81 | 305,583,002 | E:78, S:3 |
| `closure` | 6 | 337,897 | E:1, S:5 |
| `concept` | 1 | 5,836 | E:1 |
| `conv` | 2 | 12,478 | E:1, S:1 |
| `crystal` | 71 | 75,880,421 | E:71 |
| `detectors` | 10 | 7,511,430 | E:1, S:9 |
| `dial` | 2 | 9,504 | E:1, S:1 |
| `emulator` | 3 | 138,758 | E:1, S:2 |
| `evolved6502` | 13 | 54,149,794 | E:13 |
| `exectrace` | 3 | 47,887 | E:2, S:1 |
| `impact` | 42 | 5,898,917 | E:37, S:5 |
| `impact_dryrun` | 32 | 2,828,664 | E:8, S:24 |
| `impact_equiv` | 4 | 52,170 | E:2, S:2 |
| `inflow_insoup` | 3 | 36,216 | E:3 |
| `invasion` | 10 | 2,088,416 | E:10 |
| `invasion_aligned` | 2 | 203,215 | E:1, S:1 |
| `invasion_closer` | 8 | 517,507 | A:8 |
| `invasion_pair` | 3 | 157,417 | A:3 |
| `ladder` | 18 | 3,282,562 | E:2, S:16 |
| `literal_channel` | 22 | 1,750,974 | E:14, S:8 |
| `lod` | 30 | 8,032,479 | E:9, S:21 |
| `lod_n6` | 16 | 7,722,788 | E:3, S:13 |
| `marker_check` | 2 | 7,120 | E:1, S:1 |
| `mutscan` | 4 | 489,446 | E:3, S:1 |
| `mutscan_I` | 2 | 28,312 | E:2 |
| `mutscan_K` | 2 | 98,010 | E:2 |
| `mutscan_M` | 2 | 173,218 | A:2 |
| `n2` | 11 | 47,105 | E:8, S:3 |
| `n5` | 3 | 387,439 | E:1, S:2 |
| `night` | 1 | 11,950 | A:1 |
| `offset` | 16 | 478,773 | E:14, S:2 |
| `pe` | 2 | 7,281 | E:2 |
| `pilot6502` | 9 | 2,676,407 | A:9 |
| `population` | 6 | 6,183,265 | E:4, S:2 |
| `py65_difftest` | 14 | 16,071,492 | E:1, S:13 |
| `r4` | 10 | 80,686 | E:5, S:5 |
| `review_r2` | 21 | 445,475 | E:14, S:7 |
| `ring_sweep` | 6 | 363,815 | A:6 |
| `s6502` | 265 | 15,961,301 | E:138, S:127 |
| `s8086` | 1025 | 709,269,913 | E:625, S:400 |
| `serial_retention` | 9 | 935,573 | E:7, S:2 |
| `stageA` | 12 | 1,485,456 | A:12 |
| `stageB` | 34 | 3,349,696 | A:34 |
| `stageC` | 62 | 7,149,343 | E:2, S:60 |
| `stageD` | 38 | 2,781,361 | E:2, S:36 |
| `stageE` | 164 | 25,839,177 | E:5, S:159 |
| `stageF` | 33 | 1,643,080 | E:2, S:31 |
| `stageG` | 39 | 1,708,255 | E:9, S:30 |
| `stageH` | 35 | 972,539 | E:3, S:32 |
| `stageI` | 31 | 726,579 | E:5, S:26 |
| `stageK` | 16 | 432,615 | E:2, S:14 |
| `stageL` | 20 | 738,008 | E:2, S:18 |
| `stageM` | 19 | 856,488 | E:2, S:17 |
| `tier2` | 80 | 1,456,874 | E:58, S:22 |
| `toxin` | 7 | 138,359 | E:2, S:5 |
| `video` | 1 | 3,467 | E:1 |
| `zsoup` | 14 | 99,490 | E:1, S:13 |

<details>
<summary>Exact compressed E/S/A scope for all results files</summary>

**`REPORT_stageA.md`**

A: `(this file)`.

**`REPORT_stageB.md`**

A: `(this file)`.

**`REPORT_stageC.md`**

A: `(this file)`.

**`REPORT_stageD.md`**

A: `(this file)`.

**`REPORT_stageE.md`**

A: `(this file)`.

**`REPORT_stageF.md`**

A: `(this file)`.

**`REPORT_stageG.md`**

A: `(this file)`.

**`REPORT_stageH.md`**

A: `(this file)`.

**`REPORT_stageI.md`**

A: `(this file)`.

**`assays_any_isa`**

S: `**`.

**`bff`**

E: `**`.

**`bff_closed_search`**

E: `NUMBERS_CLOSED_SEARCH_p10.md`.

S: `NUMBERS_CLOSED_SEARCH_p8.md`, `candidates_p10.csv`.

**`bff_dial_hp0.01`**

E: `**`.

**`bff_dial_hp0.03`**

E: `**`.

**`bff_dial_hp0.1`**

E: `**`.

**`bff_dial_hp0.3`**

E: `**`.

**`bff_dial_r2`**

E: `**`.

**`bff_dial_r3`**

E: `**`.

**`bff_dials`**

E: `**`.

**`bff_search`**

E: `NUMBERS_SEARCH.md`.

S: `candidates_top200.csv`, `fill_characterisation.csv`.

**`bff_stdnh`**

E: `**`.

**`biology`**

E: `individuality/**`.

S: `assembly/**`.

**`branching`**

E: `BRANCHING.md`, `SUPP_TABLE.md`, `branching_mutants.csv.gz`, `branching_sites.csv`, `branching_summary.csv`, `runtime.txt`.

S: `run.log`.

**`capacity`**

E: `**`.

**`capacity_time`**

A: `**`.

**`capacity_time_L`**

E: `**`.

**`census2`**

E: `NUMBERS_CENSUS2.md`, `fixed_points.csv`.

S: `heritable.csv`.

**`census6502`**

E: `CENSUS.md`, `ckpt/**`, `closers_open_tiles.csv`, `closers_scan.csv`, `control_z80.csv`, `control_z80_words.csv`, `exploratory_rerun.csv`, `m6502_opcodes.json`, `planted_6502.csv`, `table_closers_exact.csv`, `table_closers_law.csv`, `table_consistency.csv`, `table_diagnostics.csv`, `table_exploratory_summary.csv`, `table_hit_words.csv`, `table_hits.csv`, `table_main.csv`, `table_near.csv`, `table_random_partner_rates.csv`, `table_recheck.csv`, `table_signature_fetch.csv`, `table_word_comparison.csv`.

S: `MODAL_TAIL.md`.

**`census8086`**

E: `CENSUS.md`, `DESCRIPTIVE_CORRECTION_2026-10-12.md`, `LAW_CONTRACT.md`, `MECHANISM_CONTRACT.md`, `b6_correction_labels.csv`, `b6_correction_provenance.json`, `bursts.csv`, `correction_labels.csv`, `correction_provenance.json`, `descriptive_correction_counts.csv`, `descriptive_correction_labels.csv`, `descriptive_correction_provenance.json`, `descriptive_other_cases.csv`, `law_alternatives.csv`, `law_eligibility.csv`, `law_erased.csv`, `law_geometry_traces.csv`, `law_serial.csv.gz`, `mechanisms_L16_b128.csv`, `mechanisms_L16_b512.csv`, `mechanisms_L32_b128.csv`, `mechanisms_L32_b512.csv`, `partners.csv`, `provenance.json`, `smoke.json`, `table_audit.csv`, `table_candidates.csv`, `table_consistency.csv`, `table_hashes.csv`, `table_hits.csv`, `table_law.csv`, `table_main.csv`, `table_mechanisms.csv`, `traces_L16_b128_part001.csv.gz`, `traces_L16_b128_part002.csv.gz`, `traces_L16_b128_part003.csv.gz`, `traces_L16_b128_part004.csv.gz`, `traces_L16_b512_part001.csv.gz`, `traces_L16_b512_part002.csv.gz`, `traces_L16_b512_part003.csv.gz`, `traces_L16_b512_part004.csv.gz`, `traces_L16_b512_part005.csv.gz`, `traces_L16_b512_part006.csv.gz`, `traces_L16_b512_part007.csv.gz`, `traces_L16_b512_part008.csv.gz`, `traces_L16_b512_part009.csv.gz`, `traces_L16_b512_part010.csv.gz`, `traces_L16_b512_part011.csv.gz`, `traces_L16_b512_part012.csv.gz`, `traces_L16_b512_part013.csv.gz`, `traces_L16_b512_part014.csv.gz`, `traces_L16_b512_part015.csv.gz`, `traces_L16_b512_part016.csv.gz`, `traces_L16_b512_part017.csv.gz`, `traces_L32_b128_part001.csv.gz`, `traces_L32_b128_part002.csv.gz`, `traces_L32_b512_part001.csv.gz`, `traces_L32_b512_part002.csv.gz`, `traces_L32_b512_part003.csv.gz`, `traces_L32_b512_part004.csv.gz`, `traces_L32_b512_part005.csv.gz`, `traces_L32_b512_part006.csv.gz`, `traces_L32_b512_part007.csv.gz`, `traces_L32_b512_part008.csv.gz`, `traces_L32_b512_part009.csv.gz`, `traces_L32_b512_part010.csv.gz`, `traces_L32_b512_part011.csv.gz`, `traces_L32_b512_part012.csv.gz`, `traces_L32_b512_part013.csv.gz`, `traces_L32_b512_part014.csv.gz`, `words_L16_k1_b128.csv`, `words_L16_k1_b512.csv`, `words_L16_k2_b128.csv`, `words_L16_k2_b512.csv`, `words_L32_k1_b128.csv`, `words_L32_k1_b512.csv`, `words_L32_k2_b128.csv`, `words_L32_k2_b512.csv`.

S: `reset_correction_counts.csv`, `reset_correction_labels.csv`, `reset_correction_provenance.json`.

**`closure`**

E: `NUMBERS_CLOSURE.md`.

S: `closure_partner_independence.pdf`, `closure_partner_independence.png`, `closure_partner_independence.svg`, `closure_runs.csv`, `closure_tapes.csv`.

**`concept`**

E: `**`.

**`conv`**

E: `conv_worlds.csv`.

S: `REPORT.md`.

**`crystal`**

E: `**`.

**`detectors`**

E: `NUMBERS_DETECTORS.md`.

S: `FINDINGS.md`, `detectors_C_by_step.pdf`, `detectors_C_by_step.png`, `detectors_C_by_step.svg`, `detectors_E_by_L.pdf`, `detectors_E_by_L.png`, `detectors_E_by_L.svg`, `samples_C.csv`, `samples_E.csv`.

**`dial`**

E: `dial_worlds.csv`.

S: `DIAL.md`.

**`emulator`**

E: `RING_DIFFTEST.md`.

S: `ring_difftest_seed1.json`, `ring_difftest_seed2.json`.

**`evolved6502`**

E: `**`.

**`exectrace`**

E: `AGREEMENT.md`, `per_replicator.csv`.

S: `FINDINGS.md`.

**`impact`**

E: `README.md`, `affected_pairs.csv`, `assay_changes.csv`, `assay_columns.csv`, `assay_identity.csv`, `assay_scores.csv`, `assay_tapes.csv`, `attribution.csv`, `attribution_classes.csv`, `attribution_totals.csv`, `billing.json`, `billing_source.json`, `c1_c2_differ.csv`, `conditions.csv`, `cores.csv`, `cost.csv`, `cost_estimate.json`, `decisions.csv`, `exclusion.csv`, `execution_source.json`, `executor_usage.csv`, `families.csv`, `flagged_claims.csv`, `input_manifest.json`, `interval_validation.csv`, `limits.csv`, `metrics.csv`, `modal_call.json`, `modal_return.json`, `outcome.csv`, `rates.csv`, `run.csv`, `stop.csv`, `supplement_manifest.json`, `supplement_meta.json`, `unit_table_c0_validation.csv`, `validation.csv`.

S: `.gitignore`, `archive_hashes.csv`, `billing_validation.csv`, `local_gpu_bursts.csv`, `teardown.csv`.

**`impact_dryrun`**

E: `README.md`, `cross_local.json`, `cross_remote.json`, `exclusion.csv`, `input_manifest.json`, `modal_call.json`, `projection.csv`, `proof_summary.csv`.

S: `.gitignore`, `PROTOCOL.md`, `STATUS.md`, `attribution.csv`, `attribution_totals.csv`, `bench.csv`, `bench_raw.csv`, `billing.json`, `billing_rates.json`, `cores.csv`, `done.json`, `dryrun_affected_pairs.csv`, `dryrun_conditions.csv`, `dryrun_meta.json`, `equiv_controls.csv`, `equiv_groups.csv`, `execution_source.json`, `families.csv`, `identity.json`, `metrics.csv`, `projection_conditions.csv`, `rates.csv`, `test_checks.json`, `test_summary.json`.

**`impact_equiv`**

E: `equiv_controls.csv`, `equiv_worlds.csv`.

S: `EQUIV.md`, `equiv_groups.csv`.

**`inflow_insoup`**

E: `**`.

**`invasion`**

E: `**`.

**`invasion_aligned`**

E: `invasion_aligned.csv`.

S: `REPORT.md`.

**`invasion_closer`**

A: `**`.

**`invasion_pair`**

A: `**`.

**`ladder`**

E: `LADDER.md`, `population_worlds.csv`.

S: `atlas_groups.csv`, `atlas_worlds.csv`, `build_ladder.py`, `closure_groups.csv`, `closure_worlds.csv`, `dial_strata.csv`, `effect_sizes.csv`, `inputs.csv`, `lod_groups.csv`, `lod_lines.csv`, `lod_worlds.csv`, `n2_counts.csv`, `n2_groups.csv`, `population_groups.csv`, `s6502_groups.csv`, `s6502_worlds.csv`.

**`literal_channel`**

E: `6502_finite_harness.json`, `CHECK.md`, `census_audit.csv`, `census_provenance.json`, `direct_run.json`, `direct_search.csv`, `direct_witnesses.json`, `finite_bound_counterexamples.json`, `finite_search.csv`, `finite_witness.json`, `minimal_symbolic_search.csv`, `minimal_witnesses.json`, `pair_layouts.csv`, `verification.csv`.

S: `8086_harness_trace.csv`, `audit_census.py`, `direct_search.py`, `finite_window.py`, `minimal_run.json`, `requirements.txt`, `search.py`, `verify.py`.

**`lod`**

E: `FOUNDERS2.md`, `FOUNDING.md`, `FOUNDING_SUMMARY.md`, `LANDSCAPE.md`, `PARTS_FIRST.md`, `REACH.md`, `founders2.csv`, `founding.csv`, `open_first.csv`.

S: `CHAIN_v6.md`, `CHAIN_v6b.md`, `FOUNDER_ENTRY.md`, `NEIGHBOURS.md`, `OPEN_FIRST.md`, `PARTS_v6.md`, `TRAJ_v3.md`, `amended_rule/**`, `chain_v6.csv`, `chain_v6b.csv`, `founders_v6.csv`, `neighbours.csv`, `open_first_classes.csv`, `parts_v6.csv`, `reach.csv`, `registered_rule/**`, `traj_v3.csv`.

**`lod_n6`**

E: `N6_SCORE.md`, `founders2.csv`, `founding.csv`.

S: `CHAIN_v6b.md`, `FOUNDERS2.md`, `FOUNDING.md`, `FOUNDING_SUMMARY.md`, `OPEN_FIRST.md`, `PARTS_FIRST.md`, `PARTS_v6.md`, `REACH.md`, `chain_v6b.csv`, `open_first.csv`, `open_first_classes.csv`, `parts_v6.csv`, `reach.csv`.

**`marker_check`**

E: `marker_check.csv`.

S: `REPORT.md`.

**`mutscan`**

E: `REPORT.md`, `mutscan_sites.csv`, `mutscan_tapes.csv`.

S: `FINDINGS.md`.

**`mutscan_I`**

E: `**`.

**`mutscan_K`**

E: `**`.

**`mutscan_M`**

A: `**`.

**`n2`**

E: `BACKGROUND.md`, `BACKUP.md`, `COMPOSITION_START.md`, `CONVERT.md`, `DEMOGRAPHY.md`, `MUTUAL.md`, `SCARS.md`, `background.csv`.

S: `backup.csv`, `demography.csv`, `scars.csv`.

**`n5`**

E: `PATH.md`.

S: `path_genotypes.csv`, `path_runs.csv`.

**`night`**

A: `**`.

**`offset`**

E: `COPY_PERIOD_LAW.md`, `CORE_FAMILIES.md`, `ENCOUNTER_CENSUS.md`, `INTRUDER.md`, `OFFSET_CENSUS.md`, `OFFSET_ENCOUNTER.md`, `OFFSET_FLOW.md`, `OFFSET_POPULATION.md`, `SWITCH_TIMES.md`, `offset_cells.csv`, `offset_census.csv`, `offset_flow.csv.gz`, `offset_snapshots.csv`, `offset_switch.csv`.

S: `REPORT.md`, `encounter_census.csv`.

**`pe`**

E: `**`.

**`pilot6502`**

A: `**`.

**`population`**

E: `classes_cells.csv`, `classes_snapshots.csv`, `population_cells.csv`, `population_worlds.csv`.

S: `REPORT.md`, `REPORT_classes.md`.

**`py65_difftest`**

E: `examples.json`.

S: `DIFFTEST.md`, `counts.csv`, `coverage.json`, `execution_depth.csv`, `explained.csv`, `groups.csv`, `opcode_coverage.csv`, `s1_missing.json`, `scope_counts.csv`, `summary.json`, `trials.csv.gz`, `unsupported.csv`, `validation.json`.

**`r4`**

E: `R4_CLOSERS.md`, `R4_INVASIONS.md`, `r4_inv.csv`, `r4_long.csv`, `r4_rep.csv`.

S: `R4_LONG.md`, `R4_REP.md`, `R4_REP_CLOSERS.md`, `R4std_cells.csv`, `R4std_classes.csv`.

**`review_r2`**

E: `BFF_WINDOW.md`, `FIRST_CLOSURE.md`, `LINEAGES.md`, `ROUND2_NUMBERS.md`, `RUN_TABLE.md`, `STATS.md`, `STATS_TESTS.md`, `census_p2.csv`, `census_p3.csv`, `check_8080_closer.txt`, `first_closure.csv`, `l50_lineage.txt`, `robust_ref22.csv`, `threshold_grid.csv`.

S: `CENSUS_P2.md`, `CENSUS_P3.md`, `CENSUS_P4.md`, `ROBUST_REF22.md`, `THRESHOLD_GRID.md`, `bff_window.csv`, `census_p4.csv`.

**`ring_sweep`**

A: `**`.

**`s6502`**

E: `AUDIT.md`, `CORRECTION.md`, `CORRECTION_PREREG_TEXT.md`, `S1.md`, `SCORER_CONTROL.md`, `audit_checks.csv`, `audit_cpu.csv`, `audit_culture_sensitivity.csv`, `audit_inputs.csv`, `audit_km.csv`, `audit_mutation_race.csv`, `audit_sources.csv`, `exploratory_classification.csv`, `exploratory_disassembly.csv`, `exploratory_draws.csv`, `exploratory_encounters.csv`, `exploratory_offset_distribution.csv`, `exploratory_offsets.csv`, `exploratory_opcodes.csv`, `exploratory_partners.csv`, `exploratory_provenance.csv`, `exploratory_steps.csv`, `local_control/s1_cells.csv`, `local_control/s1_p2.csv`, `local_control/s1_worlds.csv`, `modal_rates.json`, `s1_cells.csv`, `s1_p2.csv`, `s1_verdicts.json`, `s1_worlds.csv`, `smoke_check.json`, `v2/PROVENANCE.json`, `v2/S1_v2.md`, `v2/arm_comparison.csv`, `v2/audit_findings.csv`, `v2/control/s1_cells.csv`, `v2/control/s1_p2.csv`, `v2/control/s1_worlds.csv`, `v2/control_matches.csv`, `v2/cost.csv`, `v2/cost_summary.json`, `v2/culture_summary.csv`, `v2/final_cultures.csv`, `v2/modal_apps.json`, `v2/modal_calls.json`, `v2/modal_rates.json`, `v2/partner_sensitivity.csv`, `v2/partner_summary.csv`, `v2/s1_cells.csv`, `v2/s1_cells_64.csv`, `v2/s1_p2.csv`, `v2/s1_verdicts.json`, `v2/s1_verdicts_64.json`, `v2/s1_worlds.csv`, `v2/s1_worlds_64.csv`, `v2/tape_comparison.csv`, `v2/verdict_comparison.csv`, `v2/world_comparison.csv`, `v2/worlds/**`.

S: `.gitignore`, `EXPLORATORY.md`, `LAUNCH.md`, `MODAL_SCORER_TEST.md`, `SCORER.md`, `SMOKE.md`, `launch_app_list.json`, `launch_calls.json`, `local_control/LOCAL_CONTROL.md`, `local_control/S1.md`, `local_control/SCORER_CONTROL.md`, `local_control/s1_verdicts.json`, `local_control/worlds/**`, `smoke_launch.json`, `v2/control/MODAL_SCORER_TEST.md`, `v2/control/S1.md`, `v2/control/SCORER_CONTROL.md`, `v2/control/s1_verdicts.json`, `v2/control/worlds/**`, `worlds/**`.

**`s8086`**

E: `EXPLORATORY.md`, `S2.md`, `downloads/**`, `eligibility/**`, `exploratory/**`, `frozen_report/**`, `prelaunch/ASTRA_PRELAUNCH_REVIEW.md`, `prelaunch/ASTRA_RECHECK3.md`, `prelaunch/S2_FINAL_REFREEZE_2026-10-12.md`, `prelaunch/S2_OPERATIONAL_AMENDMENT_2026-10-12.md`, `prelaunch/S2_REGISTRATION_OPTION_B.md`, `prelaunch/S2_SCORING_ONWARD_LOCK_AMENDMENT_2026-10-12.md`, `prelaunch/S2_VOLUME_PARALLEL_AMENDMENT_2026-10-12.md`, `prelaunch/descriptive_intervention_checks.jsonl.gz`, `prelaunch/descriptive_random_validation.json`, `prelaunch/freeze.json`, `prelaunch/guard_limits.json`, `prelaunch/projection.json`, `prelaunch/refreeze_local_bursts.jsonl`, `prelaunch/refreeze_test_runner.py`, `prelaunch/registration_basis.json`, `prelaunch/validation.json`, `provenance/artifact_manifest.json`, `provenance/billing_attempts.json`, `provenance/downloads.json`, `provenance/final_spend.json`, `provenance/final_spend.log`, `provenance/final_spend.py`, `provenance/final_spend_ledger.jsonl`, `provenance/final_validation.json`, `provenance/generate_report.log`, `provenance/generate_report.py`, `provenance/report_steps.json`, `provenance/run/**`, `provenance/run_provenance.json`, `provenance/verification.json`, `tables/**`.

S: `launch/**`, `prelaunch/ASTRA_RECHECK.md`, `prelaunch/ASTRA_RECHECK2.md`, `prelaunch/B5_DYNAMIC_REFREEZE_STOP_2026-10-12.md`, `prelaunch/B5_PROVIDER_LIMIT_STOP_2026-10-12.md`, `prelaunch/B6_CLOSURE_2026-10-12.md`, `prelaunch/CAUSAL_INTERVENTION_STOP_2026-10-12.md`, `prelaunch/FORECAST_OPTIONS_REPAIRED.md`, `prelaunch/FREEZE_REPORT.md`, `prelaunch/FREEZE_REPORT_BEFORE_OPTION_B.md`, `prelaunch/INDEPENDENT_REVIEW.md`, `prelaunch/PREFIX_CAUSAL_STOP_2026-10-12.md`, `prelaunch/RESET_SOURCE_R2_STOP_2026-10-12.md`, `prelaunch/S2_REGISTRATION_BASE.md`, `prelaunch/SPEC_LIMITS.md`, `prelaunch/STOPPED_REPAIR_REPORT.md`, `prelaunch/b5_focused_tests.xml.gz`, `prelaunch/b5_validation.json`, `prelaunch/b6_astra_witness.json`, `prelaunch/b6_instruction_audit.csv`, `prelaunch/b6_local_bursts.jsonl`, `prelaunch/b6_validation.json`, `prelaunch/benchmark.json`, `prelaunch/benchmark_call.json`, `prelaunch/benchmark_failed_attempt.json`, `prelaunch/benchmark_failed_billing.json`, `prelaunch/benchmark_gpu_trace_failed_attempt.json`, `prelaunch/benchmark_gpu_trace_failed_billing.json`, `prelaunch/benchmark_inputs.json`, `prelaunch/benchmark_recovery_call.json`, `prelaunch/benchmark_software_trace_failed_attempt.json`, `prelaunch/benchmark_storage_call.json`, `prelaunch/benchmark_success_billing.json`, `prelaunch/benchmark_trace_call.json`, `prelaunch/billing_final_gpu_private.json`, `prelaunch/billing_final_import.json`, `prelaunch/billing_final_software_private.json`, `prelaunch/billing_final_storage.json`, `prelaunch/causal_stop_L16.manifest.json`, `prelaunch/causal_stop_L16.wgsl`, `prelaunch/causal_stop_L32.manifest.json`, `prelaunch/causal_stop_L32.wgsl`, `prelaunch/causal_stop_N1.manifest.json`, `prelaunch/causal_stop_N1.wgsl`, `prelaunch/causal_stop_summary.json`, `prelaunch/causal_stop_validation.json.gz`, `prelaunch/certificate_L2_source_counterexample.json.gz`, `prelaunch/certificate_L2_source_counterexample_amended.json.gz`, `prelaunch/certificate_REP_L16_d16_a4.json.gz`, `prelaunch/certificate_REP_L16_d16_a4_amended.json.gz`, `prelaunch/certificate_REP_L16_d16_a5.json.gz`, `prelaunch/certificate_REP_L16_d16_a5_amended.json.gz`, `prelaunch/certificate_REP_L16_d8_a4.json.gz`, `prelaunch/certificate_REP_L16_d8_a4_amended.json.gz`, `prelaunch/certificate_REP_L16_d8_a5.json.gz`, `prelaunch/certificate_REP_L16_d8_a5_amended.json.gz`, `prelaunch/certificate_REP_L32_d16_a4.json.gz`, `prelaunch/certificate_REP_L32_d16_a4_amended.json.gz`, `prelaunch/certificate_REP_L32_d16_a5.json.gz`, `prelaunch/certificate_REP_L32_d16_a5_amended.json.gz`, `prelaunch/certificate_REP_L32_d32_a4.json.gz`, `prelaunch/certificate_REP_L32_d32_a4_amended.json.gz`, `prelaunch/certificate_REP_L32_d32_a5.json.gz`, `prelaunch/certificate_REP_L32_d32_a5_amended.json.gz`, `prelaunch/certificate_REP_L32_d8_a4.json.gz`, `prelaunch/certificate_REP_L32_d8_a4_amended.json.gz`, `prelaunch/certificate_REP_L32_d8_a5.json.gz`, `prelaunch/certificate_REP_L32_d8_a5_amended.json.gz`, `prelaunch/certificate_byte_count_dependency.json.gz`, `prelaunch/certificate_byte_count_dependency_amended.json.gz`, `prelaunch/certificate_nonREP.json.gz`, `prelaunch/certificate_nonREP_amended.json.gz`, `prelaunch/cohort.json`, `prelaunch/constructed_control_denominators.json`, `prelaunch/constructed_control_denominators_amended.json`, `prelaunch/constructed_control_validation.json`, `prelaunch/constructed_control_validation_amended.json`, `prelaunch/dependencies.json`, `prelaunch/descriptive_census_stdout.txt`, `prelaunch/descriptive_census_validation.json`, `prelaunch/descriptive_local_bursts.json`, `prelaunch/descriptive_random_bursts.json`, `prelaunch/descriptive_random_stdout.txt`, `prelaunch/failed_resource_bounds.json`, `prelaunch/fixed_L16.npz`, `prelaunch/fixed_L32.npz`, `prelaunch/forecast_options_repaired.json`, `prelaunch/freeze_before_option_b.json`, `prelaunch/input_schemas.json`, `prelaunch/ledger_scale_tests.txt`, `prelaunch/ledger_scale_tests.xml.gz`, `prelaunch/ledger_scale_validation.json`, `prelaunch/local_bursts.jsonl`, `prelaunch/modal_noninteractive_audit_2026-10-12.json`, `prelaunch/observer_hashes.json`, `prelaunch/observer_i8086_pair_maps_L16_storage.manifest.json`, `prelaunch/observer_i8086_pair_maps_L16_storage.wgsl`, `prelaunch/observer_i8086_pair_maps_L32_storage.manifest.json`, `prelaunch/observer_i8086_pair_maps_L32_storage.wgsl`, `prelaunch/operational_fault_simulation_2026-10-12.json`, `prelaunch/operational_gpu_bursts_2026-10-12.jsonl.gz`, `prelaunch/operational_ledger_scale_validation_2026-10-12.json`, `prelaunch/operational_local_bursts_2026-10-12.jsonl.gz`, `prelaunch/operational_parallel_9000_fault_simulation_2026-10-12.json`, `prelaunch/operational_parallel_9000_ledger_scale_2026-10-12.json`, `prelaunch/operational_parallel_9000_publication_probe_2026-10-12.json`, `prelaunch/operational_parallel_9000_serial_simulation_2026-10-12.json`, `prelaunch/operational_parallel_9000_tests_2026-10-12.txt`, `prelaunch/operational_parallel_9000_tests_2026-10-12.xml.gz`, `prelaunch/operational_parallel_9000_validation_2026-10-12.json`, `prelaunch/operational_parallel_9000_volume_probe_2026-10-12.json`, `prelaunch/operational_scoring_onward_validation_2026-10-12.json`, `prelaunch/operational_tests_2026-10-12.txt`, `prelaunch/operational_tests_2026-10-12.xml.gz`, `prelaunch/operational_validation_2026-10-12.json`, `prelaunch/output_size_validation.json`, `prelaunch/prefix_causal_summary.json`, `prelaunch/prefix_causal_validation.json.gz`, `prelaunch/prefix_focus_summary.json`, `prelaunch/prefix_focus_validation.json.gz`, `prelaunch/prefix_intervention_L16.manifest.json`, `prelaunch/prefix_intervention_L16.wgsl`, `prelaunch/prefix_intervention_L32.manifest.json`, `prelaunch/prefix_intervention_L32.wgsl`, `prelaunch/projection_before_option_b.json`, `prelaunch/projection_initial.json`, `prelaunch/projection_initial_original.json`, `prelaunch/refreeze_tests.txt`, `prelaunch/refreeze_tests.xml.gz`, `prelaunch/relaunch_carry_forward_2026-10-12.json`, `prelaunch/relaunch_historical_billing_2026-10-12.json`, `prelaunch/repair_validation.json`, `prelaunch/reset_census_stdout.txt`, `prelaunch/reset_census_validation.json`, `prelaunch/reset_gate_independent.manifest.json`, `prelaunch/reset_gate_independent.wgsl`, `prelaunch/reset_intervention_L16.manifest.json`, `prelaunch/reset_intervention_L16.wgsl`, `prelaunch/reset_intervention_L32.manifest.json`, `prelaunch/reset_intervention_L32.wgsl`, `prelaunch/reset_intervention_checks.jsonl.gz`, `prelaunch/reset_mechanisms_L16_b128.csv`, `prelaunch/reset_stop_focus.json.gz`, `prelaunch/reset_stop_focus_summary.json`, `prelaunch/reset_stop_tests.xml.gz`, `prelaunch/reset_stop_validation.json`, `prelaunch/rng_fixtures.json`, `prelaunch/source12_interventions_amended.jsonl.gz`, `prelaunch/source12_validation_amended.json`, `prelaunch/spend.jsonl`, `prelaunch/spend_simulation.json`, `prelaunch/spend_summary.json`, `prelaunch/validation_before_option_b.json`, `prelaunch/world_panels_L16.npz`, `prelaunch/world_panels_L32.npz`, `provenance/commit_paths.json`.

**`serial_retention`**

E: `REPORT.md`, `kin_summary.csv`, `posthoc_cells.csv`, `posthoc_summary.csv`, `sr_mutants.csv.gz`, `sr_sites.csv`, `sr_summary.csv`.

S: `KIN.md`, `POSTHOC.md`.

**`stageA`**

A: `**`.

**`stageB`**

A: `**`.

**`stageC`**

E: `FINDINGS.md`, `stage_c/NUMBERS_C.md`.

S: `NUMBERS.md`, `ZOO.md`, `assays.csv`, `atlas_L100.pdf`, `atlas_L100.png`, `atlas_L100.svg`, `atlas_L16.pdf`, `atlas_L16.png`, `atlas_L16.svg`, `atlas_L36.pdf`, `atlas_L36.png`, `atlas_L36.svg`, `c4/**`, `cells.csv`, `census/**`, `km_L100.pdf`, `km_L100.png`, `km_L100.svg`, `km_L16.pdf`, `km_L16.png`, `km_L16.svg`, `km_L36.pdf`, `km_L36.png`, `km_L36.svg`, `runs.csv`, `stage_c/C1_nulls_km.pdf`, `stage_c/C1_nulls_km.png`, `stage_c/C1_nulls_km.svg`, `stage_c/C2_mutation.pdf`, `stage_c/C2_mutation.png`, `stage_c/C2_mutation.svg`, `stage_c/C3_forest_st128_k4.pdf`, `stage_c/C3_forest_st128_k4.png`, `stage_c/C3_forest_st128_k4.svg`, `stage_c/C3_forest_st32_k2.pdf`, `stage_c/C3_forest_st32_k2.png`, `stage_c/C3_forest_st32_k2.svg`, `stage_c/c1_nulls.csv`, `stage_c/c2_emergence.csv`, `stage_c/c3_ablations_st128_k4.csv`, `stage_c/c3_ablations_st32_k2.csv`, `stage_c/c5_size.csv`, `stage_c/first_vs_final_none.csv`, `succession.csv`, `succession_cells.csv`, `zoo.csv`.

**`stageD`**

E: `FINDINGS.md`, `stage_d/NUMBERS_D.md`.

S: `NUMBERS.md`, `ZOO.md`, `assays.csv`, `atlas_L9.pdf`, `atlas_L9.png`, `atlas_L9.svg`, `cells.csv`, `census/**`, `census_512/**`, `km_L9.pdf`, `km_L9.png`, `km_L9.svg`, `runs.csv`, `stage_d/D_forest.pdf`, `stage_d/D_forest.png`, `stage_d/D_forest.svg`, `stage_d/cells_D.csv`, `succession.csv`, `succession_cells.csv`, `zoo.csv`.

**`stageE`**

E: `FINDINGS.md`, `assays.csv`, `stage_e/NUMBERS_E.md`, `stage_e/size_arms.csv`, `stage_e/unit_fitness_vs_L.csv`.

S: `NUMBERS.md`, `ZOO.md`, `atlas_L10.pdf`, `atlas_L10.png`, `atlas_L10.svg`, `atlas_L100.pdf`, `atlas_L100.png`, `atlas_L100.svg`, `atlas_L12.pdf`, `atlas_L12.png`, `atlas_L12.svg`, `atlas_L16.pdf`, `atlas_L16.png`, `atlas_L16.svg`, `atlas_L18.pdf`, `atlas_L18.png`, `atlas_L18.svg`, `atlas_L20.pdf`, `atlas_L20.png`, `atlas_L20.svg`, `atlas_L24.pdf`, `atlas_L24.png`, `atlas_L24.svg`, `atlas_L25.pdf`, `atlas_L25.png`, `atlas_L25.svg`, `atlas_L3.pdf`, `atlas_L3.png`, `atlas_L3.svg`, `atlas_L32.pdf`, `atlas_L32.png`, `atlas_L32.svg`, `atlas_L36.pdf`, `atlas_L36.png`, `atlas_L36.svg`, `atlas_L4.pdf`, `atlas_L4.png`, `atlas_L4.svg`, `atlas_L49.pdf`, `atlas_L49.png`, `atlas_L49.svg`, `atlas_L5.pdf`, `atlas_L5.png`, `atlas_L5.svg`, `atlas_L50.pdf`, `atlas_L50.png`, `atlas_L50.svg`, `atlas_L6.pdf`, `atlas_L6.png`, `atlas_L6.svg`, `atlas_L64.pdf`, `atlas_L64.png`, `atlas_L64.svg`, `atlas_L7.pdf`, `atlas_L7.png`, `atlas_L7.svg`, `atlas_L8.pdf`, `atlas_L8.png`, `atlas_L8.svg`, `atlas_L81.pdf`, `atlas_L81.png`, `atlas_L81.svg`, `atlas_L9.pdf`, `atlas_L9.png`, `atlas_L9.svg`, `cells.csv`, `km_L10.pdf`, `km_L10.png`, `km_L10.svg`, `km_L100.pdf`, `km_L100.png`, `km_L100.svg`, `km_L12.pdf`, `km_L12.png`, `km_L12.svg`, `km_L16.pdf`, `km_L16.png`, `km_L16.svg`, `km_L18.pdf`, `km_L18.png`, `km_L18.svg`, `km_L20.pdf`, `km_L20.png`, `km_L20.svg`, `km_L24.pdf`, `km_L24.png`, `km_L24.svg`, `km_L25.pdf`, `km_L25.png`, `km_L25.svg`, `km_L3.pdf`, `km_L3.png`, `km_L3.svg`, `km_L32.pdf`, `km_L32.png`, `km_L32.svg`, `km_L36.pdf`, `km_L36.png`, `km_L36.svg`, `km_L4.pdf`, `km_L4.png`, `km_L4.svg`, `km_L49.pdf`, `km_L49.png`, `km_L49.svg`, `km_L5.pdf`, `km_L5.png`, `km_L5.svg`, `km_L50.pdf`, `km_L50.png`, `km_L50.svg`, `km_L6.pdf`, `km_L6.png`, `km_L6.svg`, `km_L64.pdf`, `km_L64.png`, `km_L64.svg`, `km_L7.pdf`, `km_L7.png`, `km_L7.svg`, `km_L8.pdf`, `km_L8.png`, `km_L8.svg`, `km_L81.pdf`, `km_L81.png`, `km_L81.svg`, `km_L9.pdf`, `km_L9.png`, `km_L9.svg`, `runs.csv`, `stage_e/E1_fraction.pdf`, `stage_e/E1_fraction.png`, `stage_e/E1_fraction.svg`, `stage_e/E1b_fraction_faithful.pdf`, `stage_e/E1b_fraction_faithful.png`, `stage_e/E1b_fraction_faithful.svg`, `stage_e/E2_km_steps.pdf`, `stage_e/E2_km_steps.png`, `stage_e/E2_km_steps.svg`, `stage_e/E2b_km_interactions.pdf`, `stage_e/E2b_km_interactions.png`, `stage_e/E2b_km_interactions.svg`, `stage_e/E3_period.pdf`, `stage_e/E3_period.png`, `stage_e/E3_period.svg`, `stage_e/E4_budget.pdf`, `stage_e/E4_budget.png`, `stage_e/E4_budget.svg`, `stage_e/E5_mutation.pdf`, `stage_e/E5_mutation.png`, `stage_e/E5_mutation.svg`, `stage_e/E8_unit_fitness.pdf`, `stage_e/E8_unit_fitness.png`, `stage_e/E8_unit_fitness.svg`, `stage_e/budget_L100.csv`, `stage_e/mutation_L100.csv`, `succession.csv`, `succession_cells.csv`, `zoo.csv`.

**`stageF`**

E: `FINDINGS.md`, `stage_f/NUMBERS_F.md`.

S: `NUMBERS.md`, `ZOO.md`, `assays.csv`, `cells.csv`, `km_L10.pdf`, `km_L10.png`, `km_L10.svg`, `km_L12.pdf`, `km_L12.png`, `km_L12.svg`, `km_L16.pdf`, `km_L16.png`, `km_L16.svg`, `km_L36.pdf`, `km_L36.png`, `km_L36.svg`, `km_L8.pdf`, `km_L8.png`, `km_L8.svg`, `runs.csv`, `stage_f/F1_periods.pdf`, `stage_f/F1_periods.png`, `stage_f/F1_periods.svg`, `stage_f/F4_deadzone.pdf`, `stage_f/F4_deadzone.png`, `stage_f/F4_deadzone.svg`, `stage_f/pusher_trace.csv`, `stage_f/rings.csv`, `succession.csv`, `succession_cells.csv`, `zoo.csv`.

**`stageG`**

E: `EXACT_COPY.md`, `FINDINGS.md`, `NUMBERS.md`, `assays.csv`, `c4/NUMBERS_C4.md`, `c4/functional.csv`, `exact_copy.csv`, `stageG/NUMBERS_G.md`, `stageG/stage_g_runs.csv`.

S: `CLOSER_CENSUS.md`, `ZOO.md`, `c4/C4_functional_st128_k4.pdf`, `c4/C4_functional_st128_k4.png`, `c4/C4_functional_st128_k4.svg`, `c4/functional_fixed_steps.csv`, `cells.csv`, `closer_census.csv`, `km_L16.pdf`, `km_L16.png`, `km_L16.svg`, `km_L20.pdf`, `km_L20.png`, `km_L20.svg`, `km_L50.pdf`, `km_L50.png`, `km_L50.svg`, `km_L64.pdf`, `km_L64.png`, `km_L64.svg`, `runs.csv`, `stageG/G_heritable_fraction.pdf`, `stageG/G_heritable_fraction.png`, `stageG/G_heritable_fraction.svg`, `stageG/G_partner_independence.pdf`, `stageG/G_partner_independence.png`, `stageG/G_partner_independence.svg`, `succession.csv`, `succession_cells.csv`, `zoo.csv`.

**`stageH`**

E: `COMMANDS.md`, `c4/functional.csv`, `stageH/stage_g_runs.csv`.

S: `FINDINGS.md`, `KIN_CENSUS.md`, `NUMBERS.md`, `NUMBERS_H.md`, `ZOO.md`, `assays.csv`, `c4/C4_functional_st128_k4.pdf`, `c4/C4_functional_st128_k4.png`, `c4/C4_functional_st128_k4.svg`, `c4/NUMBERS_C4.md`, `c4/functional_fixed_steps.csv`, `cells.csv`, `kin_census.csv`, `km_L16.pdf`, `km_L16.png`, `km_L16.svg`, `per_world.csv`, `predictions.csv`, `runs.csv`, `stageH/G_heritable_fraction.pdf`, `stageH/G_heritable_fraction.png`, `stageH/G_heritable_fraction.svg`, `stageH/G_partner_independence.pdf`, `stageH/G_partner_independence.png`, `stageH/G_partner_independence.svg`, `stageH/NUMBERS_G.md`, `stageH_measures.pdf`, `stageH_measures.png`, `stageH_measures.svg`, `succession.csv`, `succession_cells.csv`, `zoo.csv`.

**`stageI`**

E: `assays.csv`, `c4/NUMBERS_C4.md`, `c4/functional.csv`, `c4/functional_fixed_steps.csv`, `stageI/stage_g_runs.csv`.

S: `COMMANDS.md`, `FINDINGS.md`, `NUMBERS.md`, `NUMBERS_I.md`, `TIMING.md`, `ZOO.md`, `c4/C4_functional_st128_k4.pdf`, `c4/C4_functional_st128_k4.png`, `c4/C4_functional_st128_k4.svg`, `cells.csv`, `km_L16.pdf`, `km_L16.png`, `km_L16.svg`, `per_world.csv`, `predictions.csv`, `runs.csv`, `stageI/G_heritable_fraction.pdf`, `stageI/G_heritable_fraction.png`, `stageI/G_heritable_fraction.svg`, `stageI/G_partner_independence.pdf`, `stageI/G_partner_independence.png`, `stageI/G_partner_independence.svg`, `stageI/NUMBERS_G.md`, `succession.csv`, `succession_cells.csv`, `zoo.csv`.

**`stageK`**

E: `SCORING.md`, `stageK/stage_g_runs.csv`.

S: `FINDINGS.md`, `ZOO.md`, `assays.csv`, `c4/**`, `stageK/G_heritable_fraction.pdf`, `stageK/G_heritable_fraction.png`, `stageK/G_heritable_fraction.svg`, `stageK/G_partner_independence.pdf`, `stageK/G_partner_independence.png`, `stageK/G_partner_independence.svg`, `stageK/NUMBERS_G.md`, `zoo.csv`.

**`stageL`**

E: `SCORING.md`, `stageL/stage_g_runs.csv`.

S: `FINDINGS.md`, `ZOO.md`, `assays.csv`, `c4/**`, `scoring_L.csv`, `stageL/G_heritable_fraction.pdf`, `stageL/G_heritable_fraction.png`, `stageL/G_heritable_fraction.svg`, `stageL/G_partner_independence.pdf`, `stageL/G_partner_independence.png`, `stageL/G_partner_independence.svg`, `stageL/NUMBERS_G.md`, `zoo.csv`.

**`stageM`**

E: `assays.csv`, `stageM/stage_g_runs.csv`.

S: `FINDINGS.md`, `SCORING.md`, `ZOO.md`, `c4/**`, `stageM/G_heritable_fraction.pdf`, `stageM/G_heritable_fraction.png`, `stageM/G_heritable_fraction.svg`, `stageM/G_partner_independence.pdf`, `stageM/G_partner_independence.png`, `stageM/G_partner_independence.svg`, `stageM/NUMBERS_G.md`, `zoo.csv`.

**`tier2`**

E: `DISPATCH.json`, `PUBLISHED_VERDICTS.csv`, `T2.md`, `T2_C4c.csv`, `T2_CLAIMS.csv`, `T2_COMMANDS.csv`, `T2_COMPONENTS.csv`, `T2_DZ.csv`, `T2_L.csv`, `T2_SCRIPT_COMPONENTS.csv`, `T2_StageE_L10.csv`, `T2_StageE_L10_worlds.csv`, `T2_StageL_status.csv`, `c1_r4_endpoints.csv`, `published_r4_endpoints.csv`, `run_markers.json`, `score_spend.json`, `scored/**`, `scoring_sources.json`, `stageL_diagnosis.json`.

S: `COST.md`, `DESIGN_FULL.json`, `INVENTORY.md`, `LAUNCH.md`, `REGISTRATION_DRAFT.md`, `SPEND.md`, `VALIDATION.md`, `bench.json`, `control_inputs.json`, `cost.csv`, `l4.json`, `launch_calls.json`, `launch_verification.json`, `machines.csv`, `mature.json`, `run_spend.json`, `score_calls.json`, `score_retry_calls.json`, `smoke.json`, `trace.json`, `validate.json`, `variants.json`.

**`toxin`**

E: `CAUSAL.md`, `COMPOSITION.md`.

S: `causal.csv`, `composition_worlds.csv`, `dial_worlds.csv`, `hazard.csv`, `positions.csv`.

**`video`**

E: `**`.

**`zsoup`**

E: `lockstep.md`.

S: `extraction.csv`, `extraction.md`, `lockstep.csv`, `lockstep_trajectory.csv`, `lockstep_windows.csv`, `nondeterminism.csv`, `nondeterminism.md`, `pairs_equivalence.csv`, `pairs_equivalence.md`, `plumbing.csv`, `plumbing.md`, `throughput.csv`, `throughput.md`.

</details>

### D. Every local runs archive item

All of `runs/` is outside Git. E directories contain required raw/cohort inputs or retained generators/analyses; keep exact canonical members identified by hashes, retaining additional logs/duplicates as S. S directories/files preserve attempts, diagnostics, discovery and archived execution history. A scientific exploratory files are retained under the archive release. Bulk archives need manifests, not Git tracking. No filename-only deduplication is authorized.

| Runs item | Category | Files | Bytes | Basis |
|---|---|---|---|---|
| `.tests_ok` | S | 1 | 0 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `REPORT_stageA.md` | S | 1 | 58,633 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `REPORT_stageA_interim.md` | S | 1 | 20,516 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `REPORT_stageB.md` | S | 1 | 58,246 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `REPORT_stageC.md` | S | 1 | 56,903 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `REPORT_stageC_aborted.md` | S | 1 | 32,472 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `REPORT_stageD.md` | S | 1 | 14,880 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `REPORT_stageE.md` | S | 1 | 160,576 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `REPORT_stageF.md` | S | 1 | 24,256 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `REPORT_stageG.md` | S | 1 | 6,828 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `REPORT_stageH.md` | S | 1 | 2,635 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `REPORT_stageH_dry.md` | S | 1 | 2,335 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `REPORT_stageI.md` | S | 1 | 3,331 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `REPORT_stageI_dry.md` | S | 1 | 1,992 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `bff` | E | 127 | 454,449,041 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `bff_closed_search` | E | 2 | 3,438,583 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `bff_closed_search10` | E | 2 | 63,451,606 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `bff_closed_search10.log` | S | 1 | 1,762 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `bff_modal` | E | 1284 | 2,630,833,653 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `bff_modal_analysis.log` | S | 1 | 14,705 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `bff_modal_analysis2.log` | S | 1 | 10,781 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `bff_modal_launch.log` | S | 1 | 61,311 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `bff_modal_launch2.log` | S | 1 | 61,339 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `bff_modal_launch3.log` | S | 1 | 31,298 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `bff_modal_launch4_lit.log` | S | 1 | 25,323 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `bff_modal_launch5_nh.log` | S | 1 | 27,552 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `bff_modal_launch6_lit_rerun.log` | S | 1 | 1,627 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `bff_search` | E | 3 | 2,531,160 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `bff_search.log` | S | 1 | 3,776 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `bff_smoke_modal.log` | S | 1 | 1,153 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `bff_smoke_modal2.log` | S | 1 | 1,166 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `bff_smoke_modal3.log` | S | 1 | 1,183 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `census2` | E | 4 | 3,867,913 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `census2.log` | S | 1 | 5,109 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `closure` | E | 6 | 337,897 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `closure.log` | S | 1 | 5,944 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `controlflow_first_vs_final.csv` | A | 1 | 10,738 | uncited pilot/sweep/old output; archive |
| `conv` | E | 15 | 3,702,774 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `conv_long` | E | 140 | 44,598,841 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `conv_modal` | S | 162 | 40,602,655 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `conv_rep` | E | 160 | 44,599,017 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `detectors` | E | 9 | 7,507,007 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `detectors.log` | S | 1 | 19,979 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `dial` | E | 360 | 80,094,319 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `dial_smoke` | S | 4 | 660,033 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `ed_regen.log` | S | 1 | 1,290 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `ed_regen2.log` | S | 1 | 983 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `figtest.pdf` | A | 1 | 12,851 | uncited pilot/sweep/old output; archive |
| `figtest.png` | A | 1 | 74,007 | uncited pilot/sweep/old output; archive |
| `figtest.svg` | A | 1 | 8,478 | uncited pilot/sweep/old output; archive |
| `invasion` | E | 9 | 2,081,837 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `invasion.log` | S | 1 | 14,393 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `invasion_closer` | E | 32 | 20,484,096 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `launch_stageF.log` | S | 1 | 35,586 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `launch_stageG.log` | S | 1 | 22,167 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod` | E | 2 | 150,796 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `lod_chain_v6.log` | S | 1 | 16,541 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_chain_v6b.log` | S | 1 | 16,145 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_local` | S | 5 | 3,854,693 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_local_31999.log` | S | 1 | 182 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_modal` | S | 78 | 23,256,102 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_modal_launch.log` | S | 1 | 15,079 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_n6_download.log` | S | 1 | 41 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_n6_modal` | E | 100 | 256,484,198 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `lod_n6_modal_launch.log` | S | 1 | 13,111 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_parts_v6.log` | S | 1 | 19,106 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_smoke` | S | 3 | 334,125 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_v2` | S | 2 | 150,131 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_v2_modal` | S | 63 | 17,769,518 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_v2_modal_launch.log` | S | 1 | 13,139 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_v3_modal` | S | 12 | 22,762,233 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_v3_modal_launch.log` | S | 1 | 14,962 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_v4_modal_launch.log` | S | 1 | 2,409 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_v5_modal_launch.log` | S | 1 | 5,872 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `lod_v6` | E | 2 | 2,883,030 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `lod_v6_modal` | E | 100 | 269,705,747 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `lod_v6_modal_launch.log` | S | 1 | 12,898 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `mutscan` | E | 300 | 6,633,537 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `n2` | E | 40 | 102,240 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `n2_run.log` | S | 1 | 1,601 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `n5` | E | 80 | 25,610,240 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `n5_run.log` | S | 1 | 3,203 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `n5_run2.log` | S | 1 | 10,277 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `n6_analyse.log` | S | 1 | 55,284 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `offset` | E | 402 | 101,680,419 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `offset_smoke` | S | 2 | 324,451 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `pilot` | A | 3 | 87,442 | uncited pilot/sweep/old output; archive |
| `pilot6502` | A | 32 | 7,030,805 | uncited pilot/sweep/old output; archive |
| `pilot_modal` | A | 6 | 153,846 | uncited pilot/sweep/old output; archive |
| `pilot_modal_summaries.json` | S | 1 | 5,270 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `preflight2_stageC.log` | S | 1 | 467 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `preflight2_stageD.log` | S | 1 | 404 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `preflight2_stageE.log` | S | 1 | 657 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `preflight_stageC.log` | S | 1 | 426 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `preflight_stageD.log` | S | 1 | 364 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `preflight_stageE.log` | S | 1 | 531 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `preflight_stageF.log` | S | 1 | 466 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `preflight_stageG.log` | S | 1 | 417 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `r4_inv` | E | 64 | 11,123,134 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `r4_smoke` | S | 2 | 322,057 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `relaunch_logs_a.txt` | S | 1 | 668 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `relaunch_logs_ap-AnN8Je9kdKmIAGlrQ4nVo3.txt` | S | 1 | 553 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `relaunch_logs_ap-PKiB4t0eiwE6Gz5D7EWvAr.txt` | S | 1 | 3,132 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `relaunch_logs_ap-k7gncn7pSzIC9M82dctWpz ap-PKiB4t0eiwE6Gz5D7EWvAr.txt` | S | 1 | 638 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `relaunch_logs_b.txt` | S | 1 | 0 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `ring_sweep` | A | 2008 | 283,060,587 | uncited pilot/sweep/old output; archive |
| `ring_sweep.log` | S | 1 | 35,983 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `s6502_raw` | E | 83 | 151,010,682 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `smoke_stageD` | S | 16 | 1,462,994 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `smoke_stageD_assay.log` | S | 1 | 702 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `smoke_stageD_succession.log` | S | 1 | 1,028 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `smoke_stageD_summaries.json` | S | 1 | 13,372 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `smoke_stageD_zoo.log` | S | 1 | 1,162 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `smoke_stageF.log` | S | 1 | 4,814 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `smoke_stageF_summaries.json` | S | 1 | 13,543 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `smoke_stageG.log` | S | 1 | 4,936 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `smoke_stageG_summaries.json` | S | 1 | 14,779 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `smoke_stageK_summaries.json` | S | 1 | 14,478 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageA` | S | 2179 | 532,176,622 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageA_analyze2.log` | S | 1 | 14,186 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageA_assay2.log` | S | 1 | 100,152 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageA_findings2.log` | S | 1 | 38 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageA_launch.log` | S | 1 | 13,403 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageA_report2.log` | S | 1 | 28 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageA_resume.log` | S | 1 | 49,902 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageA_succession2.log` | S | 1 | 18,070 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageA_summaries.json` | S | 1 | 965,234 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageA_zoo2.log` | S | 1 | 489 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageB` | S | 2302 | 652,072,360 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageB_analyze2.log` | S | 1 | 13,237 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageB_assay.log` | S | 1 | 107,941 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageB_assay2.log` | S | 1 | 101,733 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageB_findings2.log` | S | 1 | 38 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageB_launch.log` | S | 1 | 70,139 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageB_report2.log` | S | 1 | 28 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageB_size2.log` | S | 1 | 12,598 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageB_succession2.log` | S | 1 | 17,622 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageB_summaries.json` | S | 1 | 1,682,978 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageB_zoo2.log` | S | 1 | 514 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC` | E | 9797 | 3,357,063,435 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `stageC_aborted` | S | 1479 | 122,227,798 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_aborted_analyze.log` | S | 1 | 7,939 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_aborted_assay.log` | S | 1 | 71,556 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_aborted_report.log` | S | 1 | 28 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_aborted_succession.log` | S | 1 | 8,052 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_aborted_zoo.log` | S | 1 | 1,678 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_analyze.log` | S | 1 | 11,277 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_assay.log` | S | 1 | 77,891 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_c4.log` | S | 1 | 20,862 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_chain.log` | S | 1 | 102 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_fetch.log` | S | 1 | 83 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_findings.log` | S | 1 | 38 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_launch.log` | S | 1 | 38,638 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_launch2.log` | S | 1 | 19,306 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_launch3.log` | S | 1 | 62,896 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_report.log` | S | 1 | 28 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_succession.log` | S | 1 | 14,382 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_summaries.json` | S | 1 | 3,414,823 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageC_zoo.log` | S | 1 | 503 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageD` | E | 5566 | 1,497,435,063 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `stageD_analyze.log` | S | 1 | 3,057 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageD_assay.log` | S | 1 | 43,028 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageD_chain.log` | S | 1 | 172 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageD_findings.log` | S | 1 | 38 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageD_launch.log` | S | 1 | 19,543 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageD_launch2.log` | S | 1 | 38,866 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageD_report.log` | S | 1 | 28 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageD_succession.log` | S | 1 | 4,288 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageD_summaries.json` | S | 1 | 1,907,474 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageD_v1_seeds1-20` | S | 56 | 14,261,758 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageD_zoo.log` | S | 1 | 478 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageE` | E | 24389 | 12,387,014,759 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `stageE_analyze.log` | S | 1 | 31,503 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageE_assay.log` | S | 1 | 202,970 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageE_chain.log` | S | 1 | 172 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageE_findings.log` | S | 1 | 38 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageE_launch.log` | S | 1 | 141,744 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageE_report.log` | S | 1 | 28 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageE_stage_e.log` | S | 1 | 29,484 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageE_succession.log` | S | 1 | 37,884 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageE_summaries.json` | S | 1 | 8,974,818 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageE_zoo.log` | S | 1 | 476 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageF` | E | 4158 | 1,538,704,879 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `stageF_analyze.log` | S | 1 | 4,735 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageF_assay.log` | S | 1 | 34,307 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageF_chain.log` | S | 1 | 12,112 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageF_findings.log` | S | 1 | 38 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageF_report.log` | S | 1 | 28 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageF_stage_f.log` | S | 1 | 16,223 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageF_succession.log` | S | 1 | 6,693 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageF_summaries.json` | S | 1 | 1,462,783 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageF_zoo.log` | S | 1 | 461 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageG` | E | 1713 | 727,081,626 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `stageG_analyze.log` | S | 1 | 962 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageG_assay.log` | S | 1 | 12,275 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageG_c4.log` | S | 1 | 3,781 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageG_chain.log` | S | 1 | 158 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageG_findings.log` | S | 1 | 38 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageG_report.log` | S | 1 | 28 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageG_stage_g.log` | S | 1 | 1,599 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageG_stage_g2.log` | S | 1 | 1,908 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageG_succession.log` | S | 1 | 1,680 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageG_summaries.json` | S | 1 | 575,084 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageG_zoo.log` | S | 1 | 215 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH` | E | 225 | 46,534,259 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `stageH_analyze.log` | S | 1 | 605 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_assay.log` | S | 1 | 2,070 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_c4.log` | S | 1 | 874 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_dry` | S | 31 | 1,083,909 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_dry.log` | S | 1 | 576 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_dry_analyze.log` | S | 1 | 647 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_dry_assay.log` | S | 1 | 479 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_dry_c4.log` | S | 1 | 219 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_dry_findings.log` | S | 1 | 80 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_dry_report.log` | S | 1 | 70 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_dry_stage_g.log` | S | 1 | 545 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_dry_succession.log` | S | 1 | 795 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_dry_zoo.log` | S | 1 | 158 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_findings.log` | S | 1 | 38 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_launch.log` | S | 1 | 2,188 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_report.log` | S | 1 | 28 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_stage_g.log` | S | 1 | 503 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_succession.log` | S | 1 | 837 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageH_zoo.log` | S | 1 | 116 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI` | E | 215 | 68,128,122 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `stageI_analyze.log` | S | 1 | 607 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_assay.log` | S | 1 | 2,398 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_c4.log` | S | 1 | 1,209 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_dry` | S | 30 | 1,623,220 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_dry.log` | S | 1 | 630 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_dry_analyze.log` | S | 1 | 617 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_dry_assay.log` | S | 1 | 656 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_dry_c4.log` | S | 1 | 393 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_dry_findings.log` | S | 1 | 80 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_dry_report.log` | S | 1 | 70 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_dry_stage_g.log` | S | 1 | 670 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_dry_succession.log` | S | 1 | 798 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_dry_zoo.log` | S | 1 | 126 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_findings.log` | S | 1 | 38 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_launch.log` | S | 1 | 2,648 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_report.log` | S | 1 | 28 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_stage_g.log` | S | 1 | 628 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_succession.log` | S | 1 | 768 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageI_zoo.log` | S | 1 | 590 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageK` | E | 454 | 191,583,466 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `stageK_assay.log` | S | 1 | 3,813 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageK_c4.log` | S | 1 | 1,548 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageK_findings.log` | S | 1 | 1,562 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageK_stage_g.log` | S | 1 | 510 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageK_summaries.json` | S | 1 | 144,478 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageK_zoo.log` | S | 1 | 124 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageL` | E | 555 | 247,663,860 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `stageL_assay.log` | S | 1 | 3,951 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageL_c4.log` | S | 1 | 1,671 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageL_findings.log` | S | 1 | 1,562 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageL_stage_g.log` | S | 1 | 510 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageL_summaries.json` | S | 1 | 139,703 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageL_zoo.log` | S | 1 | 156 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageM` | E | 857 | 298,155,705 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `stageM_assay.log` | S | 1 | 7,431 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageM_c4.log` | S | 1 | 3,171 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageM_findings.log` | S | 1 | 1,562 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageM_stage_g.log` | S | 1 | 510 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageM_summaries.json` | S | 1 | 335,994 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `stageM_zoo.log` | S | 1 | 156 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `tests.log` | S | 1 | 100 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `tests_after_CE.log` | S | 1 | 113 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `unit_fitness_vs_ring.csv` | A | 1 | 4,511 | uncited pilot/sweep/old output; archive |
| `video` | E | 4272 | 543,689,061 | canonical raw/cohort dependency; supporting duplicates/logs retained |
| `video_run.log` | S | 1 | 148 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `video_run2.log` | S | 1 | 150 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `video_run3.log` | S | 1 | 152 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `video_run_s2002.log` | S | 1 | 152 | archived logs, attempts, discovery/diagnostic or older cohort copy |
| `video_run_s2003.log` | S | 1 | 152 | archived logs, attempts, discovery/diagnostic or older cohort copy |

### E. Every Modal volume prefix

Volume: `algocell-atlas-runs`, recursively listed via the read-only client API; no download, compute, create, commit, probe, or delete operation. Prefix category applies to canonical scientific members; `_attempts`, operational publication reservations, failed/recovered attempts and launch logs remain S. Never count those as additional worlds.

| Remote prefix | Category | Files | Bytes |
|---|---|---|---|
| `bff/` | E | 808 | 2,022,055,670 |
| `bff_dial_hp0.01/` | E | 60 | 12,035,679 |
| `bff_dial_hp0.03/` | E | 60 | 11,990,262 |
| `bff_dial_hp0.1/` | E | 60 | 11,921,362 |
| `bff_dial_hp0.3/` | E | 60 | 50,155,933 |
| `bff_dial_r2/` | E | 60 | 21,793,980 |
| `bff_dial_r3/` | E | 60 | 19,586,624 |
| `bff_smoke/` | S | 28 | 505,514 |
| `bff_stdnh/` | E | 108 | 472,369,888 |
| `census6502/` | E | 2 | 2,208,604 |
| `conv/` | E | 162 | 40,602,655 |
| `conv_long/` | E | 140 | 44,598,841 |
| `conv_rep/` | E | 160 | 44,599,017 |
| `dial/` | E | 360 | 80,094,319 |
| `dial_dry/` | S | 3 | 330,030 |
| `impact_dryrun/` | E | 864 | 28,599,049 |
| `impact_i1/` | E | 1875 | 1,577,002,109 |
| `lod/` | E | 63 | 18,648,547 |
| `lod_n6/` | E | 100 | 256,484,198 |
| `lod_v2/` | S | 60 | 16,970,307 |
| `lod_v3/` | S | 58 | 191,579,675 |
| `lod_v4/` | S | 9 | 230,040 |
| `lod_v5/` | S | 20 | 527,410 |
| `lod_v6/` | E | 100 | 269,705,747 |
| `offset/` | E | 222 | 58,139,125 |
| `pilot6502/` | A | 40 | 9,705,938 |
| `pilot_modal/` | A | 6 | 153,846 |
| `r4_inv/` | E | 64 | 11,123,134 |
| `s6502/` | E | 480 | 215,333,788 |
| `s6502_scored/` | E | 80 | 649,721 |
| `s6502_scored_v2/` | E | 80 | 3,775,456 |
| `s6502_scorer_test/` | S | 170 | 26,542,396 |
| `s6502_smoke/` | S | 12 | 2,943,899 |
| `s8086/` | E | 319 | 84,553,109 |
| `s8086_assay/` | E | 40 | 2,080 |
| `s8086_controls/` | E | 11 | 83,415,665 |
| `s8086_extract/` | E | 40 | 40,202,198 |
| `s8086_score/` | E | 40 | 79,395,840 |
| `s8086_volume_probes/` | S | 32 | 37,850 |
| `smoke_stageD/` | S | 12 | 1,460,350 |
| `smoke_stageF/` | S | 11 | 1,497,232 |
| `smoke_stageG/` | S | 12 | 2,900,221 |
| `smoke_stageK/` | S | 12 | 2,539,609 |
| `stageA/` | S | 2167 | 530,691,166 |
| `stageB/` | S | 2269 | 648,734,863 |
| `stageC/` | E | 9736 | 3,349,927,925 |
| `stageD/` | E | 5529 | 1,494,661,543 |
| `stageE/` | E | 24226 | 12,361,187,124 |
| `stageF/` | E | 4126 | 1,537,070,807 |
| `stageG/` | E | 1679 | 725,392,179 |
| `stageK/` | E | 440 | 191,154,300 |
| `stageL/` | E | 538 | 246,943,385 |
| `stageM/` | E | 840 | 297,302,176 |
| `tier2/` | E | 2103 | 503,032,952 |
| `tier2_scores/` | E | 75 | 1,700,884 |
| `tier2_smoke/` | S | 23 | 2,923,743 |

### F. App-side dependency closure and preserved boundary

| Original path (outside experiments) | Bytes | Disposition | Local imports |
|---|---|---|---|
| `scripts/export-sim.ts` | 4,381 | E paper dev/export harness; retain app copy until independent | `src/lib/gpu/shaders.ts`, `src/lib/sim/constants.ts`, `src/lib/z80-opcodes.ts` |
| `src/lib/dev/z80difftest_ring.node.ts` | 17,106 | E paper dev/export harness; retain app copy until independent | `src/lib/gpu/shaders.ts`, `src/lib/sim/constants.ts`, `src/lib/dev/z80oracle.ts`, `src/lib/dev/z80difftest_ring.ts` |
| `src/lib/dev/z80difftest_ring.ts` | 30,216 | E paper dev/export harness; retain app copy until independent | `src/lib/dev/z80oracle.ts`, `src/lib/sim/constants.ts`, `src/lib/z80-disasm.ts` |
| `src/lib/dev/z80oracle.ts` | 4,569 | E paper dev/export harness; retain app copy until independent | `src/lib/z80-opcodes.ts`, `src/lib/sim/constants.ts` |
| `src/lib/gpu/shaders.ts` | 37,571 | E historical paper copy; W live app copy | `src/lib/sim/constants.ts` |
| `src/lib/sim/constants.ts` | 2,665 | E historical paper copy; W live app copy | none |
| `src/lib/z80-disasm.ts` | 7,403 | E historical paper copy; W live app copy | none |
| `src/lib/z80-opcodes.ts` | 16,653 | E historical paper copy; W live app copy | none |

The development driver is explicitly documented as never imported by the application. Shared shaders/constants/opcode/disassembly modules remain W in the live app. Copy only the exact historical TS closure and minimal toolchain into the paper, with original source revision and file hashes. Algocell continues to use Zilion as its own application dependency. This task did not inspect or edit any Svelte component.

### G. Observed installed Python freeze

Captured with the prescribed interpreter using `python -m pip freeze`. This is evidence for a lock, not a claim that every package belongs in every platform environment.

```text
aiohappyeyeballs==2.7.1
aiohttp==3.14.4
aiosignal==1.4.0
anyio==4.15.1
attrs==26.1.0
brotli==1.2.0
cbor2==6.1.5
certifi==2026.7.22
cffi==2.1.1
click==8.5.0
contourpy==1.4.0
cycler==0.12.1
fonttools==4.66.1
frozenlist==1.8.0
grpclib==0.4.9
h2==4.4.1
hpack==4.2.0
hyperframe==6.1.0
idna==3.20
iniconfig==2.3.1
kiwisolver==1.5.1
markdown-it-py==4.2.0
matplotlib==3.11.2
mdurl==0.1.2
modal==1.6.1
multidict==6.9.1
numpy==2.5.3
packaging==26.3
pandas==3.0.6
pillow==12.3.0
pluggy==1.6.0
propcache==0.5.4
protobuf==7.36.2
py65==1.2.0
pycparser==3.0
Pygments==2.21.0
pymupdf==1.28.2
pyparsing==3.3.3
pytest==9.1.1
python-dateutil==2.9.0.post0
rendercanvas==2.7.2
rich==15.0.0
rubicon-objc==0.5.7
six==1.17.0
synchronicity==0.12.6
tabulate==0.10.0
toml==0.10.2
types-certifi==2021.10.8.3
types-toml==0.10.8.20260518
typing_extensions==4.16.0
watchfiles==1.3.0
wgpu==0.32.0
yarl==1.25.1
```

### H. Source-root inventory and read-only check record

| Root | Bracket occurrences | Distinct resolved local files |
|---|---|---|
| `manuscript/MAIN_v7.md` | 122 | 1087 |
| `manuscript/MAIN_v8.md` | 136 | 869 |
| `manuscript/SI.md` | 70 | 25 |
| `manuscript/SI_v8.md` | 72 | 278 |
| `REVISION_PREREG.md` | 24 | 0 |
| `BIOLOGY_PREREG.md` | 0 | 0 |
| `PLAN.md` | 1 | 0 |
| `manuscript/AUDIT_v7_RESOLUTION.md` | 16 | 0 |

Counts include bibliography/other brackets in the occurrence total and expanded directory citations in resolved files. Registration scientific/code references generally appear outside square brackets; named-script and producer/import closure are applied separately. The per-row resolution ledger uses a Markdown table and backticks rather than ordinary source brackets, and was additionally parsed as explicit evidence paths.

Original audit manuscript SHA-256: `c4fd7b5e15d24a993b8a4a328735241c935383ea45857fe915683dcccd906866`. V8/SI are concurrently edited working-tree documents; pin their final accepted hashes during migration, rather than using a sampled working copy as a commit identity.

| Read-only verification | Outcome |
|---|---|
| 352 vendored manifest/WGSL SHA-256 comparisons | 352 matched; no generation performed |
| 43 v8 source rows / 40 plotting-table hash rows | All matched current bytes |
| v6 source manifest (including external baseline/cache logs) | All matched at inventory time |
| Original v7 exception-ledger ID coverage | 154 unique entries; exactly the 154 flagged IDs; no missing/extra IDs |
| Current full v8 technical figure check | Failed: missing out/v8/fig1.svg; no figure outputs altered |
| Historical external architecture/8086 doc audit fingerprints | Current sibling docs differ; recover pinned originals |
| Raw Modal metadata listing | 60,714 files; 27,709,689,964 bytes; hashes/unique union not established |
| Full raw-to-number/figure reproduction | Not run: this task authorizes inventory and planning only |

To repeat the metadata measurement without starting a Modal function:

```python
from collections import defaultdict
from pathlib import Path
import os, modal
root = Path("/Users/neo/repos/algocell/experiments")
for item in sorted(root.iterdir()):
    paths = [item] if item.is_file() else [Path(d) / f for d, _, fs in os.walk(item) for f in fs]
    stats = [p.stat() for p in paths if p.is_file()]
    print(item.name, len(stats), sum(s.st_size for s in stats),
          sum(s.st_blocks * 512 for s in stats))
volume = modal.Volume.from_name("algocell-atlas-runs", create_if_missing=False)
prefixes = defaultdict(lambda: [0, 0])
for entry in volume.iterdir("/", recursive=True):
    if int(entry.type) == 1:  # regular file in Modal 1.6.1
        prefix = entry.path.strip("/").split("/")[0]
        prefixes[prefix][0] += 1
        prefixes[prefix][1] += entry.size
for prefix, (files, size) in sorted(prefixes.items()):
    print(prefix, files, size)
```

At migration, preserve this listing as a timestamped CSV/JSON plus full raw-content hashes. Rebuild essentiality by parsing the cited roots and audit evidence paths, adding figure source/hash manifest edges, AST import edges and producer/shell edges, and computing transitive reachability. Persist each edge with its reason in the new repository. The complete classification rules and exact result scopes above are the reviewable output of that analysis; transient analysis files under `/tmp` are not release dependencies.
