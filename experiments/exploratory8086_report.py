"""EXPLORATORY / post hoc Markdown generated only from committed CSVs.

No GPU use, no registered verdict reporter, no downloaded-data execution.
"""
import collections,json,subprocess
from pathlib import Path
import exploratory8086 as E

def report():
    required=['motif_scan.csv.gz','motif_hits.csv.gz','snapshot_composition.csv','input_hashes.csv','encounter_fates.csv','stageG_input_hashes.csv','first_family.csv','first_family_copies.csv','family_transmission.csv','eligibility.csv','eligibility_counts.csv','law_panel_predictions.csv','law_assay_summary.csv','law_assay_denominators.csv','publication_archive.csv','gpu_bursts.csv','verification.csv','compute_source_hashes.csv','family_world_coverage.csv','family_world_exemplars.csv']
    for name in required:
        rel=str((E.OUT/name).relative_to(E.HERE.parent))
        subprocess.run(['git','ls-files','--error-unmatch',rel],cwd=E.HERE.parent,check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['git','diff','--exit-code','HEAD','--',rel],cwd=E.HERE.parent,check=True,stdout=subprocess.DEVNULL)
    coverage=E.read_csv(E.OUT/'family_world_coverage.csv');world_examples=E.read_csv(E.OUT/'family_world_exemplars.csv');world_map={r['world']:r for r in world_examples}
    world_rows=[]
    for r in coverage:
        e=world_map.get(r['world'])
        xy=('`'+r['exemplar'][4:6]+' '+r['exemplar'][6:8]+'`') if r['exemplar'] else '—'
        world_rows.append(f"| {r['seed']} | {r['scheduled_family_genotypes']} | {r['culture_positive_scheduled_family_genotypes']} | {xy} | {r['registered_culture_status']} | {e['random16_physical_phase_exact'] if e else '—'} | {r['exemplar_max_scheduled_cells']} |")
    motif=E.read_csv(E.OUT/'motif_scan.csv.gz');fate=E.read_csv(E.OUT/'encounter_fates.csv');family=E.read_csv(E.OUT/'first_family.csv');trans=E.read_csv(E.OUT/'family_transmission.csv');elig=E.read_csv(E.OUT/'eligibility.csv');law=E.read_csv(E.OUT/'law_assay_summary.csv');archive=E.read_csv(E.OUT/'publication_archive.csv');bursts=E.read_csv(E.OUT/'gpu_bursts.csv')
    aggregate=collections.defaultdict(lambda:collections.Counter())
    for r in fate:
        key=(r['isa'],r['tile'],r['step'],r['donor_role'],r['context'])
        for k in ('partners','either_tile_phase_exact','physical_phase_exact','already_exact','new_exact','donor_retained_physical_phase','partner75','B_fetch'):aggregate[key][k]+=int(r[k])
    def frac(isa,tile,step,role,context,measure='either_tile_phase_exact'):
        r=aggregate[(isa,tile,str(step),role,context)];return f"{100*r[measure]/r['partners']:.2f}%"
    fate_rows=[]
    for isa,tile,steps in [('8086','b951',[0,20000,100000,1000000,3000000]),('8086','51b9',[0,20000,100000,1000000,3000000]),('8086','07a5',[0,20000,100000,1000000,3000000]),('Z80','01c5',[500,1000,2000,20000,100000])]:
        for t in steps:fate_rows.append(f"| {isa} `{tile}` | {t:,} | {frac(isa,tile,t,'A','soup')} / {frac(isa,tile,t,'A','uniform')} | {frac(isa,tile,t,'B','soup','donor_retained_physical_phase')} / {frac(isa,tile,t,'B','uniform','donor_retained_physical_phase')} |")
    motif_rows=[]
    for word in ['b951','55bd','57bf','07a5','ab58','5ea5','96a5']:
        rr=[r for r in motif if r['word']==word]; snap=[r for r in rr if r['source']=='snapshot'];tops=[r for r in rr if r['source']=='scheduled_top3'];full=[r for r in rr if int(r['full_weight'])]
        motif_rows.append(f"| `{word}` / `{word[2:]+word[:2]}` | {sum(int(r['full_phase0_tapes'])+int(r['full_phase1_tapes']) for r in snap):,} | {sum(int(r['partial4_tapes']) for r in snap):,} | {len([r for r in tops if int(r['full_weight'])]):,} | {min((int(r['step']) for r in full),default='—')} | {min((int(r['step']) for r in rr if int(r['partial4_tapes'])),default='—')} |")
    stats=collections.defaultdict(lambda:collections.Counter())
    for r in trans:
        if r['alternative']!='True':continue
        s=stats[r['world']];p=int(r['position']);s['tested']+=1
        if r['all16_exact']=='True' and r['all8_serial_exact']=='True':s[f'p{p}']+=1;s['both']+=1
    trace_rows=[]
    for r in family:
        st=stats[r['world']]
        trace_rows.append(f"| {r['seed']} | `{r['x']} {r['y']}` | {r['decode']} | {r['DI_after_POP']} | {r['physical_fetch_positions']} | {st['p0']} / {st['p1']} / {st['p2']} / {st['p3']} |")
    states=collections.Counter(r['status'] for r in elig);geoms=collections.Counter(int(r['d']) for r in elig if r['status']=='eligible');origins=collections.Counter((r['registered_reason'],r['status']) for r in elig)
    count_rows=[]
    for reason in sorted({r['registered_reason'] for r in elig}):count_rows.append(f"| {reason} | {sum(origins[(reason,s)] for s in states)} | {origins[(reason,'eligible')]} | {origins[(reason,'excluded')]} | {origins[(reason,'unresolved')]} |")
    total=sum(int(r['encounters']) for r in law);law_bad=sum(r['status']!='complete-zero-mismatch' for r in law);free=sum(int(r['single_assignments'])//256 for r in law);single=sum(int(r['single_assignments']) for r in law);erased=sum(int(r['erased_assignments']) for r in law);joint=sum(int(r['joint_assignments']) for r in law);cert_worlds=len({r['world'] for r in elig if r['status']=='eligible'})
    panelworlds=len({r['world'] for r in law});predcommit=law[0]['prediction_commit'];max_burst=max(float(r['burst_seconds']) for r in bursts);gpu_total=sum(float(r['burst_seconds']) for r in bursts);pause_total=sum(float(r['pause_seconds']) for r in bursts)
    assert len(elig)==779 and len(family)==10 and len(law)<=32 and all(float(r['pause_seconds'])>=float(r['burst_seconds']) for r in bursts)
    assert all(int(r['stored_bytes'])<50000000 for r in archive)
    # The S2 result remains an immutable separately registered record.
    source=E.ROOT/'S2.md';before=subprocess.check_output(['git','show','5bbc9d7^:experiments/results/s8086/S2.md'],cwd=E.HERE.parent)
    assert source.read_bytes()==before
    text=f'''# EXPLORATORY / post hoc — 8086 S2 follow-ups

Every analysis, admission, assay and conclusion here is **EXPLORATORY and post hoc**. The registered [S2 record](S2.md) is unchanged: R0 pass; R1 fail and route kill triggered; L1 untested; L2 untested with unresolved support / incomplete extraction. These follow-ups do not revise those verdicts. No population world was run, no Modal compute was used (**$0 incremental Modal spend**), and nothing was pushed.

**(a) Why did `B9 51` never become a first replicator?**

The exact cause is **not identifiable from these archives**. The best-supported explanation is a bottleneck in birth or establishment, with strong encounter-role effects once the soup has changed. The response depends on entry phase and role. In the `B9 51` entry phase, mature 8086 soup improves active copying: at 20k, planted `B9 51` copies into soup partners at {frac('8086','b951',20000,'A','soup')}, versus {frac('8086','b951',20000,'A','uniform')} into uniform partners. When planted as B, however, it survives only {frac('8086','b951',20000,'B','soup','donor_retained_physical_phase')} against soup, versus {frac('8086','b951',20000,'B','uniform','donor_retained_physical_phase')} against uniform A. In the opposite `51 B9` entry phase, active copying instead drops to {frac('8086','51b9',20000,'A','soup')} against soup versus {frac('8086','51b9',20000,'A','uniform')} against uniform partners. Thus tar can impair this pusher in a phase-specific way. At step zero the two contexts are closely matched. The role asymmetry can impede establishment; this is an inference, not a measured population growth rate.

No full `B9 51` or `51 B9` tile, and no alternating run of four or more bytes, appears in any of the 11,700 scheduled top-3 samples or 2,000,000 checkpoint cells. Isolated two-byte matches do occur and are tabulated separately. The absence of a four-byte seed in these observations supports an arrival bottleneck, but does not reveal unobserved transient seeds or how they were lost. A four-byte exact motif is rare even in uniform data, and shorter functional precursors could grow between checkpoints; these observations do not estimate a birth rate. Confidence is high in the scan and encounter measurements, moderate that receiving-role vulnerability matters in sampled mature soup, and low in a specific historical cause of the missing first replicator.

The scan covers all 20 canonical complete worlds, 585 scheduled samples per world, and checkpoints 0, 20k, 100k, 1M and 3M; snapshot bytes match their logged hashes. A partial tile means a **linear** alternating run of ≥4 bytes inside a cell; ≥8 and isolated pairs are also recorded. Both physical tile phases are searched. The following counts pool repeated observations, not unique lineages. Words are the literal/memory-fed exact-zero census witnesses across L16/L32 and budgets 128/512; both orientations are searched. L32 is only a census reference.

| EXPLORATORY word / opposite phase | Full checkpoint cells | Checkpoint cells with run ≥4 | Scheduled samples with a full tile | Earliest full observation, steps | Earliest run ≥4, steps |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(motif_rows)}

`55 BD` first appears as a full top-3 tile at 600 in world 8086019, with only 15 cells (18 at 650); it is below the registered 100-cell share threshold. `07 A5` is observed as full and partial tilings, and the re-extracted culture panel contains many nonperiodic `07 A5 …` full-tape copiers. Appearance of a motif is not a registered detection or a causal classification. Exact occurrences, times, tapes, cell indices and multiplicities are in [motif hits](exploratory/motif_hits.csv.gz); zeros and complete denominators are in [scan CSV](exploratory/motif_scan.csv.gz). [Composition](exploratory/snapshot_composition.csv) records byte fractions and diversity, without defining tar by a single byte.

For the encounter panel, 512 distinct cells are sampled without replacement per world/checkpoint; matched seeded uniform partners are used for every tile. Both orientations use physical entry at A, budget 128. “A copy” below accepts either of the two exact tile phases in B; “B survival” requires the original planted B bytes at their physical phase. Actual inputs and outputs are retained in [raw fates](exploratory/encounter_fates_raw.csv.gz), and [summaries](exploratory/encounter_fates.csv) separate exact-phase copies, pre-existing copies, new copies, donor survival, similarity and B fetch. Each table entry pools 10,240 encounters; cells within a world are not independent evolved worlds.

| EXPLORATORY tile | Checkpoint | A copy: soup / uniform | B survival: soup / uniform |
| --- | ---: | ---: | ---: |
{chr(10).join(fate_rows)}

The Z80 `01 C5` comparison uses the actual Stage G L16/budget128 executor and all 20 seeds 2001–2020, with local snapshots at 500, 1000, 2000, 20k and 100k. At 500, A-copy success is {frac('Z80','01c5',500,'A','soup')} versus {frac('Z80','01c5',500,'A','uniform')}, and B survival is {frac('Z80','01c5',500,'B','soup','donor_retained_physical_phase')} versus {frac('Z80','01c5',500,'B','uniform','donor_retained_physical_phase')}. Early Stage G soup is thus friendlier to a receiving pusher in this panel. Some Stage G partners already carry `01 C5`; the summary explicitly records these and new-copy counts. Stage G had already undergone pusher expansion by 500, so ecology and expansion are confounded. S2 has no dense pre-20k population checkpoints; different times, machine semantics, mutations and population histories preclude an architectural or causal establishment claim. Late Z80 soup also suppresses B survival. The comparison supports context dependence, not a universal 8086/Z80 ordering.

**(b) What is `5F A4 x y`, and does it carry heritable variation?**

Yes: all ten observed first-family genotypes (10 of 14 registered first copiers) transmit multiple exact alternative genotypes through eight fresh-partner passages. It is a four-byte forward-overlap copier: `POP DI; MOVSB; branch`. Initial SP=1 makes POP read A[1]=A4 and A[2]=x, so DI=0xXXA4 and DI mod 32=4; SI=0, ES=DS=0, SP becomes 3 and DF=0. Each MOVSB reads DS:SI and writes ES:DI, then increments SI and DI. All ten exemplars make 64 byte copies in 128 steps, with displacement 4, and write every A and B coordinate. The first moves are 0→4, 1→5, 2→6, 3→7; overlap propagates the first four ancestral bytes throughout the ring. B starts receiving copies at SI=12 / DI mod32=16. Final source map is `[0,1,2,3]` repeated four times, R=K=C_R={{0,1,2,3}}, E={{4,…,15}} and F is empty under the conservative law footprint.

The two variable bytes supply a taken control transfer. The frozen NMOS core interprets 60–6F as conditional-jump aliases, so ordinary modern disassembly alone would be misleading. Conditions stay true because POP/MOVSB preserve the relevant reset flags; LOOP wraps initial CX=0 to 65535, staying taken for the encounter. For rel8 branches, `4 + signed(y)` is 1 modulo 32, returning to the physical MOVSB byte. Raw IP can grow, wrap or repeat while its mapped position remains in A. The `E9 3C` exemplar fetches a third displacement byte at physical A[4], already overwritten with 5F; its displacement is 0x5F3C, and next IP 5 + displacement again maps to A[1]. Its raw IP advances by 24,384 each cycle. This explains confinement without requiring repeated *raw* instruction-start positions; the original descriptive one-pass/loop labels are unchanged.

The following table details the ten registered first-family exemplars. The all-world panel below also covers later and low-share family observations. Full step-by-step read/write/fetch/start events, first source/destination moves, numeric DI and physical fetch coverage are committed in [family traces](exploratory/first_family_traces.csv.gz) and [exemplar summary](exploratory/first_family.csv). The [per-copy pointer CSV](exploratory/first_family_copies.csv) reconstructs every full 16-bit SI/DI state from reset, the single POP and DF=0 MOVSB increments, and checks every mapped read/write against the trace. It includes each raw IP, source ancestor, byte value and before/after pointers; reconstruction is explicitly labelled.

| EXPLORATORY seed | x y | Frozen decode | DI after POP | Physical fetch positions | Transmitted alternatives at positions 0 / 1 / x / y |
| --- | --- | --- | ---: | --- | --- |
{chr(10).join(trace_rows)}

The descriptive X1 scan varies each of the four founder coordinates through all 256 values, repeats the mutant founder four times, challenges it against 16 fresh uniform partners, and separately performs eight serial fresh-partner encounters. There are 10 genotypes, {len(trans):,} assignments ({sum(s['tested'] for s in stats.values()):,} non-baseline alternatives), and {sum(s['both'] for s in stats.values()):,} alternatives that pass all 16 exact physical-phase challenges and all eight serial passages. Each genotype has at least seven transmissible y alternatives; nine also have 19 x alternatives. Four have one transmissible byte-0 alternative. Results establish constrained **heritable variation in the executed founder**, not arbitrary payload freedom. Mutations here alter every homologous founder byte; the law erasure assay below instead mutates a single physical coordinate. [All X1 assignments](exploratory/family_transmission.csv) and [serial hashes](exploratory/family_serial.csv.gz) retain failures as well as successes. This small finite panel supplies no establishment or fitness claim.

A separate **all-world descriptive trace panel** covers every world with a scheduled exact `5F A4 x y` tiling: {len(world_examples)} of 20 worlds. All {len(world_examples)} selected exemplars copy at the exact physical phase against all 16 fresh census partners, with the four-byte geometry. Seven were **unjudged** by registered culture because their maximum scheduled counts stayed below the 100-cell detection threshold; they are not classified culture-negative. The additional world 8086003 has culture-positive later family members after its different registered first copier. Worlds 8086008 and 8086015 have no scheduled full-family tape, and no such full-family checkpoint cell. Across all five checkpoints, only world 8086003 contains full-family cells (1,061 repeated observations / 21 genotypes); the scheduled record retains many transient families missed by coarse checkpoints.

| EXPLORATORY seed | Scheduled family genotypes | Registered culture-positive among them | Selected x y | Registered selected status | Exact / 16 fresh partners | Selected max scheduled cells |
| --- | ---: | ---: | --- | --- | ---: | ---: |
{chr(10).join(world_rows)}

For this panel the storage observer records **actual full SI and DI before every instruction**, rather than reconstructing them. All 306 full lane memories and original machine-record words matched uninstrumented execution. [Coverage](exploratory/family_world_coverage.csv), [exemplars](exploratory/family_world_exemplars.csv), [all 17-context pointer/fetch/data traces](exploratory/family_world_traces.csv.gz), and [per-MOVSB actual pointers/read-write pairs](exploratory/family_world_copies.csv.gz) preserve the result. Event kind 5 stores SI in the position field and DI in the value field; [the observer source](exploratory/family_pointer_observer.wgsl) is archived. The SHA in the table identifies compiled source after removing its leading explanatory label comment. This additional descriptive panel is outside the 779-candidate eligibility denominator and outside the 32-genotype exact-law intervention panel. It does not revise a registered detection.

**(c) What changes when POP ES and the other original missing opcodes are modelled?**

The separate [exploratory verifier](../../exploratory8086_law.py) transfers full-word dependencies for POP ES/CS/SS/DS and register POP, stack/segment address dependencies, frozen 60–6F aliases, ALU taint, IN's frozen all-ones return, LES/LDS, far JMP, byte immediates and register exchange. Segment taint is deliberately conservative despite the tiny address ring. Fifteen [unit/integration tests](../../tests/test_exploratory8086.py) check numeric registers and ordered stack reads against the frozen machine, verify complete full-tape and four-byte source maps, and reject arithmetic memory writes as copies. Registered `law8086.py`, machine WGSL and S2 verdict files were not edited.

All 779 culture-positive genotype-world cases were re-extracted. Admission requires agreement across all 17 archived ordered contexts; an excluded or unresolved case stops at its first failed certificate. Post hoc admission is {states['eligible']} eligible across {cert_worlds} worlds, {states['excluded']} excluded and {states['unresolved']} unresolved; L1 remains zero. Subsequent unsupported dependencies still block admission, so modelling the first missing opcode does not certify every original unresolved case.

| EXPLORATORY original registered reason | Cases | Now eligible | Now excluded | Still unresolved |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(count_rows)}

The newly admitted geometries are **{geoms[16]} full-tape copiers with d=L=16**, identity source map, R=all A and E=empty; and **{geoms[4]} four-byte overlap copiers with d=4**, F=empty and E={{4,…,15}}. The d=16 cases have forward contiguous byte moves but do **not** reduce a 16-byte genotype to a smaller overlap founder: they carry arbitrary unexecuted full-tape coordinates according to their held code footprints. All observed admissions have source start 0 and physical phase 0. No smaller-seed claim is assigned to those full-tape cases. [Every certificate and exclusion](exploratory/eligibility.csv) and [reason totals](exploratory/eligibility_counts.csv) are retained.

Before assaying, [the 32-genotype panel and all K/R/C_R/F/E/source-map predictions](exploratory/law_panel_predictions.csv) were committed in `{predcommit}`. Selection took the smallest held footprint in each of 20 worlds, all ten first-family genotypes, and two deterministic additional cases. Thus there are 22 full-tape and ten four-byte cases across {panelworlds} independent evolved worlds; it is a purposive capped panel, not a random estimate for all admissions.

The **exact frozen assay function** was reused in independent globals, with the extended mismatch verifier, logged foreground GPU wrapper, and 64 independent seeded joint assignments (`joint:256`…`joint:319`) replacing its 512-joint panel. It checks every physical output byte with the predicted source map, all 17 original census partners, all-value single substitutions, all-value mutable erasures, and serial passages to g=8 for every single/joint assignment. No rotation or favorable alignment is introduced; no candidate is removed after a mismatch.

Observed: **{law_bad} candidate mismatches / incomplete cases in {total:,} encounters**, {single:,} single assignments over {free} genotype-coordinate pairs, {joint:,} joint assignments, and {erased:,} erasure assignments. All {len(law)} selected genotypes completed with exact matches. Serial streams, generation-zero source maps, physical phase and actual input/output/partner hashes appear in [per-genotype encounter CSVs](exploratory/law_encounters/); [summary](exploratory/law_assay_summary.csv) and [denominators](exploratory/law_assay_denominators.csv) distinguish challenge and generation. For d=4, F is empty: its 64 joint assignments repeat the baseline and offer no payload evidence; the erasure checks are substantive. For d=16, E is empty: erasure controls are inapplicable. There are zero held-erased coordinates in the selected panel. An independent [CPU audit](exploratory/verification.csv) recomputed every expected offspring hash, census/serial input hash, serial partner byte stream, source map and per-case denominator against the frozen assignment generators. Finite assays support these post hoc predictions only; they do not turn the registered L2 result into a pass, certify the remaining unresolved candidates, or establish unrestricted robustness for every eligible genotype.

**(d) Is publication provenance archived?**

Yes. {len(archive)} files from the S2 worktree launch directories and billing-poller are byte-preservingly archived under [launch/exploratory-publication-archive](launch/exploratory-publication-archive/), including nested scoring/recovery runs, ledgers, watchdog logs, handoffs, invocation scripts, tests and resource readings. Files over 1 MB were deterministically gzipped, without rewriting historical contents. [The archive manifest](exploratory/publication_archive.csv) records every original path, archived path, both hashes and byte counts. Stored files total {sum(int(r['stored_bytes']) for r in archive):,} bytes; the largest is {max(int(r['stored_bytes']) for r in archive):,} bytes, below 50 MB. The historical launch records remain historical provenance; their presence is not new compute authorization or a changed registered interpretation.

**Reproduction and resource record (EXPLORATORY / post hoc).** Downloaded data were read with `/Users/neo/repos/algocell/experiments/.venv/bin/python -I`. Local canonical S2 downloads and local Stage G copies supplied all inputs; no new download or remote compute was needed. Executor/verifier byte hashes are in [source identities](exploratory/compute_source_hashes.csv). Input hashes are in [S2 inputs](exploratory/input_hashes.csv) and [Stage G inputs](exploratory/stageG_input_hashes.csv). All {len(bursts)} local GPU operations were synchronous foreground calls with equal measured pauses; maximum burst {max_burst:.3f}s, total call time {gpu_total:.3f}s, total pause {pause_total:.3f}s. [Burst ledger](exploratory/gpu_bursts.csv) includes compilation and tests. All committed files are below 50 MB.

Run from the repository root; scripts add only the trusted local experiments directory to isolated Python's import path:

```sh
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py scan
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py reextract
# Commit predictions before the law stage; execute GPU stages sequentially.
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py encounters
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py transmission
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py world_family
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py law
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py archive
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086_verify.py
# Commit the CSVs, then regenerate this presentation without compute:
/Users/neo/repos/algocell/experiments/.venv/bin/python -I experiments/exploratory8086.py report
```

[Analysis source](../../exploratory8086.py), [GPU assay wrapper](../../exploratory8086_compute.py), [CSV-to-Markdown generator](../../exploratory8086_report.py) and [CPU hash audit](../../exploratory8086_verify.py) are committed. Re-running local assays appends to the burst ledger; timings and Git prediction identities are provenance, not byte-stable scientific outputs. Registered report generation is deliberately separate.
'''
    (E.ROOT/'EXPLORATORY.md').write_text(text)
    print(E.LABEL,'generated EXPLORATORY.md from committed CSVs',flush=True)
