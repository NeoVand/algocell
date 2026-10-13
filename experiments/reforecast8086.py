"""CPU-only prospective option forecasts from frozen timings; never refreeze."""
import json
from benchmark8086 import project
from s8086 import PRE,Registration,read_json,sha

OPTIONS={'a':Registration(3000000,('A','B')),'b':Registration(3000000,('A',)),
         'c-1M':Registration(1000000,('A','B')),'c-1.5M':Registration(1500000,('A','B')),
         'd':Registration(1500000,('A',))}

def main():
    bench=read_json(PRE/'benchmark.json'); frozen=read_json(PRE/'projection.json')
    prep=frozen['job_liabilities_usd']['preparation']
    fixed=sum(frozen['job_liabilities_usd'][j] for j in ('controls','diagnosis','preparation'))+frozen['storage_termination_reserve_usd']
    result={}; rows=[]
    for name,registration in OPTIONS.items():
        projected=project(bench,prep,registration=registration)
        projected.update(option=name,prospective_only=True,launch_status='STOPPED',
                         frozen_benchmark_sha256=sha((PRE/'benchmark.json').read_bytes()),
                         frozen_fixed_allowance_usd=fixed)
        jobs=projected['job_liabilities_usd']
        parts={p:sum(v for j,v in jobs.items() if j.startswith(p+':')) for p in ('world','score','extract','law')}
        parts['fixed']=sum(jobs[j] for j in ('controls','diagnosis','preparation'))+projected['storage_termination_reserve_usd']
        assert parts['fixed']>=fixed
        assert abs(sum(parts.values())-projected['projection_usd'])<1e-9
        projected['component_totals_usd']=parts; result[name]=projected
        rows.append('| '+ ' | '.join([name,'+'.join(registration.arms),f'{registration.horizon:,}',
             str(len(registration.sample_steps)),str(len(registration.snapshots)),
             *[f'${parts[p]:.6f}' for p in ('world','score','extract','law','fixed')],
             f"${projected['projection_usd']:.6f}",f"${60-projected['projection_usd']:.6f}"])+' |')
    (PRE/'forecast_options_repaired.json').write_text(json.dumps(result,indent=2)+'\n')
    text='''# Repaired prospective forecasts — 2026-10-12

**STOPPED.** CPU reprojections from the unchanged frozen L4 measurements.
No Modal call, new benchmark, final registration, freeze, or launch receipt.
All totals include incurred preparation, all ten controls, both separate law
panels for each retained arm, and the entire original $6.119213 fixed allowance.
Controls/diagnosis also receive the repaired lifecycle envelope, so the fixed
column increases. No assay savings are assumed.

| Option | Arms | H | Samples | Checkpoints | World | Score | Extract | Laws L1+L2 | Fixed | Complete | $60 headroom |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

The frozen 20% throughput reduction, 10% runner margin, 1.5x sample/checkpoint,
scoring/extraction/assay factors, measured startup allowances, 69 extraction
candidates, and 16 cases per separate law family are retained. Horizons truncate
the original schedule; checkpoints are distinct original checkpoints through H
plus H. R2b always uses exactly 11 points H−250k,…,H at 25k spacing. Final-top,
64-cell snapshot, occurrence, censoring and completeness use the selected H.
Twenty worlds and the integer route cutoffs are unchanged in each retained arm.

Each job reserves ceil(frozen conservative body allocation) + 361 resource
seconds: two 120s startup/attempt allowances, 60s termination, 1s polling, and
60s for six bounded 10s control RPCs. This replaces the unenforced +120s margin.
The body allocation still contains its original compile/startup margin. A single
app and container pool run per job; the independent host watchdog is armed
before spawn, never renews its monotonic clock on restart, and requests app stop
before the termination/control margin begins. No next app or settlement is
allowed before provider STOPPED and an empty task list. Unknown/failed stops
retain liabilities and generate terminal records; an unconfirmed overrun cannot
be represented as completed biology or a confirmed physical bound. The mock
validates the policy, not provider availability or termination latency.

Option d remains the only listed option below $60. Selection and prospective
scientific approval remain with the user/coordinator. R2 has not been altered.
'''
    (PRE/'FORECAST_OPTIONS_REPAIRED.md').write_text(text)
    print('\n'.join(rows))

if __name__=='__main__': main()
