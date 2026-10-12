"""Generate v6.2 coordinates and supporting ED tables; no simulation."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE/'out/v6/data'


def table(name):
    return pd.read_csv(D/(name+'.csv'),low_memory=False)


def save(name,data):
    data.to_csv(D/(name+'.csv'),index=False)


def km(times,horizons):
    t = np.where(np.isfinite(times),times,horizons)
    event = np.isfinite(times)
    alive = Fraction(1,1)
    for time in sorted(set(t[event])):
        risk=int((t>=time).sum());deaths=int(((t==time)&event).sum())
        # Exact rational survival prevents 0.5000000000000001 skipping the
        # registered first crossing at precisely one half.
        alive *= Fraction(risk-deaths,risk)
        if alive<=Fraction(1,2):
            return time
    return np.nan


def main():
    z = table('z80_payload').query('eligible_dividing==True')
    save('law_population',z.groupby(['predicted','n_unexec']).size().reset_index(name='cells'))
    e = table('capacity_edits')
    save('law_z80_c1',e.groupby(['predicted_free','measured_free']).size().reset_index(name='edits'))
    evo = table('evolved6502_capacity')
    save('law_6502_c2',evo.groupby(['predicted_free','measured_free']).size().reset_index(name='roles'))
    c = table('constructed6502').query('hypotheses==True')
    save('law_6502_controls',c.groupby(['d − 6','payload carried (sites ≥ 6)']).size().reset_index(name='cases'))
    hap = table('capacity_haplotypes')
    strips = []
    for key in ('ldir32_v04','ldir32_v60'):
        edit = e[e.edit_id==key].iloc[0]
        h = hap[(hap.edit_id==key)&(hap.direction=='forward')&(hap.lineage==0)&(hap.generation==8)].iloc[0]
        sources = list(map(int,h.sources.split(';')))
        offspring = list(map(lambda v:int(v,16),h.offspring.split()))
        free = set(map(int,edit.free_positions.split(';'))) if pd.notna(edit.free_positions) else set()
        assert h.exact==h.provenance_exact==h.core_exact==1
        assert h.actual_d==edit.d and len(sources)==len(offspring)==32
        for position in range(32):
            state = 'core' if position<4 else 'transmitted' if position in free else 'erased'
            assert (sources[position]==position) if state=='transmitted' else (sources[position]<4)
            strips.append(dict(edit_id=key,lineage=0,generation=8,position=position,state=state,
                               source_position=sources[position],offspring_byte=offspring[position],
                               displacement=int(edit.d),original_byte=int(edit.original_value),edited_byte=int(edit.value)))
    save('lineage_strips',pd.DataFrame(strips))
    # All 99 working values have nominal arithmetic. Only 11 have a formal
    # Proposition 8 prediction; preserve that missingness in a separate column.
    dial = table('allele_dial')
    dial['inherited_aligned'] = dial.allele_aligned_frac>=.5
    assert int(dial.culture_replicator.sum())==99
    save('dial_coordinates',dial[['value','effective_d','culture_replicator','inherited_aligned',
                                 'allele_aligned_frac','nominal_d_minus_c','predicted_payload','prop8_eligible']])
    pe = table('parent_environment')
    save('ed_parent_environment',pe)
    save('ed_joint_challenges',pd.concat([
        table('capacity_challenges').assign(system='Z80 C1'),
        table('evolved6502_challenges').assign(system='6502 C2')],ignore_index=True))
    v = table('8086_validation').iloc[0]
    i = table('impact').query("core=='C1'").iloc[0]
    claims,components = table('claims'),table('components')
    rows = [('Instruction vectors',int(v.pass_arch),int(v.tests)),
            ('Ring encounters',int(i.unaffected),int(i.decision_conditions)),
            ('Isolated assays',int(i.assay_identical),int(i.assay_combinations)),
            ('Claim verdicts',int((claims.status=='stands').sum()),len(claims)),
            ('Claim components',int((components.comparison=='same').sum()),len(components))]
    save('validation_fractions',pd.DataFrame([dict(level=label,unchanged=n,total=total,
                                                fraction=n/total,changed=total-n) for label,n,total in rows]))
    d = table('dial_worlds')
    summary=[]
    for (cohort,prob),group in d.groupby(['cohort','p']):
        assert len(group)==10
        summary.append(dict(cohort=cohort,p=prob,worlds=10,
                            open_first=int((group.first_open==True).sum()),
                            T_majority=int((group.frac_transmitter_of_heritable>.5).sum()),
                            confinement_km=km(group.t_closed.to_numpy(),group.horizon.to_numpy())/1000))
    summary=pd.DataFrame(summary)
    registered=table('c1_dial_summary').sort_values('p')
    actual=summary.query("cohort=='C1'").sort_values('p')
    assert np.allclose(actual.confinement_km*1000,registered.KM_t_closed)
    assert np.array_equal(actual.open_first,registered.open_first)
    assert np.array_equal(actual.T_majority,registered.transmitter_majority)
    save('lethality_summary',summary)
    t = table('competition_trajectories')
    save('competition_medians',t.groupby(['cohort','tar','start','step']).agg(
        T_share=('T_share','median'),core_share=('core_share','median'),worlds=('seed','nunique')).reset_index())
    # Deterministic collision offsets are generated data, never a hidden plot transformation.
    ends=table('competition_endpoints')
    def positions(measure):
        data=[]
        for (cohort,tar),group in ends.groupby(['cohort','tar']):
            placed=[]
            span=1.1 if measure!='T_distinct_tails_end' else ends[measure].max()*1.1
            for row in group[group[measure].notna()].sort_values([measure,'start','seed']).to_dict('records'):
                y=row[measure]
                # Final Figure 4b is the smallest view: 56 mm x 49 mm.
                # Horizontal displacement of .11 and ordinate displacement of
                # .032*span each exceed a 10-pt² glyph's printed diameter.
                candidates=[0]+[s*v for v in [.12,.24,.36] for s in [-1,1]]
                distances=[np.array([((off-x)/.11)**2+((y-v)/(span*.032))**2
                                     for x,v in placed]) for off in candidates]
                costs=[int((dist<1).sum()) for dist in distances]
                minimum=min(costs)
                eligible=[j for j,cost in enumerate(costs) if cost==minimum]
                if minimum:
                    index=max(eligible,key=lambda j:distances[j].min())
                else:index=eligible[0]
                offset=candidates[index]
                placed.append((offset,y))
                data.append(dict(**row,offset=offset,measure=measure))
        return pd.DataFrame(data)
    save('endpoint_positions',pd.concat([positions(m) for m in ['T_share_end','core_share_end','T_distinct_tails_end']]))
    inputs = [p for p in D.glob('*.csv') if p.name not in ('v62_generated_hashes.csv',)]
    save('v62_generated_hashes',pd.DataFrame([dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(inputs)]))
    print('Generated v6.2 plotting tables, 64 registered lineage positions, and ED tables.')


if __name__=='__main__':
    main()
