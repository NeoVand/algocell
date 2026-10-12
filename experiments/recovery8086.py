"""Prospective infrastructure recovery: separate complete step-0 trajectories.

Checkpoints currently omit RNG/execution state. Partial trajectories cannot be
resumed or spliced while preserving the registered first-event/persistence
estimands. The first complete durable trajectory for each frozen seed is kept;
every provider attempt, including discarded partials, is logged and charged.
"""
import json,os,uuid
from pathlib import Path
from s8086 import write_new

def attempt_directory(root,world,*,attempt_id=None):
    root=Path(root);attempt_id=attempt_id or uuid.uuid4().hex
    directory=root/'attempts'/world/attempt_id
    directory.mkdir(parents=True,exist_ok=False)
    prior=sorted(p.name for p in directory.parent.iterdir() if p.is_dir() and p!=directory)
    metadata={'policy':'restart_step_0','attempt_id':attempt_id,'prior_attempt_ids':prior,
              'provider_task_id':os.environ.get('MODAL_TASK_ID'),'provider_container_id':os.environ.get('MODAL_CONTAINER_ID'),
              'checkpoint_resume':False,'frozen_seed_unchanged':True,'selection':'first_complete_durable_trajectory',
              'reason':'checkpoints omit complete RNG/execution state; never splice partial histories'}
    write_new(directory/'recovery.json',metadata)
    return directory,metadata

def publish_complete(attempt,root,world,registration):
    """Publish snapshots first, canonical complete record last; never overwrite."""
    attempt,root=Path(attempt),Path(root);source=attempt/f'{world}.jsonl';target=root/source.name
    rows=[json.loads(s) for s in source.read_text().splitlines()]
    summaries=[r for r in rows if r.get('kind')=='summary']
    if (len(summaries)!=1 or summaries[0]['steps_run']!=registration.horizon or any(r.get('kind')=='incomplete' for r in rows)
        or [r['step'] for r in rows if r.get('kind')=='sample']!=list(registration.sample_steps)
        or [r['step'] for r in rows if r.get('kind')=='snapshot']!=list(registration.snapshots)):
        raise ValueError('partial/restarted history is INCOMPLETE, cannot publish or score')
    if target.exists():
        # A retry after durable completion is an acknowledgement; never select
        # another trajectory based on its outcomes or overwrite the first.
        old=[json.loads(s) for s in target.read_text().splitlines()]
        write_new(attempt/'publication.json',{'status':'duplicate_complete_ack','selected':False})
        return next(r for r in old if r.get('kind')=='summary')
    for row in rows:
        if row.get('kind')=='snapshot':
            src=attempt/row['file'];dst=root/row['file']
            temp=root/(row['file']+'.publishing')
            with temp.open('wb') as f:f.write(src.read_bytes())
            os.replace(temp,dst)  # no canonical history exists yet
    temp=root/(source.name+'.publishing')
    with temp.open('wb') as f:f.write(source.read_bytes());f.flush();os.fsync(f.fileno())
    os.link(temp,target);temp.unlink()  # complete record appears atomically
    write_new(attempt/'publication.json',{'status':'first_complete_durable','selected':True})
    return summaries[0]
