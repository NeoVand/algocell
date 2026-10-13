"""Separately timed/accounted image prebuild. Coordinator release required.

Import and tests do not contact Modal. Dispatch accepts only the resulting
content-bound receipt; it then uses Image.from_id with no build operations.
"""
import asyncio,json,os,time
from pathlib import Path
from s8086 import PRE,HERE,RATE,sha,launch_receipt,write_new

def validated_image_id(receipt):
    data=json.loads(Path(receipt).read_text())
    if (not data.get('image_id','').startswith('im-') or not data.get('stopped_confirmed')
        or data.get('freeze_sha256')!=sha((PRE/'freeze.json').read_bytes())
        or data.get('recipe_sha256')!=sha((HERE/'modal_s8086.py').read_bytes())):
        raise ValueError('missing/mismatched prebuilt image receipt; dispatch cannot build')
    return data['image_id']

def require_prebuild():
    path=os.environ.get('S2_PREBUILT_IMAGE_RECEIPT')
    if not path:raise ValueError('separate accounted prebuild receipt required before dispatch')
    return validated_image_id(path)

def prebuild(release,ledger,output):
    receipt=launch_receipt(release)
    from dispatch8086 import SpendGuard
    from lifecycle8086 import WatchdogProcess,PREBUILD_TIMEOUT_S
    import modal
    from modal_s8086 import build_image_spec
    projection=json.loads((PRE/'projection.json').read_text())
    guard=SpendGuard(ledger,projection['job_liabilities_usd'],overhead=projection['storage_termination_reserve_usd'])
    guard.reserve('prebuild',guard.plan['prebuild'],execution_s=PREBUILD_TIMEOUT_S)
    monitor=WatchdogProcess(guard,'prebuild',None,PREBUILD_TIMEOUT_S)
    app=modal.App();image=build_image_spec();image_id=None
    try:
        with app.run(name=monitor.app_name):
            guard.transact({'kind':'lifecycle','job':'prebuild','app_id':app.app_id})
            monitor.bind_app(app.app_id);monitor.permit()
            # Provider image builds are separate from function dispatch. The
            # local timeout and independent app-stop govern this step too.
            asyncio.run(asyncio.wait_for(image.build.aio(app),PREBUILD_TIMEOUT_S))
            image_id=image.object_id
            monitor.finish();monitor.wait_confirmed()
    except BaseException as exc:
        monitor.finish()
        guard.transact({'kind':'terminal','job':'prebuild','status':'INCOMPLETE','error':type(exc).__name__,'reason':str(exc),'liability_retained':True})
        raise
    guard.settle('prebuild',max(guard.plan['prebuild'],monitor.confirmation().get('charged_usd',0)),app_id=app.app_id)
    write_new(output,{'image_id':image_id,'freeze_sha256':receipt['freeze_sha256'],'recipe_sha256':sha((HERE/'modal_s8086.py').read_bytes()),'app_id':app.app_id,'stopped_confirmed':True,'accounted_job':'prebuild','policy':'timed separate prebuild; dispatch from_id only'})
    return image_id

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--ledger',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prebuild(a.release,a.ledger,a.output)
