"""Read I1-T2 output names, summary markers and best-effort Modal billing only."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import modal
from tier2_dispatch import DATA, complete, jobs, marker_name, projection, write_spend


async def billing(app_id, started_utc):
    """Best effort: this billing RPC is optional and may not be enabled."""
    try:
        from modal_proto import api_pb2
        from google.protobuf.timestamp_pb2 import Timestamp
        from modal.client import _Client
        client = await _Client.from_env()
        start, end = Timestamp(), Timestamp()
        start.FromDatetime(datetime.fromisoformat(started_utc.replace("Z", "+00:00")))
        end.FromDatetime(datetime.now(timezone.utc))
        request = api_pb2.WorkspaceBillingReportRequest(start_timestamp=start, end_timestamp=end,
                    resolution="h", app_ids=[app_id])
        async with asyncio.timeout(8):
            rows = [r async for r in client._stub.WorkspaceBillingReport.unary_stream(request)]
        return sum(float(r.cost) for r in rows), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:180]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update-ledger", action="store_true")
    a = ap.parse_args()
    vol = modal.Volume.from_name("algocell-atlas-runs")
    grouped = defaultdict(list)
    for j in jobs():
        grouped[j["experiment"]].append(j)
    markers = {}
    for experiment, group in grouped.items():
        prefix = f"tier2/{experiment}"
        try:
            names = {e.path.removeprefix(prefix + "/") for e in vol.iterdir(prefix, recursive=True)}
        except (FileNotFoundError, modal.exception.NotFoundError):
            names = set()
        cache = {}

        def read_json(name):
            if name not in cache:
                cache[name] = json.loads(b"".join(vol.read_file(f"{prefix}/{name}")))
            return cache[name]
        count, states = 0, Counter()
        for j in group:
            count += int(complete(j, names, read_json))
            name = marker_name(j, "spend")
            if name in names:
                markers[j["id"]] = read_json(name)
                states[markers[j["id"]]["status"]] += 1
        print(f"{experiment:12} {count:3}/{len(group):3} complete; " + ", ".join(f"{v} {k}" for k, v in states.items()))
    p = projection(jobs(), markers)
    billed, error = None, None
    launch_path = DATA / "launch_calls.json"
    if launch_path.exists():
        launch = json.loads(launch_path.read_text())
        from modal._utils.async_utils import synchronize_api
        billed, error = synchronize_api(billing)(launch["app_id"], launch["started_utc"])
        print(f"App: {launch['app_id']}")
    message = f"${billed:.4f} (metered; may lag)" if billed is not None else f"unavailable ({error or 'not launched'})"
    print(f"Modal spend so far: {message}")
    print(f"Container runtime estimate: ${p['estimated_execution_usd']:.4f}; all-in remaining-work forecast: ${p['total_forecast_usd']:.4f} / $40")
    if a.update_ledger:
        write_spend(DATA / "SPEND.md", jobs(), markers, message)
    if p["total_forecast_usd"] > 40:
        print("BUDGET STOP: required remaining work exceeds $40; coordinator must amend registration before further dispatch.")


if __name__ == "__main__":
    main()
