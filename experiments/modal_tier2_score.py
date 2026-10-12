"""Score existing I1-T2 outputs only. Never calls a world runner.

modal run --detach modal_tier2_score.py --stage-l-scoreable
Input files are linked into container scratch; analysis never writes to tier2/.
Serial/branching reuse completed C1 measurement tables and call frozen --report.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
DATA = HERE / "results/tier2"
# Exported read-only from registration/launch commit 7821ff0, including helpers.
# Keep later unrelated edits on feat/experiments-runner out of frozen scoring.
SOURCES = Path(os.environ.get("TIER2_FROZEN_SOURCES", "/tmp/algocell-t2s-score/frozen_sources"))
if not SOURCES.exists():
    import subprocess
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "7821ff0", "experiments/algocell_exp",
         "experiments/conds", "experiments/manuscript/figures/soup_stills.py"], cwd=HERE, text=True).splitlines()
    paths += ["experiments/" + p for p in subprocess.check_output(
        ["git", "ls-tree", "--name-only", "7821ff0:experiments"], cwd=HERE, text=True).splitlines() if p.endswith(".py")]
    for name in paths:
        target = SOURCES / name.removeprefix("experiments/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "show", "7821ff0:" + name], cwd=HERE))
RATE = 0.00025286
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libvulkan1", "libx11-6", "libxext6", "libxcb1", "libegl1", "libgles2", "libglvnd0")
         .pip_install("wgpu==0.32.0", "numpy==2.5.3", "brotli==1.2.0", "pandas==3.0.6",
                      "matplotlib", "tabulate", "pillow")
         .env({"WGPU_BACKEND_TYPE": "Vulkan", "XDG_RUNTIME_DIR": "/tmp", "MPLBACKEND": "Agg"})
         .add_local_dir(SOURCES / "algocell_exp", remote_path="/root/algocell_exp")
         .add_local_dir(SOURCES / "conds", remote_path="/root/conds")
         .add_local_file(DATA / "scoring_sources.json", remote_path="/root/scoring_sources.json")
         .add_local_file(SOURCES / "manuscript/figures/soup_stills.py",
                         remote_path="/root/manuscript/figures/soup_stills.py"))
for path in sorted(SOURCES.glob("*.py")):
    image = image.add_local_file(path, remote_path=f"/root/{path.name}")
vol = modal.Volume.from_name("algocell-atlas-runs")
app = modal.App("algocell-i1-tier2-scoring")


def score_existing(experiment: str, score_tag: str) -> dict:
    import shutil
    import subprocess
    import sys
    import time
    import pandas as pd

    started = time.time()
    vol.reload()
    out = Path("/runs/tier2_scores") / score_tag / experiment
    out.mkdir(parents=True, exist_ok=True)
    previous = out / "execution.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        if old["status"] != "failed":
            raise RuntimeError("Refusing to overwrite completed or running scoring")
        archive = out / "failed_attempt"
        archive.mkdir(exist_ok=True)
        for p in [previous, *out.glob("*.log")]:
            shutil.copy2(p, archive / p.name)
    record = dict(experiment=experiment, started_epoch=started, status="running",
                  gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip(),
                  source_hashes={}, commands=[], input_root="/runs/tier2", output_root=str(out))
    if "L4" not in record["gpu"]:
        raise RuntimeError("Scoring requires the costed L4 GPU")
    for binding in json.loads(Path("/root/scoring_sources.json").read_text())["scripts"]:
        sha = hashlib.sha256((Path("/root") / binding["script"]).read_bytes()).hexdigest()
        if sha != binding["prepared_sha256"]:
            raise RuntimeError(f"Frozen scoring hash mismatch: {binding['script']}")
        record["source_hashes"][binding["script"]] = sha

    def persist():
        record["updated_epoch"] = time.time()
        (out / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
        vol.commit()

    def run(script, *args):
        command = [sys.executable, str(Path("/root") / script), *map(str, args)]
        log = out / f"{len(record['commands']):02d}_{Path(script).stem}.log"
        item = dict(argv=command, started_epoch=time.time(), log=log.name)
        record["commands"].append(item)
        persist()
        print("SCORING", experiment, script, flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, cwd="/root", stdout=stream, stderr=subprocess.STDOUT)
        item.update(finished_epoch=time.time(), returncode=result.returncode)
        persist()
        if result.returncode:
            raise RuntimeError(f"{script} failed: {log.read_text()[-2000:]}")

    def view(name, sources):
        directory = Path("/tmp/tier2_score_inputs") / name
        directory.mkdir(parents=True, exist_ok=True)
        for source in sources:
            for p in (Path("/runs/tier2") / source).iterdir():
                if p.is_file() and (p.suffix in (".jsonl", ".npy") or p.name.endswith((".summary.json", ".u8.br"))):
                    (directory / p.name).symlink_to(p)
        return directory

    persist()
    try:
        if experiment == "DZ":
            r = view("dial", ["DZ"])
            run("dial_soups.py", "--analyse", "--machine", "c1", "--runs", r, "--out", out)
        elif experiment == "E":
            r = view("offset", ["E", "E-L32L"])
            run("offset_switch.py", "--analyse", "--machine", "c1", "--indir", r, "--out", out)
            run("offset_encounter.py", "--machine", "tier2c1_assay", "--runs", r, "--out", out / "OFFSET_ENCOUNTER.md")
        elif experiment == "R4":
            r = view("r4_inv", ["R4"])
            run("r4_analyse.py", "--machine", "c1", "--runs", r, "--seed-path-root",
                "/Users/neo/repos/algocell/experiments/runs/r4_inv/r4_inv", "--out", out)
        elif experiment == "L":
            r = view("stageL", ["L"])
            run("assay_batch.py", r, "--machine", "tier2c1_assay")
            run("c4_functional.py", r, "--machine", "tier2c1_assay")
            run("stage_g.py", "--dir", r, "--out", out, "--no-verdicts", "--machine", "tier2c1_assay")
            run("capacity_over_time.py", "--yes", "--stages", "L", "--machine", "tier2c1_assay",
                "--runs-dir", r, "--runs-table", out / "stage_g_runs.csv", "--out", out / "capacity_time_L")
            run("score_L.py", "--cap", out / "capacity_time_L/capacity_over_time.csv",
                "--runs", out / "stage_g_runs.csv", "--out", out / "SCORING.md")
            shutil.copytree(r / "analysis", out / "analysis")
        elif experiment in ("F", "StageE-L10"):
            r = view("stageF" if experiment == "F" else "stageE", [experiment])
            run("assay_batch.py", r, "--machine", "tier2c1_assay")
            shutil.copytree(r / "analysis", out / "analysis")
            if experiment == "F":
                run("stage_f.py", r, "--f4c-only", "--machine", "tier2c1_assay", "--out", out)
            else:
                # The full Stage E plotter assumes other arms and a finite KM
                # median. This registered addendum is tables only. Call its
                # unchanged table functions; no sweep or intrinsic assays.
                helper = Path("/tmp/stage_e_frozen_tables.py")
                helper.write_text('''import sys\nsys.path.insert(0, "/root")\nimport pandas as pd\nfrom pathlib import Path\nfrom stage_e import interaction_clock, cell_rows\nr, out = map(Path, sys.argv[1:])\na = pd.read_csv(r / "analysis/assays.csv")\nclock = interaction_clock(str(r), [str(r / f) for f in a.file])\na = a.merge(clock.rename_axis("file").reset_index(), on="file", how="left")\nt = cell_rows(a, ["ablation", "arm", "tape_len"])\nt.to_csv(out / "size_arms.csv", index=False)\n(out / "NUMBERS_E.md").write_text("# Stage E nominal L10: frozen descriptive table\\n\\n" + t.to_markdown(index=False) + "\\n")\n''')
                record["stage_e_table_adapter_sha256"] = hashlib.sha256(helper.read_bytes()).hexdigest()
                run(str(helper), r, out)
        elif experiment in ("Serial", "Branching"):
            prefix = "sr" if experiment == "Serial" else "branching"
            source = Path("/runs/tier2") / experiment
            for suffix in ("summary.csv", "sites.csv"):
                files = sorted(source.glob(f"*/{prefix}_{suffix}"))
                expected = 14 if experiment == "Serial" else 7
                if len(files) != expected:
                    raise RuntimeError(f"Incomplete {experiment} measurements: {len(files)}/{expected}")
                pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(out / f"{prefix}_{suffix}", index=False)
            usage = [json.loads(p.read_text()) for p in sorted(source.glob("*/executor_usage.json"))]
            (out / "measurement_executor_usage.json").write_text(json.dumps(usage, indent=2) + "\n")
            run("serial_retention.py" if experiment == "Serial" else "branching_transfer.py", "--report", "--out", out)
        else:
            raise ValueError(experiment)
        record["status"] = "complete"
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        record["finished_epoch"] = time.time()
        record["runtime_estimate_usd"] = (record["finished_epoch"] - started) * RATE
        persist()
    return record


COMMON = dict(image=image, gpu="L4", cpu=1, memory=8192, retries=0,
              single_use_containers=True, volumes={"/runs": vol})
short_score = app.function(timeout=600, **COMMON)(score_existing)
long_score = app.function(name="long_score", timeout=9000, **COMMON)(score_existing)


@app.local_entrypoint()
def main(stage_l_scoreable: bool = False, only: str = "", score_tag: str = ""):
    import time
    import subprocess

    # Seven short calls + one long call bound allocated execution at $3.34;
    # $1.66 remains within the user's $5 scoring ceiling for overhead.
    tag = score_tag or time.strftime("c1_%Y%m%dT%H%M%SZ", time.gmtime())
    experiments = ["DZ", "E", "R4", "F", "StageE-L10", "Serial", "Branching"]
    if stage_l_scoreable:
        experiments.append("L")
    if only:
        if only not in experiments or only == "L":
            raise ValueError("Only a specified existing short scorer may be retried")
        experiments = [only]
    record = dict(app_id=app.app_id, score_tag=tag,
                  started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
                  max_allocated_runtime_usd=sum(9000 if e == "L" else 600 for e in experiments) * RATE, calls=[])
    path = DATA / ("score_retry_calls.json" if only else "score_calls.json")
    for experiment in experiments:
        call = (long_score if experiment == "L" else short_score).spawn(experiment, tag)
        record["calls"].append(dict(experiment=experiment, call_id=call.object_id))
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(experiment, call.object_id, flush=True)
    print(json.dumps(record), flush=True)
