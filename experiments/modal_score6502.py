"""One scoring container per world, with gathered local S1 verdicts.

Default mode uploads ONLY archived DZ lp0 inputs to a unique scratch prefix:
  modal run modal_score6502.py --archive /path/to/runs/dial/dial

After the scorer is committed and released by the coordinator:
  modal run modal_score6502.py --registered --arms A,B,C,D

No directory listing of registered inputs is needed: filenames are predetermined.
Test mode cannot use any prefix other than s6502_scorer_test/<id>.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

import modal

HERE = Path(__file__).resolve().parent
SCORER_FILES = ("score6502.py", "population_classes.py", "mutscan.py", "dial_soups.py",
                "run6502.py", "algocell_exp/executor.py", "algocell_exp/assay.py",
                "algocell_exp/exectrace.py", "algocell_exp/zmachine.py",
                "algocell_exp/m6502_instructions.json")
# Immutable cached dependency image of modal_dial.py (Python 3.12, wgpu 0.32.0,
# numpy 2.5.3, brotli 1.2.0, pandas 3.0.6, Vulkan libraries). Referencing its ID
# permits a Mesa layer before adding source mounts, with no pip/npm installs.
DZ_DEPENDENCY_IMAGE = "im-T2VaU4HVHQXvzDSkbRye1y"
base_image = modal.Image.from_id(DZ_DEPENDENCY_IMAGE)


def source_image(base):
    built = base.add_local_dir(str(HERE / "algocell_exp"), remote_path="/root/algocell_exp")
    for filename in ("score6502.py", "population_classes.py", "mutscan.py", "run6502.py", "dial_soups.py"):
        built = built.add_local_file(str(HERE / filename), remote_path=f"/root/{filename}")
    return built


image = source_image(base_image)
cpu_image = source_image(base_image.apt_install("mesa-vulkan-drivers"))
app = modal.App("algocell-s6502-scorer-v2")
vol = modal.Volume.from_name("algocell-atlas-runs")


def locations(prefix: str, arm: str, seed: int, registered: bool) -> tuple[str, str, str, str, str]:
    """Pure validation before touching the volume, including before reload."""
    if registered:
        from run6502 import ARMS, SEEDS, world_name
        if prefix != "s6502" or arm not in ARMS or seed not in SEEDS:
            raise ValueError("invalid registered prefix, arm or seed")
        return f"s6502/{arm}", f"s6502_scored_v2/{arm}", world_name(arm, seed), ARMS[arm]["machine"], "s1"
    parts = PurePosixPath(prefix).parts
    if (len(parts) != 2 or parts[0] != "s6502_scorer_test"
            or not re.fullmatch(r"[A-Za-z0-9_-]+", parts[1])
            or arm != "DZ" or seed not in range(9001, 9011)):
        raise ValueError("test mode accepts only s6502_scorer_test/<id> and archived lp0 seeds")
    return prefix, f"{prefix}/scored/DZ", f"lp0_s{seed}", "z80c02_assay_L16", "dz"


@app.function(image=image, gpu="L4", cpu=1, memory=2048, timeout=120, max_containers=20, volumes={"/vol": vol},
              retries=0, single_use_containers=True)
def world(prefix: str, arm: str, seed: int, registered: bool, scorer_commit: str) -> dict:
    return _score_volume_world(prefix, arm, seed, registered, scorer_commit)


def _score_volume_world(prefix: str, arm: str, seed: int, registered: bool, scorer_commit: str,
                        *, device=None, adapter_info=None) -> dict:
    import sys
    import time
    sys.path.insert(0, "/root")
    import score6502 as S

    source, destination, name, machine, fmt = locations(prefix, arm, seed, registered)
    if not re.fullmatch(r"[0-9a-f]{40}", scorer_commit):
        raise ValueError("full scorer git commit required")
    vol.reload()
    result_path = Path("/vol") / destination / f"{name}.json"
    if result_path.exists():
        raise FileExistsError(f"scored result exists: {destination}/{name}.json")
    started = time.perf_counter()
    result = S.score_world(Path("/vol") / source / f"{name}.jsonl", machine, fmt, device=device,
                           sensitivities=registered)
    result["modal"] = {"scorer_commit": scorer_commit, "app_id": app.app_id,
                       "input_prefix": source, "result_path": f"{destination}/{name}.json",
                       "wall_s": round(time.perf_counter() - started, 3),
                       "adapter": adapter_info, "software_vulkan": device is not None}
    if device is None:
        from algocell_exp.soup import get_device
        result["modal"]["adapter"] = dict(get_device().adapter.info)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    vol.commit()
    return result


# Scratch-only software Vulkan test path, for validation while all GPUs are busy.
# Same vendored WGSL and the same world() body; no host-side ISA interpreter.
@app.function(image=cpu_image, timeout=4 * 60 * 60,
              volumes={"/vol": vol}, retries=0, single_use_containers=True,
              env={"XDG_RUNTIME_DIR": "/tmp", "WGPU_BACKEND_TYPE": "Vulkan"})
def world_cpu_test(prefix: str, arm: str, seed: int, registered: bool, scorer_commit: str) -> dict:
    if registered:
        raise ValueError("software Vulkan test mode cannot access registered worlds")
    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(force_fallback_adapter=True)
    return _score_volume_world(prefix, arm, seed, False, scorer_commit,
                               device=adapter.request_device_sync(), adapter_info=dict(adapter.info))


def parse_seeds(selection: str) -> list[int]:
    seeds = []
    for part in selection.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(part))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be nonempty and unique")
    return seeds


def committed_scorer() -> str:
    """Refuse production reads when scorer or vendored machines are uncommitted."""
    import subprocess
    for name in (*SCORER_FILES, "modal_score6502.py"):
        frozen = subprocess.check_output(["git", "show", f"HEAD:experiments/{name}"], cwd=HERE)
        if frozen != (HERE / name).read_bytes():
            raise ValueError(f"commit the scorer before scoring registered inputs: {name}")
    diff = subprocess.check_output(["git", "status", "--porcelain", "--", "algocell_exp/machines"], cwd=HERE)
    if diff:
        raise ValueError("commit vendored machine changes before registered scoring")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()


@app.local_entrypoint()
def main(archive: str = "", out: str = "results/s6502/v2", registered: bool = False,
         arms: str = "A,B,C,D", seeds: str = "", prefix: str = "", gather_only: bool = False,
         cpu_test: bool = False):
    import subprocess
    import uuid
    import score6502 as S

    if cpu_test and registered:
        raise ValueError("--cpu-test is restricted to scratch DZ validation")

    selected_seeds = parse_seeds(seeds or ("65001-65020" if registered else "9001-9010"))
    selected_arms = arms.split(",") if registered else ["DZ"]
    if not selected_arms or len(selected_arms) != len(set(selected_arms)):
        raise ValueError("arms must be nonempty and unique")
    if gather_only and not registered and not prefix:
        raise ValueError("scratch --gather-only requires the exact test --prefix")
    prefix = prefix or ("s6502" if registered else f"s6502_scorer_test/{uuid.uuid4().hex}")
    jobs = [(prefix, arm, seed, registered) for arm in selected_arms for seed in selected_seeds]
    for args in jobs:
        locations(*args)
    commit = (committed_scorer() if registered else
              subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip())
    if not registered and not gather_only:
        if not archive:
            raise ValueError("test mode requires --archive (archived Z80 DZ directory)")
        archive_path = Path(archive).resolve()
        # Upload exactly these known Z80 files; never list the volume or archive.
        with vol.batch_upload() as upload:
            for seed in selected_seeds:
                for filename in (f"lp0_s{seed}.jsonl", f"lp0_s{seed}_t300000.npy"):
                    upload.put_file(str(archive_path / filename), f"/{prefix}/{filename}")
    output = Path(out)
    results = []
    print(json.dumps({"app_id": app.app_id, "input_prefix": prefix, "worlds": len(jobs),
                      "registered": registered, "commit": commit}), flush=True)
    # Separate single-use containers, with explicit call IDs and awaited results.
    # Gather reads the committed JSON back from the volume, verifying round trips.
    calls = []
    scoring_world = world_cpu_test if cpu_test else world
    if not gather_only:
        for args in jobs:
            call = scoring_world.spawn(*args, commit)
            calls.append(call)
            print(json.dumps({"function_id": scoring_world.object_id, "call_id": call.object_id,
                              "arm": args[1], "seed": args[2]}), flush=True)
        try:
            print(f"test/scoring queue: {scoring_world.get_current_stats()}", flush=True)
        except Exception as error:
            print(f"queue stats unavailable: {type(error).__name__}", flush=True)
    for i, args in enumerate(jobs):
        returned = calls[i].get() if calls else None
        _, destination, name, _, _ = locations(*args)
        result = json.loads(b"".join(vol.read_file(f"/{destination}/{name}.json")))
        if returned is not None and returned != result:
            raise ValueError(f"volume result differs from returned result: {name}")
        S.write_json(output / "worlds" / f"{name}.json", result)
        results.append(result)
        print(json.dumps({"world": name, "wall_s": result["modal"]["wall_s"],
                          "volume_roundtrip": True}), flush=True)
    S.report(results, output, report_name="S1_v2.md" if registered else "S1.md")
    if not registered:
        complete_control = set(selected_seeds) == set(range(9001, 9011))
        matched = S.control_report(results, HERE / "results/dial/dial_worlds.csv", output) if complete_control else None
        lines = ["# Modal scoring-wrapper test (generated)", "",
                 f"App: `{app.app_id}`; scratch input prefix: `{prefix}`.",
                 ("One single-use CPU/software-Vulkan container per archived Z80 world (Mesa), retries=0; all calls awaited."
                  if cpu_test else "One single-use GPU container per archived Z80 world (L40S/A10G/L4), retries=0; all calls awaited."),
                 "Results read back from the scratch volume and compared exactly with each returned result, "
                 "then gathered locally for tables/verdicts.", "",
                 "| world | seconds | saved volume result | machine-only execution |", "|---|---:|---|---|"]
        for r in sorted(results, key=lambda r: r["world"]["seed"]):
            lines.append(f"| {r['world']['world']} | {r['modal']['wall_s']} | {r['modal']['result_path']} | "
                         f"{not bool(r['usage']['shader'])} |")
        lines += ["", f"Full positive control: {'PASS' if matched else 'FAIL' if matched is False else 'subset only'}.",
                  "Test-mode path validation rejects s6502/, path traversal, non-DZ arms and registered seeds.",
                  "No registered 6502 output was read, fetched, listed or scored."]
        (output / "MODAL_SCORER_TEST.md").write_text("\n".join(lines) + "\n")
        if matched is False:
            raise SystemExit("positive control failed; investigate before committing")
