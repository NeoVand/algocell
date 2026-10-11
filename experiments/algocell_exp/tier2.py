"""Explicit, checked I1-T2 world selection. No selection changes the frozen shader path."""
from __future__ import annotations

import json
from functools import lru_cache

from .impact_worlds import MACHINES_DIR, WORLDS


def tag_for(L: int, *, variant=None, lethal=False, P=None) -> str:
    if variant:
        return f"{variant}_L{L}"
    return f"lethal_L{L}" if lethal else f"L{L}" + (f"_P{P}" if P and P != 2 * L else "")


def machine_for(selection: str, tag: str, *, maps=False) -> str:
    """c0/c1 select the matching world; full names must match its complete execution spec."""
    name = (assay_name(tag, selection) if maps else f"imp_{tag}_{selection}") if selection in ("c0", "c1") else selection
    manifest = json.loads((MACHINES_DIR / f"{name}.manifest.json").read_text())
    actual = dict(manifest["world"])
    expected = dict(WORLDS[tag].spec())
    # Maps are observation only; every other world field, including raw PCG hooks, must match.
    actual.pop("outputs", None)
    expected.pop("outputs", None)
    if actual != expected or manifest["core"]["id"] != "z80":
        raise ValueError(f"{name}: world does not match {tag}")
    if maps and not all(k in {f["name"] for f in manifest["record"]["fields"]} for k in ("fetchMap", "startMap")):
        raise ValueError(f"{name}: scoring needs fetchMap and startMap")
    return name


def assay_name(tag: str, core: str) -> str:
    if tag.startswith("L"):
        return f"tier2{core}_assay_{tag}"
    if tag.startswith("lethal_"):
        return f"tier2{core}_assay_lethal_{tag[7:]}"
    return f"tier2_{tag}_{core}_assay"


def soup_for(selection: str, L: int, seed: int, *, variant=None, lethal=False,
             mutations_per_step=None, P=None):
    from .zsoup import ZSoup

    name = machine_for(selection, tag_for(L, variant=variant, lethal=lethal, P=P))
    return ZSoup(name, seed=seed, mutations_per_step=mutations_per_step)


@lru_cache(maxsize=None)
def assay_machine(selection: str, L: int, variant=None):
    from .zmachine import load_machine

    return load_machine(machine_for(selection, tag_for(L, variant=variant), maps=True))


def execute_convention(pairs, L, steps, variant, seed, suppress, traced, selection):
    from .executor import pack_fetch_masks
    from .isa import masks, resolve

    m = assay_machine(selection, L, variant)
    r = m.run(pairs, budget=steps, seed=seed, lane_base=0, ablation=masks(resolve(list(suppress))))
    return (r.memory, pack_fetch_masks(r.fetch_map)) if traced else r.memory


def check_run_core(record: dict, selection: str, tag: str) -> None:
    """A relocated C1 scorer must never silently score a C0-world archive as a C1 run."""
    desired = machine_for(selection, tag)
    used = record.get("machine")
    used = used.get("machine") if isinstance(used, dict) else used
    if used != desired:
        raise ValueError(f"scoring {desired}, but run metadata names {used!r}")


def check_batch_core(record: dict, spec: str | None) -> None:
    """Tier-2 batch scoring requires a world run on the chosen core, including ring/rule."""
    if spec not in ("tier2c0_assay", "tier2c1_assay"):
        return
    core = "c0" if spec == "tier2c0_assay" else "c1"
    L = int(record["tape_length"])
    tag = tag_for(L, lethal=record.get("zero_halts", False), P=record.get("mem_length"))
    check_run_core(record, core, tag)
