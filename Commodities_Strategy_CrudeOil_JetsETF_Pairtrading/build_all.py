"""Build-Pipeline des Strategy Labs.

Aufruf:
    python build_all.py            # alle Reports
    python build_all.py 00 41      # nur ausgewählte Reports
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "reports"
sys.path.insert(0, str(ROOT))

REGISTRY: dict[str, tuple[str, str]] = {
    "00": ("reports.r00_baseline", "Baseline-Reproduktion & Diagnose"),
    "41": ("reports.r41_sector_rotation", "Sektor-Rotation Deep Dive"),
    "42": ("reports.r42_pandemic_monitor", "Pandemie- & Kriegs-Frühwarnsystem"),
    "43": ("reports.r43_flash_crash_opt", "Flash-Crash-Optimierung"),
    "44": ("reports.r44_structures_grid", "Strukturvarianten, Kombinationen & Grid"),
    "99": ("reports.r99_index", "Dashboard-Index"),
}


def main(argv: list[str]) -> int:
    import numpy as np
    np.random.seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    keys = [a for a in argv if a in REGISTRY] or list(REGISTRY)
    failed = []
    for k in keys:
        mod_name, label = REGISTRY[k]
        print(f"\n=== [{k}] {label} " + "=" * (46 - len(label)))
        t0 = time.time()
        try:
            mod = __import__(mod_name, fromlist=["build"])
            mod.build(OUT)
            print(f"    fertig in {time.time() - t0:.1f}s")
        except Exception:  # noqa: BLE001
            failed.append(k)
            traceback.print_exc()
    print("\n" + "=" * 60)
    print(f"Ausgabeverzeichnis: {OUT}")
    if failed:
        print(f"FEHLGESCHLAGEN: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
