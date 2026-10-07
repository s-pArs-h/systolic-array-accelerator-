#!/usr/bin/env python3
"""Area sweep: synthesise systolic_array for several array sizes with Yosys
(Xilinx 7-series mapping) and tabulate resources and peak throughput.

Resource numbers are estimates from the open-source flow; Fmax and final
utilisation come from Vivado (see README).

usage: python3 synth/sweep.py [N ...]          (default: 2 4 8 16)
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RTL = [ROOT / "rtl" / f for f in ("sa_pe.sv", "sa_skew.sv", "systolic_array.sv")]
REPORTS = ROOT / "synth" / "reports"
CELLS = {"LUT": r"LUT[1-6]", "FF": r"FD[CPRSE]+", "DSP48E1": r"DSP48E1", "CARRY4": r"CARRY4"}


def synth(n):
    REPORTS.mkdir(parents=True, exist_ok=True)
    script = (f"read_verilog -sv {' '.join(str(p) for p in RTL)}; "
              f"chparam -set N {n} systolic_array; "
              "synth_xilinx -family xc7 -top systolic_array -flatten; stat")
    out = subprocess.run(["yosys", "-p", script], capture_output=True, text=True, check=True).stdout
    (REPORTS / f"artix7_N{n}.log").write_text(out)
    stat = out[out.rindex("Printing statistics"):]
    return {name: sum(int(c) for _, c in re.findall(rf"^\s+({pat})\s+(\d+)$", stat, re.M))
            for name, pat in CELLS.items()}


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [2, 4, 8, 16]
    lines = ["| N | PEs | LUTs | FFs | DSP48E1 | CARRY4 | MACs / cycle | LUTs per PE | full-rate K |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n in sizes:
        c = synth(n)
        lines.append(f"| {n} | {n*n} | {c['LUT']} | {c['FF']} | {c['DSP48E1']} | {c['CARRY4']} "
                     f"| {n*n} | {c['LUT'] / (n*n):.0f} | >= {3*n + 1} |")
    table = "\n".join(lines)
    (REPORTS / "sweep.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
