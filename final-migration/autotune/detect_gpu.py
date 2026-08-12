#!/usr/bin/env python3
"""Query one CUDA ordinal and select the frozen autotune workflow."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


def smi_index(device: dict[str, object]) -> int:
    pci = device["pci"]
    assert isinstance(pci, dict)
    wanted = f"{int(pci['domain']):08X}:{int(pci['bus']):02X}:{int(pci['device']):02X}.0"
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    for line in output.splitlines():
        index, bus = (part.strip() for part in line.split(",", 1))
        if bus.upper() == wanted:
            return int(index)
    raise RuntimeError(f"CUDA device PCI identity {wanted} is absent from nvidia-smi")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--print-smi-index", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "python"))
    from portable_cuda_device import query_cuda_device
    from autotune import classify_device

    device = query_cuda_device(args.device)
    if args.print_smi_index:
        print(smi_index(device))
        return 0
    try:
        classification = classify_device(device)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    payload = {"schema": 1, **classification, "device": device}
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
