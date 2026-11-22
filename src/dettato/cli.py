"""dettato FILE_OR_FOLDER ... -o OUT_DIR

Writes OUT_DIR/<name>.mid (one track per instrument + a drum track) and OUT_DIR/<name>.json (a summary) for every audio file.
Folders are searched recursively and their structure is kept under OUT_DIR."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .audio import EXTENSIONS


def gather(inputs):
    out = []
    for p in inputs:
        if os.path.isdir(p):
            for root, _, files in sorted(os.walk(p)):
                for f in sorted(files):
                    if f.lower().endswith(EXTENSIONS):
                        full = os.path.join(root, f)
                        out.append((os.path.splitext(os.path.relpath(full, p))[0], full))
        else:
            out.append((os.path.splitext(os.path.basename(p))[0], p))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dettato", description="Audio to multi-track MIDI.")
    ap.add_argument("inputs", nargs="+", help="audio files or folders")
    ap.add_argument("-o", "--out", default="dettato_out", help="output folder (default: dettato_out)")
    ap.add_argument("--device", default=None, help="cuda, mps, cpu, cuda:1, ... (default: cuda if available, else mps, else cpu)")
    a = ap.parse_args(argv)
    songs = gather(a.inputs)
    if not songs:
        sys.exit("no audio files found")
    from .pipeline import Dettato

    model = Dettato.from_pretrained(a.device)
    print(f"dettato: {len(songs)} file(s) on {model.device}", flush=True)
    failed = 0
    for name, path in songs:
        t0 = time.time()
        try:
            res = model.transcribe(path)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAILED {path}: {e!r}", flush=True)
            continue
        base = os.path.join(a.out, name)
        res.save_midi(base + ".mid")
        with open(base + ".json", "w") as fh:
            json.dump(res.summary(), fh, indent=1)
        tracks = ", ".join(f"{t.name} ({len(t.notes)})" for t in res.tracks)
        print(f"{name}: {res.seconds:.0f} s in {time.time() - t0:.0f} s -> {base}.mid: {tracks}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
