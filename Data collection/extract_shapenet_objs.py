#!/usr/bin/env python3
"""
extract_shapenet_objs.py  –  ShapeNetCore v2 extractor

• Downloads each synset ZIP from a Hugging Face dataset
• Extracts every   <synset>/<model>/models/model_normalized.obj
• Writes it as     <output_dir>/<model>.obj
• Deletes the ZIP immediately to save disk space
• Safe to interrupt and resume

Requires: huggingface_hub ≥ 0.22
"""

import argparse, multiprocessing as mp, os, sys, zipfile, shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

# --------------------------------------------------------------------- #
def list_zips(repo_id: str):
    api = HfApi()
    return [f for f in api.list_repo_files(repo_id, repo_type="dataset")
            if f.endswith(".zip")]

# --------------------------------------------------------------------- #
def process_zip(job):
    zrel, repo_id, out_dir = job
    try:
        print(f"[DL ] {zrel}", flush=True)
        zlocal = hf_hub_download(
            repo_id         = repo_id,
            filename        = zrel,
            repo_type       = "dataset",
            force_filename  = Path(zrel).name,   # readable cache name
            resume_download = True
        )
    except Exception as e:
        print(f"[ERR] {zrel}: {e}", flush=True)
        return 0

    count = 0
    try:
        with zipfile.ZipFile(zlocal) as zf:
            for member in zf.namelist():
                if not member.endswith("models/model_normalized.obj"):
                    continue
                parts = Path(member).parts
                if len(parts) < 4:
                    continue
                model_id = parts[1]                 # REAL model ID
                dst = out_dir / f"{model_id}.obj"
                tmp = dst.with_suffix(".tmp")
                with zf.open(member) as src, open(tmp, "wb") as dstf:
                    shutil.copyfileobj(src, dstf, 1 << 20)
                tmp.rename(dst)
                count += 1
    except zipfile.BadZipFile:
        print(f"[WARN] corrupt ZIP skipped: {zrel}", flush=True)

    try:
        os.remove(zlocal)
    except Exception:
        pass
    print(f"[DONE] {zrel} → {count} OBJ", flush=True)
    return count

# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id",   required=True,
                    help="e.g. ShapeNet/ShapeNetCore")
    ap.add_argument("--output-dir", default="models_flat",
                    help="destination folder for OBJ files")
    ap.add_argument("--num-procs", type=int, default=8,
                    help="parallel workers (match CPU cores)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    zips = list_zips(args.repo_id)            # add synset filter here if needed
    print(f"{len(zips)} archives; OBJs → {out_dir}\n", flush=True)

    total = 0
    with mp.Pool(args.num_procs) as pool:
        for n in pool.imap_unordered(process_zip,
                                     [(z, args.repo_id, out_dir) for z in zips]):
            total += n
    print(f"\nFinished: {total} OBJ files in {out_dir}", flush=True)

# --------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
