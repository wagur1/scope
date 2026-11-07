#!/usr/bin/env python
"""Kaggle token-pool runner: push / status / fetch the E1 smoke kernel across
a pool of accounts (one kernel per account -> synthetic-grid replication +
determinism cross-check).

Pool file (JSON: {username: api_key}) is NEVER committed — pass its path on
the command line; kernels are pushed PRIVATE.

Usage:
  python scripts/kaggle_pool.py push  <pool.json> [--only u1,u2] [--limit N]
  python scripts/kaggle_pool.py status <pool.json>
  python scripts/kaggle_pool.py fetch <pool.json> [-o results/kaggle]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNEL_CODE = REPO / "kaggle" / "kernel_driver.py"
SLUG = "scope-e1-smoke"


def _env(user: str, key: str) -> dict:
    """Isolated credential env per account.

    kaggle CLI 2.x prefers KAGGLE_API_TOKEN over kaggle.json /
    KAGGLE_CONFIG_DIR / KAGGLE_USERNAME+KEY (verified: a stray global
    KAGGLE_API_TOKEN hijacked every push to one account). So the pool key
    MUST be injected as KAGGLE_API_TOKEN, with the legacy vars aligned too.
    """
    cfgdir = Path(os.environ.get("KAGGLE_POOL_TMP", Path.home() / ".kaggle" / "pool")) / user
    cfgdir.mkdir(parents=True, exist_ok=True)
    (cfgdir / "kaggle.json").write_text(json.dumps({"username": user, "key": key}))
    return dict(os.environ,
                KAGGLE_API_TOKEN=key,
                KAGGLE_CONFIG_DIR=str(cfgdir),
                KAGGLE_USERNAME=user, KAGGLE_KEY=key)


def _kaggle(user: str, key: str, args: list[str]) -> subprocess.CompletedProcess:
    # `kaggle` console script may be absent from PATH on Windows; module entry works
    return subprocess.run([sys.executable, "-m", "kaggle", *args], env=_env(user, key),
                          capture_output=True, text=True)


def _push(user: str, key: str, seed: int, slug: str = SLUG,
          code_file: Path | None = None, datasets: str = "",
          gpu: bool = False) -> tuple[bool, str]:
    """Push a kernel as `user` (private). seed differs per account ->
    same-seed blocks across accounts double as the encoder-determinism check."""
    workdir = Path(os.environ.get("KAGGLE_POOL_TMP", Path.home() / ".kaggle" / "pool")) / user
    workdir.mkdir(parents=True, exist_ok=True)
    args_datasets = datasets
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": str((code_file or KERNEL_CODE).resolve()),
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": bool(gpu),
        "enable_internet": True,
        "dataset_sources": [d for d in args_datasets.split(",") if d],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (workdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    proc = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(workdir)],
                          capture_output=True, text=True, env=_env(user, key))
    text = (proc.stdout + proc.stderr).strip()
    ok = proc.returncode == 0 and "successfully" in text.lower()
    return ok, text.splitlines()[-1] if text else "no output"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["push", "status", "fetch"])
    ap.add_argument("pool", help="path to pool json (never committed)")
    ap.add_argument("--only", help="comma-separated usernames subset")
    ap.add_argument("--limit", type=int, default=0, help="use first N accounts")
    ap.add_argument("-o", "--out", default="results/kaggle")
    ap.add_argument("--slug", default=SLUG, help="kernel slug (push/status/fetch)")
    ap.add_argument("--code", default=None, help="kernel code file (push)")
    ap.add_argument("--datasets", default="", help="comma-separated dataset_sources (push)")
    ap.add_argument("--gpu", action="store_true", help="enable GPU accelerator (push)")
    ap.add_argument("--wait", action="store_true", help="push: poll until complete")
    args = ap.parse_args()

    pool = json.loads(Path(args.pool).read_text())
    users = list(pool.items())
    if args.only:
        keep = {u.strip() for u in args.only.split(",")}
        users = [(u, k) for u, k in users if u in keep]
    if args.limit:
        users = users[: args.limit]
    print(f"pool: {len(users)} accounts")

    if args.command == "push":
        for i, (user, key) in enumerate(users):
            ok, msg = _push(user, key, seed=i, slug=args.slug,
                            code_file=Path(args.code) if args.code else None,
                            datasets=args.datasets, gpu=args.gpu)
            print(f"push {user}: {'OK' if ok else 'FAIL'} | {msg}")
            time.sleep(1.0)
        if args.wait:
            print("\npolling (60s interval)...")
            done = {}
            while len(done) < len(users):
                for user, key in users:
                    if user in done:
                        continue
                    s = _kaggle(user, key, ["kernels", "status", f"{user}/{args.slug}"])
                    text = (s.stdout + s.stderr).strip()
                    state = text.split()[-1] if text else "?"
                    if any(k in text.lower() for k in ("complete", "error", "cancel")):
                        done[user] = state
                        print(f"  {user}: {state}")
                if len(done) < len(users):
                    time.sleep(60)
            print(json.dumps(done, indent=2))

    elif args.command == "status":
        for user, key in users:
            s = _kaggle(user, key, ["kernels", "status", f"{user}/{args.slug}"])
            line = (s.stdout + s.stderr).strip().splitlines()[-1] if (s.stdout + s.stderr).strip() else "?"
            print(f"{user}: {line}")

    elif args.command == "fetch":
        out_root = Path(args.out)
        out_root.mkdir(parents=True, exist_ok=True)
        for user, key in users:
            dest = out_root / user
            r = _kaggle(user, key, ["kernels", "output", f"{user}/{args.slug}", "-p", str(dest)])
            ok = r.returncode == 0 and (dest / "summary.json").exists()
            print(f"fetch {user}: {'OK' if ok else 'EMPTY/FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
