"""One-shot patch: shard mode for kaggle_pool.py (idempotent-ish; run once)."""
from pathlib import Path

p = Path("scripts/kaggle_pool.py")
s = p.read_text(encoding="utf-8")

NL = chr(10)  # real newline
Q = chr(34)   # quote

s = s.replace(
    '    ap.add_argument("--gpu", action="store_true", help="enable GPU accelerator (push)")',
    '    ap.add_argument("--gpu", action="store_true", help="enable GPU accelerator (push)")'
    + NL
    + '    ap.add_argument("--shard", action="store_true",'
    + NL
    + '                    help="shard mode: per-account wrapper, shard i -> seq i%7, replicate i//7, full grid, 300 frames")',
)

s = s.replace(
    "    meta = {" + NL + '        "id": f"{user}/{slug}",',
    "    code = code_file or KERNEL_CODE" + NL
    + "    if args_shard:" + NL
    + "        wrapper = workdir / f" + Q + "wrapper_{user}.py" + Q + NL
    + "        lines = [" + NL
    + '            "import os, subprocess, sys",' + NL
    + "            \"os.chdir('/kaggle/working')\"," + NL
    + "            \"if not os.path.exists('scope'):\"," + NL
    + "            \"    subprocess.run(['git', 'clone', 'https://github.com/wagur1/scope.git'])\"," + NL
    + "            \"sys.path.insert(0, '/kaggle/working/scope/kaggle')\"," + NL
    + "            \"from real_e1_kernel import main\"," + NL
    + "            f\"main(shard={seed}, frame_limit=300, \"  + NL"
    + "            \"qp_grid=[22, 24, 27, 29, 32, 34, 37, 39, 42])\"," + NL
    + "        ]" + NL
    + "        wrapper.write_text(NL.join(lines) + NL)" + NL
    + "        code = wrapper" + NL
    + "    meta = {" + NL + '        "id": f"{user}/{slug}",',
)

s = s.replace(
    "def _push(user: str, key: str, seed: int, slug: str = SLUG,"
    + NL
    + "          code_file: Path | None = None, datasets: str = \"\","
    + NL
    + "          gpu: bool = False) -> tuple[bool, str]:",
    "def _push(user: str, key: str, seed: int, slug: str = SLUG,"
    + NL
    + "          code_file: Path | None = None, datasets: str = \"\","
    + NL
    + "          gpu: bool = False, args_shard: bool = False) -> tuple[bool, str]:",
)

s = s.replace(
    "                            datasets=args.datasets, gpu=args.gpu)",
    "                            datasets=args.datasets, gpu=args.gpu,"
    + NL
    + "                            args_shard=args.shard)",
)

p.write_text(s, encoding="utf-8")

import ast
ast.parse(s)
print("pool runner shard mode OK")
