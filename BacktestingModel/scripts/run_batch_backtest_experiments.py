import argparse
import glob
import os
import subprocess
from pathlib import Path


def run(cmd, log_path=None):
    print("\n$ " + " ".join(cmd))

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="config/experiments/generated")
    parser.add_argument("--pattern", default="*.yaml")
    parser.add_argument("--skip-fill", action="store_true")
    parser.add_argument("--skip-enrich", action="store_true")
    parser.add_argument("--skip-pnl", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--max-quotes", default="")
    parser.add_argument("--background-logs", action="store_true")
    args = parser.parse_args()

    exp_files = sorted(glob.glob(str(Path(args.exp_dir) / args.pattern)))

    if not exp_files:
        raise RuntimeError(f"No experiment yaml found under {args.exp_dir} with pattern {args.pattern}")

    print("experiments:")
    for p in exp_files:
        print(" ", p)

    for exp in exp_files:
        tag = Path(exp).stem
        cmd = [
            "python",
            "scripts/run_backtest_experiment.py",
            "--exp",
            exp,
        ]

        if args.skip_fill:
            cmd.append("--skip-fill")
        if args.skip_enrich:
            cmd.append("--skip-enrich")
        if args.skip_pnl:
            cmd.append("--skip-pnl")
        if args.skip_final:
            cmd.append("--skip-final")
        if args.max_quotes:
            cmd += ["--max-quotes", str(args.max_quotes)]

        log_path = None
        if args.background_logs:
            log_path = f"logs/batch_backtest_{tag}.log"

        run(cmd, log_path=log_path)

    print("\nall experiments finished")


if __name__ == "__main__":
    main()
