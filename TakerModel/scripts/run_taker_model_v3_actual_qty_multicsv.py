import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def get_nested(cfg, keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True, help="Original v3 TakerModel config yaml")
    ap.add_argument("--positions", default=None, help="Optional existing v3 positions csv")
    ap.add_argument("--output_dir", default=None, help="Actual-qty output directory")
    ap.add_argument("--tag", default=None, help="Output tag")
    ap.add_argument("--run_v3_first", action="store_true", help="Run original v3 first, then actual-qty accounting")

    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--fee_bps", type=float, default=None)
    ap.add_argument("--slippage_bps", type=float, default=None)
    ap.add_argument("--clip_short_sell", action="store_true")

    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. infer positions path
    if args.positions is not None:
        pos_path = Path(args.positions)
    else:
        pos_path = get_nested(cfg, ["output", "position_output_path"])
        if pos_path is None:
            pos_path = get_nested(cfg, ["output", "positions_output_path"])
        if pos_path is None:
            pos_path = get_nested(cfg, ["output", "positions_path"])
        if pos_path is None:
            raise RuntimeError(
                "Cannot infer positions path from config. "
                "Please pass --positions explicitly."
            )
        pos_path = Path(pos_path)

    # 2. optionally run original v3 first
    if args.run_v3_first:
        v3_script = Path("TakerModel/scripts/run_taker_model_v3_exit_control_multicsv.py")
        if not v3_script.exists():
            raise FileNotFoundError(v3_script)

        cmd = [
            sys.executable,
            str(v3_script),
            "--config",
            str(config_path),
        ]

        print("running original v3 first:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    if not pos_path.exists:
        raise FileNotFoundError(pos_path)

    if not pos_path.exists():
        raise FileNotFoundError(
            f"positions file not found: {pos_path}\n"
            f"Either run original v3 first or pass --run_v3_first."
        )

    # 3. infer output dir
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        if pos_path.parent.name == "positions":
            out_dir = pos_path.parent.parent / "actual_qty"
        else:
            out_dir = pos_path.parent / "actual_qty"

    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. infer tag
    if args.tag is not None:
        tag = args.tag
    else:
        tag = pos_path.stem
        tag = tag.replace("taker_positions_feature_transformer_h120_oos_", "")
        tag = tag.replace("taker_positions_", "")
        tag = tag + "_actual_qty"

    # 5. infer costs
    capital = args.capital
    if capital is None:
        capital = get_nested(cfg, ["backtest", "capital"])
    if capital is None:
        capital = get_nested(cfg, ["execution", "capital"])
    if capital is None:
        capital = get_nested(cfg, ["portfolio", "capital"])
    if capital is None:
        capital = 200_000_000

    fee_bps = args.fee_bps
    if fee_bps is None:
        fee_bps = get_nested(cfg, ["cost", "fee_bps"])
    if fee_bps is None:
        fee_bps = get_nested(cfg, ["costs", "fee_bps"])
    if fee_bps is None:
        fee_bps = get_nested(cfg, ["execution", "fee_bps"])
    if fee_bps is None:
        fee_bps = 0.5

    slippage_bps = args.slippage_bps
    if slippage_bps is None:
        slippage_bps = get_nested(cfg, ["cost", "slippage_bps"])
    if slippage_bps is None:
        slippage_bps = get_nested(cfg, ["costs", "slippage_bps"])
    if slippage_bps is None:
        slippage_bps = get_nested(cfg, ["execution", "slippage_bps"])
    if slippage_bps is None:
        slippage_bps = 0.0

    v4_script = Path("TakerModel/scripts/run_taker_model_v4_actual_qty_from_positions.py")
    if not v4_script.exists():
        raise FileNotFoundError(
            f"{v4_script} not found. "
            f"Please create v4 actual-qty script first."
        )

    cmd = [
        sys.executable,
        str(v4_script),
        "--positions",
        str(pos_path),
        "--output_dir",
        str(out_dir),
        "--tag",
        tag,
        "--capital",
        str(float(capital)),
        "--fee_bps",
        str(float(fee_bps)),
        "--slippage_bps",
        str(float(slippage_bps)),
    ]

    if args.clip_short_sell:
        cmd.append("--clip_short_sell")

    print("\n===== v3 actual-qty wrapper =====")
    print("config    :", config_path)
    print("positions :", pos_path)
    print("output_dir:", out_dir)
    print("tag       :", tag)
    print("capital   :", capital)
    print("fee_bps   :", fee_bps)
    print("slip_bps  :", slippage_bps)
    print("\nrunning actual-qty accounting:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print("\nfinished.")
    print("summary should be:")
    print(out_dir / f"share_level_actual_qty_summary_{tag}.csv")
    print("daily should be:")
    print(out_dir / f"share_level_actual_qty_daily_{tag}.csv")
    print("minute should be:")
    print(out_dir / f"share_level_actual_qty_minute_{tag}.csv")


if __name__ == "__main__":
    main()
