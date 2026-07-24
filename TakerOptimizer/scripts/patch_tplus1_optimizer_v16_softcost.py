# -*- coding: utf-8 -*-
from pathlib import Path
import shutil

ROOT = Path("/mnt/data1/fwz/HFT_010-dev_fwz")
SRC = ROOT / "TakerOptimizer/scripts/run_tplus1_aware_pure_cs_optimizer.py"
DST = ROOT / "TakerOptimizer/scripts/run_tplus1_aware_pure_cs_optimizer_v16_softcost.py"

if not SRC.exists():
    raise FileNotFoundError(SRC)

text = SRC.read_text().replace("\r\n", "\n")
lines = text.splitlines()

def find_line(lines, pred, start=0):
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    return -1

def remove_duplicate_warm_start(lines):
    marker = "# ===== WARM START BENCHMARK PORTFOLIO PATCH ====="
    out = lines[:]
    while True:
        s = find_line(out, lambda x: marker in x)
        if s < 0:
            break

        e = -1
        for j in range(s, len(out)):
            if 'print("================================\\n")' in out[j]:
                e = j + 1
                break

        if e < 0:
            # fallback: remove until main loop init if the exact print line is not found
            for j in range(s, len(out)):
                if "current_date = None" in out[j] or "target_rows = []" in out[j]:
                    e = j
                    break

        if e < 0:
            raise RuntimeError("found duplicate warm-start marker but cannot determine block end")

        del out[s:e]
    return out

def insert_after_out_dataframe(lines):
    if any('out["soft_cost_signal"]' in x for x in lines):
        return lines

    s = find_line(lines, lambda x: "out = pd.DataFrame({" in x)
    if s < 0:
        raise RuntimeError("cannot find out = pd.DataFrame({")

    bal = 0
    started = False
    e = -1
    for j in range(s, len(lines)):
        line = lines[j]
        for ch in line:
            if ch == "(":
                bal += 1
                started = True
            elif ch == ")":
                bal -= 1
        if started and bal <= 0:
            e = j
            break

    if e < 0:
        raise RuntimeError("cannot find end of out = pd.DataFrame({...}) block")

    indent = lines[s].split("out = pd.DataFrame", 1)[0]
    block = [
        "",
        f'{indent}out["soft_cost_signal"] = (',
        f'{indent}    df[soft_col].astype(float).values',
        f'{indent}    if soft_col is not None',
        f'{indent}    else np.zeros(len(df), dtype=float)',
        f'{indent})',
    ]
    return lines[:e + 1] + block + lines[e + 1:]

def insert_after_line(lines, idx, block):
    return lines[:idx + 1] + block + lines[idx + 1:]

# 1) clean duplicate warm-start block in generated v16 file
lines = remove_duplicate_warm_start(lines)

# 2) add soft_col pick_col after sig_col
if not any("soft_col = pick_col(" in x for x in lines):
    i = find_line(lines, lambda x: "sig_col = pick_col" in x and 'name="signal"' in x)
    if i < 0:
        raise RuntimeError("cannot find sig_col pick_col line")
    indent = lines[i].split("sig_col", 1)[0]
    block = [
        "",
        f"{indent}soft_col = pick_col(",
        f"{indent}    df,",
        f'{indent}    ["soft_cost_signal", "pred_ret_h10", "pred_ret_10", "pred_h10", "alpha_h10", "signal_h10"],',
        f"{indent}    required=False,",
        f'{indent}    name="soft cost signal",',
        f"{indent})",
    ]
    lines = insert_after_line(lines, i, block)

# 3) add standardized out["soft_cost_signal"] after DataFrame construction
lines = insert_after_out_dataframe(lines)

# 4) add argparse args
if not any("--soft-cost-scale" in x for x in lines):
    i = find_line(lines, lambda x: 'ap.add_argument("--lambda-ridge"' in x)
    if i < 0:
        raise RuntimeError("cannot find --lambda-ridge argument line")
    indent = lines[i].split("ap.add_argument", 1)[0]
    block = [
        f'{indent}ap.add_argument("--soft-cost-scale", type=float, default=0.0005)',
        f'{indent}ap.add_argument("--soft-cost-signal-clip", type=float, default=5.0)',
        f'{indent}ap.add_argument("--use-execution-cost-in-objective", type=int, default=1)',
    ]
    lines = insert_after_line(lines, i, block)

# 5) add soft-cost arrays after alpha signal line
if not any("V16 SOFTCOST PATCH START" in x for x in lines):
    i = find_line(lines, lambda x: 'signal = zscore_signal(cur["signal"].values' in x)
    if i < 0:
        raise RuntimeError("cannot find signal scaling line in solve_one_rebalance")
    indent = lines[i].split("signal", 1)[0]
    block = [
        "",
        f"{indent}# ===== V16 SOFTCOST PATCH START =====",
        f'{indent}soft_raw = cur["soft_cost_signal"].values if "soft_cost_signal" in cur.columns else np.zeros(n, dtype=float)',
        f"{indent}soft_trade_signal = zscore_signal(soft_raw, float(args.soft_cost_signal_clip)) * float(args.soft_cost_scale)",
        "",
        f'{indent}mid_px = cur["mid_price"].values.astype(float)',
        f'{indent}bid_px = cur["bid_price"].values.astype(float)',
        f'{indent}ask_px = cur["ask_price"].values.astype(float)',
        f"{indent}mid_safe = np.where(mid_px > 0, mid_px, np.maximum((bid_px + ask_px) / 2.0, 1e-12))",
        f"{indent}fee_rate_for_obj = float(args.fee_bps) / 10000.0",
        "",
        f"{indent}buy_exec_cost = np.zeros(n, dtype=float)",
        f"{indent}sell_exec_cost = np.zeros(n, dtype=float)",
        f'{indent}if int(getattr(args, "use_execution_cost_in_objective", 1)) == 1:',
        f"{indent}    buy_exec_cost = np.maximum(ask_px / mid_safe - 1.0, 0.0) + fee_rate_for_obj",
        f"{indent}    sell_exec_cost = np.maximum(1.0 - bid_px / mid_safe, 0.0) + fee_rate_for_obj",
        "",
        f"{indent}buy_cost = float(args.lambda_turnover) + buy_exec_cost - soft_trade_signal",
        f"{indent}sell_cost = float(args.lambda_turnover) + sell_exec_cost + soft_trade_signal",
        f"{indent}# ===== V16 SOFTCOST PATCH END =====",
    ]
    lines = insert_after_line(lines, i, block)

# 6) replace objective block by line positions
s = find_line(lines, lambda x: x.strip() == "obj = (")
if s < 0:
    raise RuntimeError("cannot find obj = ( line")

e = find_line(lines, lambda x: "prob = cp.Problem(cp.Maximize(obj), constraints)" in x, start=s)
if e < 0:
    raise RuntimeError("cannot find prob = cp.Problem line after obj")

indent = lines[s].split("obj", 1)[0]
new_obj = [
    f"{indent}obj = (",
    f"{indent}    signal @ w",
    f"{indent}    - cp.sum(cp.multiply(buy_cost, buy))",
    f"{indent}    - cp.sum(cp.multiply(sell_cost, sell))",
    f"{indent}    - float(args.lambda_active) * cp.sum(active_abs)",
    f"{indent}    - float(args.lambda_ridge) * cp.sum_squares(w - bench)",
    f"{indent})",
    "",
    f"{indent}prob = cp.Problem(cp.Maximize(obj), constraints)",
]
lines = lines[:s] + new_obj + lines[e + 1:]

# 7) add solution diagnostics
if not any('"avg_buy_cost"' in x for x in lines):
    i = find_line(lines, lambda x: '"max_affordable_gross": max_affordable_gross,' in x)
    if i < 0:
        raise RuntimeError("cannot find max_affordable_gross return line")
    indent = lines[i].split('"max_affordable_gross"', 1)[0]
    block = [
        f'{indent}"avg_buy_cost": float(np.mean(buy_cost)),',
        f'{indent}"avg_sell_cost": float(np.mean(sell_cost)),',
        f'{indent}"avg_soft_trade_signal": float(np.mean(soft_trade_signal)),',
        f'{indent}"std_soft_trade_signal": float(np.std(soft_trade_signal)),',
    ]
    lines = insert_after_line(lines, i, block)

# 8) add summary diagnostics, best effort
if not any('"avg_buy_cost": sol.get("avg_buy_cost"' in x for x in lines):
    i = find_line(lines, lambda x: '"objective": sol["objective"],' in x)
    if i >= 0:
        indent = lines[i].split('"objective"', 1)[0]
        block = [
            f'{indent}"avg_buy_cost": sol.get("avg_buy_cost", np.nan),',
            f'{indent}"avg_sell_cost": sol.get("avg_sell_cost", np.nan),',
            f'{indent}"avg_soft_trade_signal": sol.get("avg_soft_trade_signal", np.nan),',
            f'{indent}"std_soft_trade_signal": sol.get("std_soft_trade_signal", np.nan),',
        ]
        lines = insert_after_line(lines, i, block)
    else:
        print("[warn] cannot find summary objective line; skipped summary soft-cost diagnostics")

out_text = "\n".join(lines) + "\n"

if DST.exists():
    bak = DST.with_suffix(DST.suffix + ".bak")
    shutil.copy2(DST, bak)
    print("[backup]", bak)

DST.write_text(out_text)
print("[patched]", DST)
print("[ok] generated v16 softcost optimizer")
