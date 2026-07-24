# -*- coding: utf-8 -*-
from pathlib import Path
import shutil

p = Path("TakerOptimizer/scripts/run_tplus1_aware_pure_cs_optimizer_v16_softcost.py")
if not p.exists():
    raise FileNotFoundError(p)

text = p.read_text().replace("\r\n", "\n")
bak = p.with_suffix(p.suffix + ".bak_v16b_safe_softcost")
shutil.copy2(p, bak)
print("[backup]", bak)

# Add args.
if "--soft-cost-max-frac" not in text:
    text = text.replace(
        'ap.add_argument("--soft-cost-scale", type=float, default=0.0005)',
        'ap.add_argument("--soft-cost-scale", type=float, default=0.0005)\n'
        '    ap.add_argument("--soft-cost-max-frac", type=float, default=0.25)\n'
        '    ap.add_argument("--min-trade-cost", type=float, default=0.0005)'
    )

old = '''buy_cost = float(args.lambda_turnover) + buy_exec_cost - soft_trade_signal
    sell_cost = float(args.lambda_turnover) + sell_exec_cost + soft_trade_signal'''

new = '''base_buy_cost = float(args.lambda_turnover) + buy_exec_cost
    base_sell_cost = float(args.lambda_turnover) + sell_exec_cost

    # Safe soft-cost:
    # h10 may tilt trading preference, but cannot erase execution cost.
    # This prevents h10 from becoming a high-turnover short-horizon alpha.
    soft_cap = float(args.soft_cost_max_frac) * np.minimum(base_buy_cost, base_sell_cost)
    soft_adj = np.clip(soft_trade_signal, -soft_cap, soft_cap)

    buy_cost = np.maximum(float(args.min_trade_cost), base_buy_cost - soft_adj)
    sell_cost = np.maximum(float(args.min_trade_cost), base_sell_cost + soft_adj)'''

if old not in text:
    raise RuntimeError("cannot find old buy_cost/sell_cost block; already patched or format changed")

text = text.replace(old, new)

p.write_text(text)
print("[patched]", p)
