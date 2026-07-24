import argparse
import os
import yaml
from pathlib import Path


def deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="config/experiments/factor_registry.yaml")
    parser.add_argument("--out-dir", default="config/experiments/generated")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.registry, "r") as f:
        reg = yaml.safe_load(f)

    defaults = reg.get("defaults", {})
    experiments = reg.get("experiments", [])

    written = []

    for item in experiments:
        tag = item["tag"]

        cfg = {
            "experiment": {
                "tag": tag,
                "description": item.get("description", f"{tag} backtest experiment"),
            },
            "input": {
                "quote_decision_path": item["quote_decision_path"],
                "base_config_path": defaults.get("base_config_path", "config/backtest.yaml"),
            },
            "signal": {
                "col": item["signal_col"],
                "n_bins": item.get("n_bins", 5),
            },
            "fill_model": defaults.get("fill_model", {}),
            "policy": defaults.get("policy", {}),
            "inventory": defaults.get("inventory", {}),
            "fee": defaults.get("fee", {}),
            "output": {
                "fill_prefix": item.get("fill_prefix", "queue_mult_0p05"),
            },
        }

        # allow per-experiment overrides
        for key in ["fill_model", "policy", "inventory", "fee", "output"]:
            if key in item:
                cfg[key] = deep_merge(cfg.get(key, {}), item[key])

        out_path = Path(args.out_dir) / f"{tag}.yaml"

        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        written.append(str(out_path))

    print("written experiment configs:")
    for p in written:
        print(" ", p)


if __name__ == "__main__":
    main()
