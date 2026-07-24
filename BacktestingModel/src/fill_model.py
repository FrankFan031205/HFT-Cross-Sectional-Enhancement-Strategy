import numpy as np
import pandas as pd


def _as_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def normalize_trade_side(trades, cfg):
    buy_values = {str(x).upper() for x in cfg["trade_side"]["aggressive_buy_values"]}
    sell_values = {str(x).upper() for x in cfg["trade_side"]["aggressive_sell_values"]}

    side_raw = trades["side"].astype(str).str.upper()
    side_norm = np.where(
        side_raw.isin(buy_values),
        "BUY",
        np.where(side_raw.isin(sell_values), "SELL", "UNKNOWN"),
    )

    out = trades.copy()
    out["side_norm"] = side_norm
    return out


def estimate_queue_ahead(row, side, quote_price, cfg):
    max_level = int(cfg["backtest"]["max_book_level"])
    eps = float(cfg["backtest"]["price_eps"])

    if side == "BUY":
        best = row.get("bid1", np.nan)
        if pd.notna(best) and quote_price > float(best) + eps:
            return 0.0, 0, "improve_price"

        for k in range(1, max_level + 1):
            pcol = f"bid{k}"
            vcol = f"bid{k}_volume"
            if pcol not in row or vcol not in row:
                continue
            p = row.get(pcol)
            v = row.get(vcol)
            if pd.isna(p) or pd.isna(v):
                continue
            if abs(float(p) - quote_price) <= eps:
                return float(v), k, "book_level"

        return np.inf, None, "out_of_book"

    else:
        best = row.get("ask1", np.nan)
        if pd.notna(best) and quote_price < float(best) - eps:
            return 0.0, 0, "improve_price"

        for k in range(1, max_level + 1):
            pcol = f"ask{k}"
            vcol = f"ask{k}_volume"
            if pcol not in row or vcol not in row:
                continue
            p = row.get(pcol)
            v = row.get(vcol)
            if pd.isna(p) or pd.isna(v):
                continue
            if abs(float(p) - quote_price) <= eps:
                return float(v), k, "book_level"

        return np.inf, None, "out_of_book"


class QueueAwareTradeFillModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.latency = pd.Timedelta(milliseconds=int(cfg["backtest"]["latency_ms"]))
        self.ttl = pd.Timedelta(milliseconds=int(cfg["backtest"]["quote_ttl_ms"]))
        self.order_qty = float(cfg["backtest"]["order_qty"])
        self.eps = float(cfg["backtest"]["price_eps"])
        self.mode = cfg["fill_model"].get("mode", "queue_aware_trade")
        self.full_fill_if_price_crossed = bool(cfg["fill_model"].get("full_fill_if_price_crossed", True))
        self.allow_partial_fill = bool(cfg["fill_model"].get("allow_partial_fill", True))
        self.output_all_orders = bool(cfg["fill_model"].get("output_all_orders", False))

    def _make_trade_groups(self, trades):
        trades = normalize_trade_side(trades, self.cfg)
        groups = {}

        for sid, g in trades.groupby("securityid", sort=False):
            g = g.sort_values("datetime").reset_index(drop=True)
            groups[sid] = {
                "time": g["datetime"].to_numpy(dtype="datetime64[ns]"),
                "price": g["price"].to_numpy(dtype=float),
                "qty": g["qty"].to_numpy(dtype=float),
                "side": g["side_norm"].to_numpy(dtype=object),
            }

        return groups

    def _base_result(self, row, order_side, quote_price, queue_ahead, book_level, queue_source):
        return {
            "decision_time": row["datetime"],
            "securityid": row["securityid"],
            "side": order_side,
            "quote_price": quote_price,
            "fill_price": quote_price,
            "fill_qty": 0.0,
            "fill_time": pd.NaT,
            "queue_ahead_initial": queue_ahead,
            "queue_ahead_remaining": queue_ahead,
            "book_level": book_level,
            "queue_source": queue_source,
            "latency_ms": int(self.cfg["backtest"]["latency_ms"]),
            "quote_ttl_ms": int(self.cfg["backtest"]["quote_ttl_ms"]),
            "fill_model": self.mode,
            "filled": False,
            "touch_type": "",
        }

    def _simulate_one_order(self, row, order_side, quote_price, trade_group):
        decision_time = row["datetime"]
        start_time = decision_time + self.latency
        end_time = start_time + self.ttl

        queue_ahead, book_level, queue_source = estimate_queue_ahead(
            row=row,
            side=order_side,
            quote_price=quote_price,
            cfg=self.cfg,
        )

        queue_mult = float(self.cfg.get("fill_model", {}).get("queue_ahead_multiplier", 1.0))
        if np.isfinite(queue_ahead):
            queue_ahead = queue_ahead * queue_mult

        result = self._base_result(row, order_side, quote_price, queue_ahead, book_level, queue_source)

        if trade_group is None:
            return result

        if np.isinf(queue_ahead):
            return result

        times = trade_group["time"]
        prices = trade_group["price"]
        qtys = trade_group["qty"]
        sides = trade_group["side"]

        left = np.searchsorted(times, np.datetime64(start_time), side="left")
        right = np.searchsorted(times, np.datetime64(end_time), side="left")

        fill_qty = 0.0
        q_remain = float(queue_ahead)
        first_fill_time = pd.NaT

        for i in range(left, right):
            trade_price = prices[i]
            trade_qty = qtys[i]
            trade_side = sides[i]
            trade_time = pd.Timestamp(times[i])

            if order_side == "BUY":
                if trade_side != "SELL":
                    continue

                touched = trade_price <= quote_price + self.eps
                same_level = abs(trade_price - quote_price) <= self.eps
                crossed = trade_price < quote_price - self.eps

                if not touched:
                    continue

                if self.mode == "touched_trade":
                    result["filled"] = True
                    result["fill_qty"] = self.order_qty
                    result["fill_time"] = trade_time
                    result["queue_ahead_remaining"] = q_remain
                    result["touch_type"] = "crossed" if crossed else "same_level"
                    return result

                if self.full_fill_if_price_crossed and crossed:
                    fill_qty = self.order_qty
                    q_remain = 0.0
                    first_fill_time = trade_time
                    result["touch_type"] = "crossed"
                    break

                if same_level:
                    remaining_trade_qty = trade_qty

                    if q_remain > 0:
                        consumed = min(q_remain, remaining_trade_qty)
                        q_remain -= consumed
                        remaining_trade_qty -= consumed

                    if q_remain <= self.eps and remaining_trade_qty > 0:
                        take = min(self.order_qty - fill_qty, remaining_trade_qty)
                        if take > 0 and pd.isna(first_fill_time):
                            first_fill_time = trade_time
                        fill_qty += take
                        result["touch_type"] = "same_level"

            else:
                if trade_side != "BUY":
                    continue

                touched = trade_price >= quote_price - self.eps
                same_level = abs(trade_price - quote_price) <= self.eps
                crossed = trade_price > quote_price + self.eps

                if not touched:
                    continue

                if self.mode == "touched_trade":
                    result["filled"] = True
                    result["fill_qty"] = self.order_qty
                    result["fill_time"] = trade_time
                    result["queue_ahead_remaining"] = q_remain
                    result["touch_type"] = "crossed" if crossed else "same_level"
                    return result

                if self.full_fill_if_price_crossed and crossed:
                    fill_qty = self.order_qty
                    q_remain = 0.0
                    first_fill_time = trade_time
                    result["touch_type"] = "crossed"
                    break

                if same_level:
                    remaining_trade_qty = trade_qty

                    if q_remain > 0:
                        consumed = min(q_remain, remaining_trade_qty)
                        q_remain -= consumed
                        remaining_trade_qty -= consumed

                    if q_remain <= self.eps and remaining_trade_qty > 0:
                        take = min(self.order_qty - fill_qty, remaining_trade_qty)
                        if take > 0 and pd.isna(first_fill_time):
                            first_fill_time = trade_time
                        fill_qty += take
                        result["touch_type"] = "same_level"

            if fill_qty >= self.order_qty:
                fill_qty = self.order_qty
                break

        if fill_qty > 0:
            result["filled"] = True
            result["fill_qty"] = fill_qty
            result["fill_time"] = first_fill_time
            result["queue_ahead_remaining"] = max(q_remain, 0.0)

        return result

    def simulate(self, quotes, trades, max_quotes=None):
        q = quotes.copy()
        q["datetime"] = pd.to_datetime(q["datetime"])
        q["securityid"] = q["securityid"].astype(str).str.zfill(6)
        q = q.sort_values(["securityid", "datetime"]).reset_index(drop=True)

        if max_quotes is not None:
            q = q.head(int(max_quotes))

        trade_groups = self._make_trade_groups(trades)

        records = []
        total_orders = 0

        try:
            from tqdm import tqdm
            iterator = tqdm(q.to_dict("records"), total=len(q))
        except Exception:
            iterator = q.to_dict("records")

        for row in iterator:
            sid = row["securityid"]
            tg = trade_groups.get(sid)

            if _as_bool(row.get("quote_bid", False)):
                total_orders += 1
                price = row.get("bid_quote_price", np.nan)
                if pd.notna(price):
                    res = self._simulate_one_order(row, "BUY", float(price), tg)
                    if self.output_all_orders or res["filled"]:
                        records.append(res)

            if _as_bool(row.get("quote_ask", False)):
                total_orders += 1
                price = row.get("ask_quote_price", np.nan)
                if pd.notna(price):
                    res = self._simulate_one_order(row, "SELL", float(price), tg)
                    if self.output_all_orders or res["filled"]:
                        records.append(res)

        fills = pd.DataFrame(records)
        if len(fills) == 0:
            fills = pd.DataFrame(
                columns=[
                    "decision_time", "securityid", "side", "quote_price",
                    "fill_price", "fill_qty", "fill_time", "queue_ahead_initial",
                    "queue_ahead_remaining", "book_level", "queue_source",
                    "latency_ms", "quote_ttl_ms", "fill_model", "filled", "touch_type"
                ]
            )

        filled_count = int(fills["filled"].sum()) if len(fills) else 0
        print(f"total_quote_orders={total_orders}")
        print(f"filled_orders={filled_count}")
        if total_orders > 0:
            print(f"fill_rate={filled_count / total_orders:.6f}")

        return fills
