from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional, Tuple
from pathlib import Path
from datetime import datetime
import math
import json


def _get(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _is_nan(x: Any) -> bool:
    try:
        return x != x
    except Exception:
        return False


def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None or _is_nan(x):
        return default
    try:
        return float(x)
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def floor_to_tick(price: float, tick: float) -> float:
    return round(math.floor(price / tick + 1e-12) * tick, 10)


def ceil_to_tick(price: float, tick: float) -> float:
    return round(math.ceil(price / tick - 1e-12) * tick, 10)


def round_size_to_lot(size: float, lot_size: int) -> int:
    if size <= 0:
        return 0
    return int(math.floor(size / lot_size) * lot_size)


def _time_str(x: Any) -> Optional[str]:
    if x is None or _is_nan(x):
        return None

    if hasattr(x, "strftime"):
        try:
            return x.strftime("%H:%M:%S")
        except Exception:
            pass

    s = str(x).strip()

    if "_" in s:
        tail = s.split("_")[-1]
        digits = "".join(ch for ch in tail if ch.isdigit())
        if len(digits) >= 6:
            return f"{digits[0:2]}:{digits[2:4]}:{digits[4:6]}"

    if " " in s:
        s = s.split(" ")[-1]

    if "T" in s:
        s = s.split("T")[-1]

    if "." in s:
        s = s.split(".")[0]

    if len(s) >= 8 and s[2] == ":" and s[5] == ":":
        return s[:8]

    try:
        dt = datetime.fromisoformat(str(x))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return None


@dataclass
class QuoteDecision:
    datetime: Any = None
    securityid: Any = None

    bid1: Optional[float] = None
    ask1: Optional[float] = None
    mid_price: Optional[float] = None
    spread: Optional[float] = None
    spread_ticks: Optional[float] = None

    raw_pred: Optional[float] = None
    pred_used: Optional[float] = None
    calibrated_pred: Optional[float] = None
    calibration_method: str = "none"
    calibration_a: Optional[float] = None
    calibration_b: Optional[float] = None

    raw_alpha_ticks: Optional[float] = None
    model_alpha_ticks: Optional[float] = None
    unclipped_alpha_ticks: Optional[float] = None
    fair_shift_ticks: Optional[float] = None
    alpha_ticks: Optional[float] = None
    alpha_bucket: str = "unknown"
    alpha_clipped: bool = False

    microprice: Optional[float] = None
    microprice_shift_ticks: float = 0.0

    bid_depth_1: float = 0.0
    ask_depth_1: float = 0.0
    total_depth_1: float = 0.0
    book_imbalance_1: float = 0.0

    bid_depth_5: float = 0.0
    ask_depth_5: float = 0.0
    total_depth_5: float = 0.0
    book_imbalance_5: float = 0.0

    book_pressure_shift_ticks: float = 0.0

    trade_imbalance: float = 0.0
    trade_pressure_shift_ticks: float = 0.0

    cancel_pressure_bid: float = 0.0
    cancel_pressure_ask: float = 0.0
    cancel_pressure_imbalance: float = 0.0
    cancel_pressure_shift_ticks: float = 0.0

    volatility_ticks: float = 0.0
    volatility_regime: str = "unknown"
    liquidity_state: str = "unknown"
    microstructure_fair_price: Optional[float] = None

    fair_price: Optional[float] = None
    quote_fair_price: Optional[float] = None

    position: int = 0
    position_ratio: float = 0.0
    inventory_skew: float = 0.0
    inventory_penalty: float = 0.0
    inventory_exit_bonus: float = 0.0
    inventory_state: str = "normal_inventory"

    adverse_buffer: float = 0.0
    bid_adverse_buffer: float = 0.0
    ask_adverse_buffer: float = 0.0
    adverse_buffer_ticks: Optional[float] = None
    bid_adverse_buffer_ticks: Optional[float] = None
    ask_adverse_buffer_ticks: Optional[float] = None

    bid_risk_score: float = 0.0
    ask_risk_score: float = 0.0
    alpha_bid_risk_ticks: float = 0.0
    alpha_ask_risk_ticks: float = 0.0

    bid_fee: float = 0.0
    ask_fee: float = 0.0

    edge_threshold: float = 0.0
    edge_threshold_ticks: float = 0.0

    bid_threshold: float = 0.0
    ask_threshold: float = 0.0
    bid_threshold_ticks: float = 0.0
    ask_threshold_ticks: float = 0.0

    quote_bid: bool = False
    bid_price: Optional[float] = None
    bid_size: int = 0
    bid_size_after_inventory: int = 0
    bid_edge: Optional[float] = None
    bid_edge_ticks: Optional[float] = None
    bid_reason: str = "not_evaluated"

    quote_ask: bool = False
    ask_price: Optional[float] = None
    ask_size: int = 0
    ask_size_after_inventory: int = 0
    ask_edge: Optional[float] = None
    ask_edge_ticks: Optional[float] = None
    ask_reason: str = "not_evaluated"

    risk_state: str = "normal"
    quote_style: str = "none"

    symbol_bucket: str = "global"
    spread_regime: str = "unknown"
    time_regime: str = "unknown"
    applied_param_set: str = "global"
    symbol_specific_threshold: float = 0.0
    symbol_specific_max_position: int = 0
    symbol_specific_quote_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FairValueMarketMaker:
    def __init__(self, config: Mapping[str, Any]):
        self.config = config

        self.datetime_col = _get(config, "columns.datetime_col", "datetime")
        self.symbol_col = _get(config, "columns.symbol_col", "securityid")
        self.bid_col = _get(config, "columns.bid_col", "bid1")
        self.ask_col = _get(config, "columns.ask_col", "ask1")
        self.mid_col = _get(config, "columns.mid_col", "mid_price")
        self.spread_col = _get(config, "columns.spread_col", "spread")
        self.pred_col = _get(config, "columns.pred_col", "pred_ret")
        self.bid_volume_col = _get(config, "columns.bid_volume_col", "bid1_volume")
        self.ask_volume_col = _get(config, "columns.ask_volume_col", "ask1_volume")
        self.limit_up_col = _get(config, "columns.limit_up_col", "limit_up_price")
        self.limit_down_col = _get(config, "columns.limit_down_col", "limit_down_price")

        self.tick_size = float(_get(config, "market.tick_size", 0.01))
        self.lot_size = int(_get(config, "market.lot_size", 100))
        self.min_price = float(_get(config, "market.min_price", 0.01))

        self.calibration_enabled = bool(_get(config, "calibration.enabled", False))
        self.calibration_fallback_to_raw = bool(_get(config, "calibration.fallback_to_raw", True))
        self.calibration_path = (
            _get(config, "calibration.calibration_path", None)
            or _get(config, "calibration.path", None)
        )
        self.calibration = self._load_calibration()

        self.alpha_scale = float(_get(config, "prediction.alpha_scale", 1.0))
        self.use_calibrated_pred = bool(_get(config, "prediction.use_calibrated_pred", True))
        self.calibrated_alpha_scale = float(_get(config, "prediction.calibrated_alpha_scale", 1.0))

        self.clip_pred = bool(_get(config, "prediction.clip_pred", True))
        self.max_abs_pred = float(_get(config, "prediction.max_abs_pred", 0.01))

        self.clip_alpha_ticks = bool(_get(config, "prediction.clip_alpha_ticks", True))
        self.max_fair_shift_ticks = float(_get(config, "prediction.max_fair_shift_ticks", 5.0))
        self.weak_alpha_ticks = float(_get(config, "prediction.weak_alpha_ticks", 0.3))
        self.strong_alpha_ticks = float(_get(config, "prediction.strong_alpha_ticks", 1.0))

        self.microstructure_enabled = bool(_get(config, "microstructure.enabled", False))
        self.use_microprice = bool(_get(config, "microstructure.use_microprice", True))
        self.use_book_pressure = bool(_get(config, "microstructure.use_book_pressure", True))
        self.use_trade_pressure = bool(_get(config, "microstructure.use_trade_pressure", True))
        self.use_cancel_pressure = bool(_get(config, "microstructure.use_cancel_pressure", True))
        self.use_volatility_regime = bool(_get(config, "microstructure.use_volatility_regime", True))
        self.use_liquidity_regime = bool(_get(config, "microstructure.use_liquidity_regime", True))

        self.microprice_col = _get(config, "microstructure.microprice_col", "microprice")
        self.microprice_shift_ticks_col = _get(config, "microstructure.microprice_shift_ticks_col", "microprice_shift_ticks")

        self.book_imbalance_col = _get(config, "microstructure.book_imbalance_col", "book_imbalance_5")
        self.fallback_book_imbalance_col = _get(config, "microstructure.fallback_book_imbalance_col", "book_imbalance_1")
        self.book_pressure_scale = float(_get(config, "microstructure.book_pressure_scale", 0.5))

        self.trade_imbalance_col = _get(config, "microstructure.trade_imbalance_col", "trade_imbalance")
        self.trade_pressure_scale = float(_get(config, "microstructure.trade_pressure_scale", 0.3))

        self.cancel_pressure_bid_col = _get(config, "microstructure.cancel_pressure_bid_col", "cancel_pressure_bid")
        self.cancel_pressure_ask_col = _get(config, "microstructure.cancel_pressure_ask_col", "cancel_pressure_ask")
        self.cancel_pressure_imbalance_col = _get(config, "microstructure.cancel_pressure_imbalance_col", "cancel_pressure_imbalance")
        self.cancel_pressure_scale = float(_get(config, "microstructure.cancel_pressure_scale", 0.2))

        self.volatility_ticks_col = _get(config, "microstructure.volatility_ticks_col", "volatility_ticks")
        self.high_volatility_ticks = float(_get(config, "microstructure.high_volatility_ticks", 2.0))
        self.low_volatility_ticks = float(_get(config, "microstructure.low_volatility_ticks", 0.5))

        self.liquidity_state_col = _get(config, "microstructure.liquidity_state_col", "liquidity_state")
        self.total_depth_col = _get(config, "microstructure.total_depth_col", "total_depth_5")
        self.low_liquidity_depth = float(_get(config, "microstructure.low_liquidity_depth", 1000))
        self.high_liquidity_depth = float(_get(config, "microstructure.high_liquidity_depth", 10000))

        self.max_microstructure_shift_ticks = float(_get(config, "microstructure.max_microstructure_shift_ticks", 3.0))

        self.quote_mode = _get(config, "strategy.quote_mode", "adaptive")
        self.allow_cross = bool(_get(config, "strategy.allow_cross", False))
        self.default_quote_style = _get(config, "strategy.default_quote_style", "join_best")
        self.improve_when_strong_alpha = bool(_get(config, "strategy.improve_when_strong_alpha", True))
        self.max_improve_ticks = int(_get(config, "strategy.max_improve_ticks", 1))
        self.allow_bid = bool(_get(config, "strategy.allow_bid", True))
        self.allow_ask = bool(_get(config, "strategy.allow_ask", True))
        self.allow_one_sided_quote = bool(_get(config, "strategy.allow_one_sided_quote", True))

        self.fee_mode = _get(config, "cost.fee_mode", "rate")
        self.fee_rate = float(_get(config, "cost.fee_rate", 0.0001))
        self.fee_per_share = float(_get(config, "cost.fee_per_share", 0.0))

        self.edge_threshold_ticks = float(_get(config, "edge.edge_threshold_ticks", 0.2))
        self.base_adverse_buffer_ticks = float(_get(config, "edge.base_adverse_buffer_ticks", 0.5))
        self.dynamic_adverse_buffer = bool(_get(config, "edge.dynamic_adverse_buffer", True))
        self.volatility_buffer_multiplier = float(_get(config, "edge.volatility_buffer_multiplier", 1.0))
        self.spread_buffer_multiplier = float(_get(config, "edge.spread_buffer_multiplier", 0.2))
        self.inventory_buffer_multiplier = float(_get(config, "edge.inventory_buffer_multiplier", 0.5))

        self.side_specific_adverse_buffer = bool(_get(config, "edge.side_specific_adverse_buffer", True))
        self.alpha_toxic_buffer_multiplier = float(_get(config, "edge.alpha_toxic_buffer_multiplier", 1.0))
        self.trade_imbalance_buffer_multiplier = float(_get(config, "edge.trade_imbalance_buffer_multiplier", 0.5))
        self.cancel_pressure_buffer_multiplier = float(_get(config, "edge.cancel_pressure_buffer_multiplier", 0.5))
        self.missing_risk_feature_default = float(_get(config, "edge.missing_risk_feature_default", 0.0))

        self.default_position = int(_get(config, "inventory.default_position", 0))
        self.max_position = int(_get(config, "inventory.max_position", 5000))
        self.warning_position_ratio = float(_get(config, "inventory.warning_position_ratio", 0.7))
        self.danger_position_ratio = float(_get(config, "inventory.danger_position_ratio", 0.9))
        self.inventory_skew_ticks = float(_get(config, "inventory.inventory_skew_ticks", 2.0))
        self.nonlinear_inventory_skew = bool(_get(config, "inventory.nonlinear_inventory_skew", True))
        self.inventory_skew_power = float(_get(config, "inventory.inventory_skew_power", 2.0))
        self.inventory_threshold_adjustment_ticks = float(_get(config, "inventory.inventory_threshold_adjustment_ticks", 0.0))

        self.base_quote_size = int(_get(config, "inventory.base_quote_size", 100))
        self.min_quote_size = int(_get(config, "inventory.min_quote_size", 100))
        self.max_quote_size = int(_get(config, "inventory.max_quote_size", 500))
        self.inventory_size_decay = float(_get(config, "inventory.inventory_size_decay", 0.5))

        self.quote_size_mode = _get(config, "quote_size.mode", "edge_and_inventory")
        self.edge_size_scaling = bool(_get(config, "quote_size.edge_size_scaling", True))
        self.target_edge_ticks = float(_get(config, "quote_size.target_edge_ticks", 2.0))
        self.max_size_multiplier = float(_get(config, "quote_size.max_size_multiplier", 3.0))
        self.inventory_aware_size = bool(_get(config, "quote_size.inventory_aware_size", True))
        self.increase_exit_side_size = bool(_get(config, "quote_size.increase_exit_side_size", True))
        self.decrease_entry_side_size = bool(_get(config, "quote_size.decrease_entry_side_size", True))

        self.enable_basic_book_filter = bool(_get(config, "filters.enable_basic_book_filter", True))
        self.enable_prediction_filter = bool(_get(config, "filters.enable_prediction_filter", True))
        self.enable_spread_filter = bool(_get(config, "filters.enable_spread_filter", True))
        self.enable_liquidity_filter = bool(_get(config, "filters.enable_liquidity_filter", True))
        self.enable_limit_price_filter = bool(_get(config, "filters.enable_limit_price_filter", True))
        self.enable_time_filter = bool(_get(config, "filters.enable_time_filter", False))
        self.enable_volatility_filter = bool(_get(config, "filters.enable_volatility_filter", False))

        self.min_spread_ticks = float(_get(config, "filters.min_spread_ticks", 1.0))
        self.max_spread_ticks = float(_get(config, "filters.max_spread_ticks", 10.0))
        self.min_bid1_volume = float(_get(config, "filters.min_bid1_volume", 100))
        self.min_ask1_volume = float(_get(config, "filters.min_ask1_volume", 100))
        self.limit_price_buffer_ticks = float(_get(config, "filters.limit_price_buffer_ticks", 2.0))

        self.trading_start_time = _get(config, "filters.trading_start_time", None)
        self.trading_end_time = _get(config, "filters.trading_end_time", None)

        self.use_spread_as_vol_proxy = bool(_get(config, "volatility.use_spread_as_vol_proxy", True))
        self.high_vol_spread_ticks = float(_get(config, "volatility.high_vol_spread_ticks", 5.0))
        self.max_volatility = _get(config, "volatility.max_volatility", None)
        if self.max_volatility is not None:
            self.max_volatility = float(self.max_volatility)

    def _load_calibration(self) -> Optional[Dict[str, Any]]:
        if not self.calibration_enabled:
            return None

        if not self.calibration_path:
            if self.calibration_fallback_to_raw:
                return None
            raise ValueError("calibration.enabled=true but calibration_path is empty")

        path = Path(self.calibration_path)
        if not path.exists():
            if self.calibration_fallback_to_raw:
                print(f"warning: calibration file not found, fallback to raw prediction: {path}")
                return None
            raise FileNotFoundError(f"calibration file not found: {path}")

        with open(path, "r") as f:
            return json.load(f)

    def generate_quote(self, row: Mapping[str, Any], position: Optional[int] = None) -> QuoteDecision:
        pos = self.default_position if position is None else int(position)

        dt = row.get(self.datetime_col)
        sid = row.get(self.symbol_col)

        bid1 = _to_float(row.get(self.bid_col))
        ask1 = _to_float(row.get(self.ask_col))
        mid = _to_float(row.get(self.mid_col))
        pred_raw = _to_float(row.get(self.pred_col))

        if mid is None and bid1 is not None and ask1 is not None:
            mid = (bid1 + ask1) / 2.0

        spread = _to_float(row.get(self.spread_col))
        if spread is None and bid1 is not None and ask1 is not None:
            spread = ask1 - bid1

        decision = QuoteDecision(
            datetime=dt,
            securityid=sid,
            bid1=bid1,
            ask1=ask1,
            mid_price=mid,
            spread=spread,
            raw_pred=pred_raw,
            position=pos,
        )

        if spread is not None and self.tick_size > 0:
            decision.spread_ticks = spread / self.tick_size

        no_quote_reason = self._pretrade_filter(row, bid1, ask1, mid, spread, pred_raw)
        if no_quote_reason is not None:
            decision.risk_state = no_quote_reason
            decision.bid_reason = no_quote_reason
            decision.ask_reason = no_quote_reason
            return decision

        position_ratio = 0.0
        if self.max_position > 0:
            position_ratio = _clip(pos / self.max_position, -1.0, 1.0)

        inventory_state = self._inventory_state(pos, position_ratio)

        alpha_info = self._compute_alpha_info(mid, pred_raw)
        model_alpha_ticks = alpha_info["model_alpha_ticks"]

        micro_info = self._compute_microstructure_info(
            row=row,
            bid1=bid1,
            ask1=ask1,
            mid=mid,
            model_alpha_ticks=model_alpha_ticks,
        )

        if self.microstructure_enabled:
            fair_price = micro_info["microstructure_fair_price"]
        else:
            fair_price = alpha_info["fair_price"]

        final_alpha_ticks = (fair_price - mid) / self.tick_size
        final_alpha_ticks = _clip(final_alpha_ticks, -self.max_fair_shift_ticks, self.max_fair_shift_ticks)
        fair_price = mid + final_alpha_ticks * self.tick_size

        alpha_bucket = self._alpha_bucket(final_alpha_ticks)

        inventory_skew = self._compute_inventory_skew(position_ratio)
        quote_fair_price = fair_price - inventory_skew

        (
            adverse_buffer,
            bid_adverse_buffer,
            ask_adverse_buffer,
            bid_risk_score,
            ask_risk_score,
            alpha_bid_risk_ticks,
            alpha_ask_risk_ticks,
            trade_imbalance,
            cancel_pressure_bid,
            cancel_pressure_ask,
        ) = self._compute_side_adverse_buffers(
            row=row,
            spread=spread,
            position_ratio=position_ratio,
            alpha_ticks=final_alpha_ticks,
            micro_info=micro_info,
        )

        bid_price, ask_price, quote_style = self._generate_candidate_quotes(
            bid1=bid1,
            ask1=ask1,
            alpha_ticks=final_alpha_ticks,
        )

        bid_fee = self._fee_per_share_at_price(bid_price)
        ask_fee = self._fee_per_share_at_price(ask_price)

        edge_threshold = self.edge_threshold_ticks * self.tick_size
        bid_threshold, ask_threshold = self._side_thresholds(edge_threshold, position_ratio)

        spread_ticks = spread / self.tick_size
        adverse_buffer_ticks = adverse_buffer / self.tick_size
        bid_adverse_buffer_ticks = bid_adverse_buffer / self.tick_size
        ask_adverse_buffer_ticks = ask_adverse_buffer / self.tick_size

        bid_edge = quote_fair_price - bid_price - bid_fee - bid_adverse_buffer
        ask_edge = ask_price - quote_fair_price - ask_fee - ask_adverse_buffer

        bid_edge_ticks = bid_edge / self.tick_size
        ask_edge_ticks = ask_edge / self.tick_size

        quote_bid = bid_edge >= bid_threshold
        quote_ask = ask_edge >= ask_threshold

        bid_reason = "positive_edge" if quote_bid else "insufficient_edge"
        ask_reason = "positive_edge" if quote_ask else "insufficient_edge"

        if not self.allow_bid:
            quote_bid = False
            bid_reason = "bid_disabled"

        if not self.allow_ask:
            quote_ask = False
            ask_reason = "ask_disabled"

        if pos >= self.max_position:
            quote_bid = False
            bid_reason = "max_long_reached"

        if pos <= -self.max_position:
            quote_ask = False
            ask_reason = "max_short_reached"

        if not self.allow_one_sided_quote and quote_bid != quote_ask:
            quote_bid = False
            quote_ask = False
            bid_reason = "one_sided_quote_disabled"
            ask_reason = "one_sided_quote_disabled"

        bid_size = self._compute_quote_size("bid", bid_edge, position_ratio) if quote_bid else 0
        ask_size = self._compute_quote_size("ask", ask_edge, position_ratio) if quote_ask else 0

        risk_state = self._risk_state(
            position_ratio=position_ratio,
            alpha_ticks=final_alpha_ticks,
            alpha_clipped=alpha_info["alpha_clipped"],
            volatility_regime=micro_info["volatility_regime"],
            liquidity_state=micro_info["liquidity_state"],
        )

        decision.raw_pred = pred_raw
        decision.pred_used = alpha_info["pred_used"]
        decision.calibrated_pred = alpha_info["calibrated_pred"]
        decision.calibration_method = alpha_info["calibration_method"]
        decision.calibration_a = alpha_info["calibration_a"]
        decision.calibration_b = alpha_info["calibration_b"]

        decision.raw_alpha_ticks = alpha_info["raw_alpha_ticks"]
        decision.model_alpha_ticks = model_alpha_ticks
        decision.unclipped_alpha_ticks = alpha_info["unclipped_alpha_ticks"]
        decision.fair_shift_ticks = final_alpha_ticks
        decision.alpha_ticks = final_alpha_ticks
        decision.alpha_bucket = alpha_bucket
        decision.alpha_clipped = alpha_info["alpha_clipped"]

        decision.microprice = micro_info["microprice"]
        decision.microprice_shift_ticks = micro_info["microprice_shift_ticks"]

        decision.bid_depth_1 = micro_info["bid_depth_1"]
        decision.ask_depth_1 = micro_info["ask_depth_1"]
        decision.total_depth_1 = micro_info["total_depth_1"]
        decision.book_imbalance_1 = micro_info["book_imbalance_1"]

        decision.bid_depth_5 = micro_info["bid_depth_5"]
        decision.ask_depth_5 = micro_info["ask_depth_5"]
        decision.total_depth_5 = micro_info["total_depth_5"]
        decision.book_imbalance_5 = micro_info["book_imbalance_5"]

        decision.book_pressure_shift_ticks = micro_info["book_pressure_shift_ticks"]
        decision.trade_imbalance = micro_info["trade_imbalance"]
        decision.trade_pressure_shift_ticks = micro_info["trade_pressure_shift_ticks"]

        decision.cancel_pressure_bid = micro_info["cancel_pressure_bid"]
        decision.cancel_pressure_ask = micro_info["cancel_pressure_ask"]
        decision.cancel_pressure_imbalance = micro_info["cancel_pressure_imbalance"]
        decision.cancel_pressure_shift_ticks = micro_info["cancel_pressure_shift_ticks"]

        decision.volatility_ticks = micro_info["volatility_ticks"]
        decision.volatility_regime = micro_info["volatility_regime"]
        decision.liquidity_state = micro_info["liquidity_state"]
        decision.microstructure_fair_price = micro_info["microstructure_fair_price"]

        decision.fair_price = fair_price
        decision.quote_fair_price = quote_fair_price

        decision.position_ratio = position_ratio
        decision.inventory_skew = inventory_skew
        decision.inventory_penalty = abs(position_ratio) * self.inventory_size_decay
        decision.inventory_exit_bonus = abs(position_ratio) * self.inventory_size_decay
        decision.inventory_state = inventory_state

        decision.adverse_buffer = adverse_buffer
        decision.bid_adverse_buffer = bid_adverse_buffer
        decision.ask_adverse_buffer = ask_adverse_buffer
        decision.adverse_buffer_ticks = adverse_buffer_ticks
        decision.bid_adverse_buffer_ticks = bid_adverse_buffer_ticks
        decision.ask_adverse_buffer_ticks = ask_adverse_buffer_ticks

        decision.bid_risk_score = bid_risk_score
        decision.ask_risk_score = ask_risk_score
        decision.alpha_bid_risk_ticks = alpha_bid_risk_ticks
        decision.alpha_ask_risk_ticks = alpha_ask_risk_ticks

        decision.bid_fee = bid_fee
        decision.ask_fee = ask_fee

        decision.edge_threshold = edge_threshold
        decision.edge_threshold_ticks = self.edge_threshold_ticks
        decision.bid_threshold = bid_threshold
        decision.ask_threshold = ask_threshold
        decision.bid_threshold_ticks = bid_threshold / self.tick_size
        decision.ask_threshold_ticks = ask_threshold / self.tick_size

        decision.spread_ticks = spread_ticks

        decision.quote_bid = quote_bid
        decision.bid_price = bid_price if quote_bid else None
        decision.bid_size = bid_size
        decision.bid_size_after_inventory = bid_size
        decision.bid_edge = bid_edge
        decision.bid_edge_ticks = bid_edge_ticks
        decision.bid_reason = bid_reason

        decision.quote_ask = quote_ask
        decision.ask_price = ask_price if quote_ask else None
        decision.ask_size = ask_size
        decision.ask_size_after_inventory = ask_size
        decision.ask_edge = ask_edge
        decision.ask_edge_ticks = ask_edge_ticks
        decision.ask_reason = ask_reason

        decision.risk_state = risk_state
        decision.quote_style = quote_style
        decision.spread_regime = self._spread_regime(spread_ticks)
        decision.time_regime = self._time_regime(row.get(self.datetime_col))

        return decision

    def _compute_alpha_info(self, mid: float, pred_raw: float) -> Dict[str, Any]:
        raw_scaled_pred = self.alpha_scale * pred_raw
        raw_alpha_ticks = mid * raw_scaled_pred / self.tick_size

        calibration_method = "raw_no_calibration"
        calibration_a = None
        calibration_b = None
        calibrated_pred = pred_raw

        has_calibration = self.calibration_enabled and self.calibration is not None

        if has_calibration and self.use_calibrated_pred:
            calibration_a = _to_float(self.calibration.get("a"), 1.0)
            calibration_b = _to_float(self.calibration.get("b"), 0.0)
            calibration_method = str(self.calibration.get("method", "global_linear"))
            calibrated_pred = calibration_a * pred_raw + calibration_b
            pred_before_clip = self.calibrated_alpha_scale * calibrated_pred
        else:
            pred_before_clip = raw_scaled_pred

        pred_used = pred_before_clip
        if self.clip_pred:
            pred_used = _clip(pred_used, -self.max_abs_pred, self.max_abs_pred)

        unclipped_alpha_ticks = mid * pred_used / self.tick_size

        if self.clip_alpha_ticks:
            model_alpha_ticks = _clip(
                unclipped_alpha_ticks,
                -self.max_fair_shift_ticks,
                self.max_fair_shift_ticks,
            )
        else:
            model_alpha_ticks = unclipped_alpha_ticks

        alpha_clipped = abs(unclipped_alpha_ticks - model_alpha_ticks) > 1e-12
        fair_price = mid + model_alpha_ticks * self.tick_size

        return {
            "pred_used": pred_used,
            "calibrated_pred": calibrated_pred,
            "calibration_method": calibration_method,
            "calibration_a": calibration_a,
            "calibration_b": calibration_b,
            "raw_alpha_ticks": raw_alpha_ticks,
            "unclipped_alpha_ticks": unclipped_alpha_ticks,
            "model_alpha_ticks": model_alpha_ticks,
            "alpha_clipped": alpha_clipped,
            "fair_price": fair_price,
        }

    def _alpha_bucket(self, alpha_ticks: float) -> str:
        x = abs(alpha_ticks)
        if x < self.weak_alpha_ticks:
            return "weak"
        if x < self.strong_alpha_ticks:
            return "normal"
        return "strong"

    def _row_float(self, row: Mapping[str, Any], col: str, default: float = 0.0) -> float:
        if not col:
            return default
        return _to_float(row.get(col), default)

    def _compute_depth(self, row: Mapping[str, Any], side: str, levels: int) -> float:
        total = 0.0
        for i in range(1, levels + 1):
            names = [
                f"{side}{i}_volume",
                f"{side}volume{i}",
                f"{side}_volume_{i}",
                f"{side}volume{i}",
                f"{side}vol{i}",
            ]
            val = None
            for name in names:
                if name in row:
                    val = _to_float(row.get(name), None)
                    break
            if val is not None:
                total += val
        return total

    def _compute_microstructure_info(
        self,
        row: Mapping[str, Any],
        bid1: float,
        ask1: float,
        mid: float,
        model_alpha_ticks: float,
    ) -> Dict[str, Any]:
        bid1_vol = _to_float(row.get(self.bid_volume_col), 0.0)
        ask1_vol = _to_float(row.get(self.ask_volume_col), 0.0)

        microprice = _to_float(row.get(self.microprice_col), None)
        if microprice is None and bid1_vol + ask1_vol > 0:
            microprice = (ask1 * bid1_vol + bid1 * ask1_vol) / (bid1_vol + ask1_vol)
        if microprice is None:
            microprice = mid

        microprice_shift_ticks = _to_float(row.get(self.microprice_shift_ticks_col), None)
        if microprice_shift_ticks is None:
            microprice_shift_ticks = (microprice - mid) / self.tick_size

        bid_depth_1 = bid1_vol
        ask_depth_1 = ask1_vol
        total_depth_1 = bid_depth_1 + ask_depth_1
        book_imbalance_1 = (bid_depth_1 - ask_depth_1) / total_depth_1 if total_depth_1 > 0 else 0.0

        bid_depth_5 = _to_float(row.get("bid_depth_5"), None)
        ask_depth_5 = _to_float(row.get("ask_depth_5"), None)

        if bid_depth_5 is None:
            bid_depth_5 = self._compute_depth(row, "bid", 5)
        if ask_depth_5 is None:
            ask_depth_5 = self._compute_depth(row, "ask", 5)

        total_depth_5 = bid_depth_5 + ask_depth_5
        computed_book_imbalance_5 = (bid_depth_5 - ask_depth_5) / total_depth_5 if total_depth_5 > 0 else book_imbalance_1

        book_imbalance_1_input = _to_float(row.get("book_imbalance_1"), book_imbalance_1)
        book_imbalance_5_input = _to_float(row.get("book_imbalance_5"), computed_book_imbalance_5)

        book_imbalance = _to_float(row.get(self.book_imbalance_col), None)
        if book_imbalance is None:
            book_imbalance = _to_float(row.get(self.fallback_book_imbalance_col), book_imbalance_5_input)

        book_pressure_shift_ticks = self.book_pressure_scale * book_imbalance if self.use_book_pressure else 0.0

        trade_imbalance = _to_float(row.get(self.trade_imbalance_col), 0.0)
        trade_pressure_shift_ticks = self.trade_pressure_scale * trade_imbalance if self.use_trade_pressure else 0.0

        cancel_pressure_bid = _to_float(row.get(self.cancel_pressure_bid_col), 0.0)
        cancel_pressure_ask = _to_float(row.get(self.cancel_pressure_ask_col), 0.0)

        cancel_pressure_imbalance = _to_float(row.get(self.cancel_pressure_imbalance_col), None)
        if cancel_pressure_imbalance is None:
            cancel_pressure_imbalance = cancel_pressure_ask - cancel_pressure_bid

        cancel_pressure_shift_ticks = (
            self.cancel_pressure_scale * cancel_pressure_imbalance
            if self.use_cancel_pressure
            else 0.0
        )

        volatility_ticks = _to_float(row.get(self.volatility_ticks_col), None)
        if volatility_ticks is None:
            volatility_ticks = _to_float(row.get("spread_ticks"), 0.0)

        if volatility_ticks >= self.high_volatility_ticks:
            volatility_regime = "high_volatility"
        elif volatility_ticks <= self.low_volatility_ticks:
            volatility_regime = "low_volatility"
        else:
            volatility_regime = "normal_volatility"

        liquidity_state = row.get(self.liquidity_state_col)
        if liquidity_state is None or _is_nan(liquidity_state):
            total_depth = _to_float(row.get(self.total_depth_col), total_depth_5)
            if total_depth <= self.low_liquidity_depth:
                liquidity_state = "low_liquidity"
            elif total_depth >= self.high_liquidity_depth:
                liquidity_state = "high_liquidity"
            else:
                liquidity_state = "normal_liquidity"
        else:
            liquidity_state = str(liquidity_state)

        micro_shift_ticks = (
            book_pressure_shift_ticks
            + trade_pressure_shift_ticks
            + cancel_pressure_shift_ticks
        )
        micro_shift_ticks = _clip(
            micro_shift_ticks,
            -self.max_microstructure_shift_ticks,
            self.max_microstructure_shift_ticks,
        )

        if self.microstructure_enabled:
            if self.use_microprice:
                microstructure_fair_price = microprice + (model_alpha_ticks + micro_shift_ticks) * self.tick_size
            else:
                microstructure_fair_price = mid + (model_alpha_ticks + micro_shift_ticks) * self.tick_size
        else:
            microstructure_fair_price = mid + model_alpha_ticks * self.tick_size

        return {
            "microprice": microprice,
            "microprice_shift_ticks": microprice_shift_ticks,
            "bid_depth_1": bid_depth_1,
            "ask_depth_1": ask_depth_1,
            "total_depth_1": total_depth_1,
            "book_imbalance_1": book_imbalance_1_input,
            "bid_depth_5": bid_depth_5,
            "ask_depth_5": ask_depth_5,
            "total_depth_5": total_depth_5,
            "book_imbalance_5": book_imbalance_5_input,
            "book_pressure_shift_ticks": book_pressure_shift_ticks,
            "trade_imbalance": trade_imbalance,
            "trade_pressure_shift_ticks": trade_pressure_shift_ticks,
            "cancel_pressure_bid": cancel_pressure_bid,
            "cancel_pressure_ask": cancel_pressure_ask,
            "cancel_pressure_imbalance": cancel_pressure_imbalance,
            "cancel_pressure_shift_ticks": cancel_pressure_shift_ticks,
            "volatility_ticks": volatility_ticks,
            "volatility_regime": volatility_regime,
            "liquidity_state": liquidity_state,
            "microstructure_fair_price": microstructure_fair_price,
        }

    def _pretrade_filter(
        self,
        row: Mapping[str, Any],
        bid1: Optional[float],
        ask1: Optional[float],
        mid: Optional[float],
        spread: Optional[float],
        pred: Optional[float],
    ) -> Optional[str]:
        if self.enable_basic_book_filter:
            if bid1 is None or ask1 is None or mid is None:
                return "missing_book"
            if bid1 < self.min_price or ask1 < self.min_price or mid < self.min_price:
                return "invalid_price"
            if ask1 <= bid1:
                return "crossed_or_locked_book"

        if self.enable_prediction_filter:
            if pred is None:
                return "missing_prediction"

        if self.enable_spread_filter:
            if spread is None or spread <= 0:
                return "invalid_spread"
            spread_ticks = spread / self.tick_size
            eps = 1e-8
            if spread_ticks + eps < self.min_spread_ticks:
                return "spread_too_small"
            if spread_ticks - eps > self.max_spread_ticks:
                return "spread_too_large"

        if self.enable_liquidity_filter:
            bid_vol = _to_float(row.get(self.bid_volume_col))
            ask_vol = _to_float(row.get(self.ask_volume_col))

            if bid_vol is not None and bid_vol < self.min_bid1_volume:
                return "low_bid_liquidity"
            if ask_vol is not None and ask_vol < self.min_ask1_volume:
                return "low_ask_liquidity"

        if self.enable_limit_price_filter:
            limit_up = _to_float(row.get(self.limit_up_col))
            limit_down = _to_float(row.get(self.limit_down_col))
            buffer_price = self.limit_price_buffer_ticks * self.tick_size

            if limit_up is not None and mid is not None and mid >= limit_up - buffer_price:
                return "near_limit_up"
            if limit_down is not None and mid is not None and mid <= limit_down + buffer_price:
                return "near_limit_down"

        if self.enable_time_filter:
            t = _time_str(row.get(self.datetime_col))
            if t is not None:
                if self.trading_start_time is not None and t < self.trading_start_time:
                    return "before_trading_time"
                if self.trading_end_time is not None and t > self.trading_end_time:
                    return "after_trading_time"

        if self.enable_volatility_filter and self.max_volatility is not None:
            vol = _to_float(row.get(self.volatility_ticks_col), None)
            if vol is not None and vol > self.max_volatility:
                return "high_volatility"

        return None

    def _compute_inventory_skew(self, position_ratio: float) -> float:
        if self.nonlinear_inventory_skew:
            signed = math.copysign(abs(position_ratio) ** self.inventory_skew_power, position_ratio)
        else:
            signed = position_ratio
        return self.inventory_skew_ticks * signed * self.tick_size

    def _inventory_state(self, pos: int, position_ratio: float) -> str:
        if pos >= self.max_position:
            return "max_long_reached"
        if pos <= -self.max_position:
            return "max_short_reached"
        if abs(position_ratio) >= self.danger_position_ratio:
            return "danger_inventory"
        if abs(position_ratio) >= self.warning_position_ratio:
            return "warning_inventory"
        return "normal_inventory"

    def _compute_adverse_buffer(self, spread: float, position_ratio: float) -> float:
        base = self.base_adverse_buffer_ticks * self.tick_size

        if not self.dynamic_adverse_buffer:
            return base

        spread_ticks = spread / self.tick_size

        spread_component = max(0.0, spread_ticks - self.min_spread_ticks)
        spread_component *= self.spread_buffer_multiplier * self.tick_size

        inventory_component = abs(position_ratio) * self.inventory_buffer_multiplier * self.tick_size

        volatility_component = 0.0
        if self.use_spread_as_vol_proxy and spread_ticks > self.high_vol_spread_ticks:
            volatility_component = spread_ticks - self.high_vol_spread_ticks
            volatility_component *= self.volatility_buffer_multiplier * self.tick_size

        return base + spread_component + inventory_component + volatility_component

    def _compute_side_adverse_buffers(
        self,
        row: Mapping[str, Any],
        spread: float,
        position_ratio: float,
        alpha_ticks: float,
        micro_info: Mapping[str, Any],
    ):
        base_buffer = self._compute_adverse_buffer(spread, position_ratio)

        trade_imbalance = _to_float(micro_info.get("trade_imbalance"), self.missing_risk_feature_default)
        cancel_pressure_bid = _to_float(micro_info.get("cancel_pressure_bid"), self.missing_risk_feature_default)
        cancel_pressure_ask = _to_float(micro_info.get("cancel_pressure_ask"), self.missing_risk_feature_default)

        if not self.side_specific_adverse_buffer:
            return (
                base_buffer,
                base_buffer,
                base_buffer,
                0.0,
                0.0,
                0.0,
                0.0,
                trade_imbalance,
                cancel_pressure_bid,
                cancel_pressure_ask,
            )

        alpha_bid_risk_ticks = max(0.0, -alpha_ticks) * self.alpha_toxic_buffer_multiplier
        alpha_ask_risk_ticks = max(0.0, alpha_ticks) * self.alpha_toxic_buffer_multiplier

        bid_trade_risk_ticks = max(0.0, -trade_imbalance) * self.trade_imbalance_buffer_multiplier
        ask_trade_risk_ticks = max(0.0, trade_imbalance) * self.trade_imbalance_buffer_multiplier

        bid_cancel_risk_ticks = max(0.0, cancel_pressure_bid) * self.cancel_pressure_buffer_multiplier
        ask_cancel_risk_ticks = max(0.0, cancel_pressure_ask) * self.cancel_pressure_buffer_multiplier

        bid_risk_score = alpha_bid_risk_ticks + bid_trade_risk_ticks + bid_cancel_risk_ticks
        ask_risk_score = alpha_ask_risk_ticks + ask_trade_risk_ticks + ask_cancel_risk_ticks

        bid_adverse_buffer = base_buffer + bid_risk_score * self.tick_size
        ask_adverse_buffer = base_buffer + ask_risk_score * self.tick_size

        return (
            base_buffer,
            bid_adverse_buffer,
            ask_adverse_buffer,
            bid_risk_score,
            ask_risk_score,
            alpha_bid_risk_ticks,
            alpha_ask_risk_ticks,
            trade_imbalance,
            cancel_pressure_bid,
            cancel_pressure_ask,
        )

    def _side_thresholds(self, base_threshold: float, position_ratio: float) -> Tuple[float, float]:
        bid_threshold = base_threshold
        ask_threshold = base_threshold

        adj = self.inventory_threshold_adjustment_ticks * abs(position_ratio) * self.tick_size

        if position_ratio > 0:
            bid_threshold += adj
            ask_threshold = max(0.0, ask_threshold - adj)
        elif position_ratio < 0:
            ask_threshold += adj
            bid_threshold = max(0.0, bid_threshold - adj)

        return bid_threshold, ask_threshold

    def _generate_candidate_quotes(self, bid1: float, ask1: float, alpha_ticks: float):
        bid_price = bid1
        ask_price = ask1
        quote_style = self.default_quote_style

        if self.quote_mode == "adaptive" and self.improve_when_strong_alpha:
            improve = self.max_improve_ticks * self.tick_size

            if alpha_ticks >= self.strong_alpha_ticks:
                improved_bid = bid1 + improve
                if self.allow_cross or improved_bid < ask1:
                    bid_price = improved_bid
                    quote_style = "improve_bid"

            elif alpha_ticks <= -self.strong_alpha_ticks:
                improved_ask = ask1 - improve
                if self.allow_cross or improved_ask > bid1:
                    ask_price = improved_ask
                    quote_style = "improve_ask"

        bid_price = floor_to_tick(bid_price, self.tick_size)
        ask_price = ceil_to_tick(ask_price, self.tick_size)

        if not self.allow_cross:
            bid_price = min(bid_price, ask1 - self.tick_size)
            ask_price = max(ask_price, bid1 + self.tick_size)
            bid_price = floor_to_tick(bid_price, self.tick_size)
            ask_price = ceil_to_tick(ask_price, self.tick_size)

        return bid_price, ask_price, quote_style

    def _fee_per_share_at_price(self, price: float) -> float:
        if self.fee_mode == "rate":
            return price * self.fee_rate + self.fee_per_share
        return self.fee_per_share

    def _compute_quote_size(self, side: str, edge: float, position_ratio: float) -> int:
        if self.quote_size_mode == "fixed":
            raw_size = self.base_quote_size
        else:
            raw_size = self.base_quote_size

            if self.edge_size_scaling:
                edge_ticks = max(0.0, edge / self.tick_size)
                if self.target_edge_ticks > 0:
                    multiplier = edge_ticks / self.target_edge_ticks
                else:
                    multiplier = 1.0
                multiplier = _clip(multiplier, 1.0, self.max_size_multiplier)
                raw_size *= multiplier

            if self.inventory_aware_size:
                inv_abs = abs(position_ratio)
                decay = self.inventory_size_decay * inv_abs

                if position_ratio > 0:
                    if side == "bid" and self.decrease_entry_side_size:
                        raw_size *= max(0.0, 1.0 - decay)
                    if side == "ask" and self.increase_exit_side_size:
                        raw_size *= 1.0 + decay

                elif position_ratio < 0:
                    if side == "ask" and self.decrease_entry_side_size:
                        raw_size *= max(0.0, 1.0 - decay)
                    if side == "bid" and self.increase_exit_side_size:
                        raw_size *= 1.0 + decay

        raw_size = _clip(raw_size, self.min_quote_size, self.max_quote_size)
        size = round_size_to_lot(raw_size, self.lot_size)

        if size < self.min_quote_size:
            return 0

        return int(size)

    def _risk_state(
        self,
        position_ratio: float,
        alpha_ticks: float,
        alpha_clipped: bool = False,
        volatility_regime: str = "unknown",
        liquidity_state: str = "unknown",
    ) -> str:
        if position_ratio >= 1.0:
            return "max_long_reached"
        if position_ratio <= -1.0:
            return "max_short_reached"
        if abs(position_ratio) >= self.danger_position_ratio:
            return "danger_inventory"
        if abs(position_ratio) >= self.warning_position_ratio:
            return "high_inventory"
        if liquidity_state == "low_liquidity":
            return "low_liquidity"
        if volatility_regime == "high_volatility":
            return "high_volatility"
        if alpha_clipped:
            return "alpha_clipped"
        if abs(alpha_ticks) >= self.strong_alpha_ticks:
            return "strong_alpha"
        if abs(alpha_ticks) < self.weak_alpha_ticks:
            return "weak_alpha"
        return "normal"

    def _spread_regime(self, spread_ticks: float) -> str:
        if spread_ticks <= 1.0:
            return "tight_spread"
        if spread_ticks <= 3.0:
            return "normal_spread"
        if spread_ticks <= 5.0:
            return "wide_spread"
        return "very_wide_spread"

    def _time_regime(self, dt_value: Any) -> str:
        t = _time_str(dt_value)
        if t is None:
            return "unknown_time"
        if t < "09:35:00":
            return "open_period"
        if t >= "14:55:00":
            return "pre_close"
        if "11:30:00" <= t < "13:00:00":
            return "midday_break"
        if t < "11:30:00":
            return "morning_normal"
        return "afternoon_normal"
