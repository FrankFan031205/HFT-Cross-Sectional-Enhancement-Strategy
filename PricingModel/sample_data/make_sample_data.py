import pandas as pd
import numpy as np
from pathlib import Path

out = Path("PricingModel/sample_data")
out.mkdir(parents=True, exist_ok=True)

n = 1000

date = "20241022"
code = "000001"

timestamps = np.arange(n)

mid = 10 + np.cumsum(np.random.normal(0, 0.002, n))
bid1 = np.round(mid - 0.005, 2)
ask1 = np.round(mid + 0.005, 2)

market = pd.DataFrame({
    "date": date,
    "code": code,
    "timestamp": timestamps,
    "bid1": bid1,
    "ask1": ask1,
})

pred = pd.DataFrame({
    "date": date,
    "code": code,
    "timestamp": timestamps,
    "pred_ret": np.random.normal(0, 0.001, n),
})

market.to_csv(out / "market.csv", index=False)
pred.to_csv(out / "pred.csv", index=False)

print("sample data generated")
print(out / "market.csv")
print(out / "pred.csv")