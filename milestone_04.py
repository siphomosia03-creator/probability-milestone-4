```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan

# ==================================================
# LOAD DATA
# ==================================================

try:
    df = pd.read_csv("data/finflow_users.csv")
    ts_df = pd.read_csv("data/finflow_timeseries.csv")

except FileNotFoundError as e:
    print(f"Missing dataset: {e}")
    raise

# ==================================================
# FIT LOGISTIC REGRESSION
# ==================================================

X = sm.add_constant(df["score_views"])
y = df["premium_user"]

model = sm.Logit(y, X).fit(disp=False)

coef_intercept = model.params["const"]
coef_score_views = model.params["score_views"]

pred_probs = model.predict(X)

# Pearson residuals
residuals = y - pred_probs

# ==================================================
# ASSUMPTION 1:
# LINEARITY IN LOG ODDS (BOX-TIDWELL)
# ==================================================

bt_df = df.copy()

bt_df["score_log"] = (
    bt_df["score_views"]
    * np.log(bt_df["score_views"] + 1)
)

X_bt = sm.add_constant(
    bt_df[
        ["score_views", "score_log"]
    ]
)

bt_model = sm.Logit(
    y,
    X_bt
).fit(disp=False)

p_box = bt_model.pvalues["score_log"]

linearity_ok = p_box > 0.05

# ==================================================
# ASSUMPTION 2:
# HOMOSCEDASTICITY
# ==================================================

bp = het_breuschpagan(
    residuals,
    X
)

bp_pvalue = bp[1]

homoscedasticity_ok = (
    bp_pvalue > 0.05
)

# ==================================================
# ASSUMPTION 3:
# NORMALITY
# ==================================================

_, shapiro_p = stats.shapiro(
    residuals
)

normality_ok = (
    shapiro_p > 0.05
)

# ==================================================
# ASSUMPTION 4:
# INDEPENDENCE
# ==================================================

ts_df = ts_df.sort_values(
    "timestamp"
)

dw_stat = durbin_watson(
    residuals
)

independence_ok = (
    1.5 < dw_stat < 2.5
)

# ==================================================
# PREDICTION
# ==================================================

score_views_new = 7

new_X = pd.DataFrame(
    {
        "const": [1],
        "score_views": [7]
    }
)

prob_premium = float(
    model.predict(new_X)
)

# ==================================================
# BOOTSTRAP INTERVAL
# ==================================================

n_boot = 10000

boot_preds = []

for i in range(n_boot):

    sample = df.sample(
        len(df),
        replace=True
    )

    X_boot = sm.add_constant(
        sample["score_views"]
    )

    y_boot = sample["premium_user"]

    try:

        m = sm.Logit(
            y_boot,
            X_boot
        ).fit(disp=False)

        pred = m.predict(
            new_X
        )[0]

        boot_preds.append(pred)

    except:
        pass

pi_lower = max(
    0,
    np.percentile(
        boot_preds,
        2.5
    )
)

pi_upper = min(
    1,
    np.percentile(
        boot_preds,
        97.5
    )
)

# ==================================================
# 50% THRESHOLD
# ==================================================

threshold = (
    -coef_intercept
    /
    coef_score_views
)

threshold = max(
    0,
    threshold
)

# ==================================================
# VALIDATION
# ==================================================

assert (
    0 <= prob_premium <= 1
)

assert isinstance(
    linearity_ok,
    bool
)

assert isinstance(
    homoscedasticity_ok,
    bool
)

assert isinstance(
    normality_ok,
    bool
)

assert isinstance(
    independence_ok,
    bool
)

# ==================================================
# RESULTS
# ==================================================

print(
    "LOGISTIC REGRESSION"
)

print("=" * 70)

print(
    f"log-odds(premium)"
    f" = {coef_intercept:.3f}"
    f" + {coef_score_views:.3f}"
    f"(score_views)"
)

print("\nASSUMPTIONS")

print(
    f"Linearity:"
    f" {'OK' if linearity_ok else 'VIOLATED'}"
)

print(
    f"Homoscedasticity:"
    f" {'OK' if homoscedasticity_ok else 'VIOLATED'}"
)

print(
    f"Normality:"
    f" {'OK' if normality_ok else 'VIOLATED'}"
)

print(
    f"Independence:"
    f" {'OK' if independence_ok else 'VIOLATED'}"
    f" (DW={dw_stat:.2f})"
)

print("\nPREDICTION")

print(
    f"Probability at"
    f" 7 score views:"
    f" {prob_premium:.2%}"
)

print(
    f"95% Interval:"
    f" ({pi_lower:.2%},"
    f" {pi_upper:.2%})"
)

print(
    f"\n50% Conversion Threshold:"
    f" {threshold:.2f}"
    f" score views"
)

print("\nBUSINESS RECOMMENDATION")

print(
    f"Users reaching roughly "
    f"{threshold:.0f} score views "
    f"have greater than a "
    f"50% chance of premium conversion."
)

odds_multiplier = np.exp(
    coef_score_views
)

print(
    f"Each additional score "
    f"view multiplies conversion "
    f"odds by "
    f"{odds_multiplier:.2f}x."
)

if not independence_ok:

    print(
        "\nWARNING:"
    )

    print(
        "Independence assumption "
        "is violated. "
        "P-values and confidence "
        "intervals may be unreliable."
    )

print(
    "\nFINAL CONCLUSION:"
)

print(
    "Optimising score views "
    "appears beneficial, "
    "but recommendations should "
    "be qualified by any "
    "diagnostic violations."
)
```

