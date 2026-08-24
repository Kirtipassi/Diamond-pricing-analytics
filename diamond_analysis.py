"""
Diamond Pricing Analytics — Exploratory & Predictive Analysis
Dataset: 53,940 real diamond transactions (carat, cut, color, clarity, price, dimensions)
Source: originally published with ggplot2 (Hadley Wickham); mirrored at mwaskom/seaborn-data.

Goal (data-analyst framing):
  1. Clean and validate transaction-level data
  2. Quantify which attributes drive price (carat, cut, color, clarity)
  3. Segment inventory into pricing tiers and quantify each tier's share
  4. Detect pricing outliers / data quality issues
  5. Build a simple, interpretable price-prediction model and report accuracy
  6. Export cleaned data + an interactive dashboard (importable into Tableau too)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_absolute_error

sns.set_theme(style="whitegrid")
OUT = "/home/claude/project"

# ---------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------
df = pd.read_csv(f"{OUT}/diamonds.csv")
n_raw = len(df)

# Data quality: remove physically impossible rows (0 dimensions = data entry errors)
df = df[(df.x > 0) & (df.y > 0) & (df.z > 0)]
# Remove extreme dimension outliers (likely typos, e.g. y=58.9 vs normal ~5-10mm)
df = df[(df.y < 20) & (df.z < 20) & (df.table < 90) & (df.table > 40) & (df.depth < 80) & (df.depth > 40)]

n_clean = len(df)
print(f"Rows: {n_raw} raw -> {n_clean} clean ({n_raw - n_clean} removed as data-quality issues)")

# Ordered categories (quality ordering, not alphabetical — matters for correlation/EDA)
cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order = ["J", "I", "H", "G", "F", "E", "D"]          # J worst -> D best
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]  # worst -> best

df["cut"] = pd.Categorical(df["cut"], categories=cut_order, ordered=True)
df["color"] = pd.Categorical(df["color"], categories=color_order, ordered=True)
df["clarity"] = pd.Categorical(df["clarity"], categories=clarity_order, ordered=True)

df["price_per_carat"] = df["price"] / df["carat"]

# ---------------------------------------------------------------
# 2. PRICE SEGMENTATION (business framing: budget / mid / premium / luxury)
# ---------------------------------------------------------------
bins = [0, 1000, 5000, 10000, np.inf]
labels = ["Budget (<$1K)", "Mid ($1K-5K)", "Premium ($5K-10K)", "Luxury ($10K+)"]
df["price_segment"] = pd.cut(df["price"], bins=bins, labels=labels)

segment_summary = (
    df.groupby("price_segment", observed=True)
    .agg(count=("price", "size"), avg_carat=("carat", "mean"), avg_price=("price", "mean"))
    .assign(pct_of_inventory=lambda d: (d["count"] / len(df) * 100).round(1))
)
print("\n--- Price Segment Summary ---")
print(segment_summary)

# ---------------------------------------------------------------
# 3. WHAT DRIVES PRICE? correlation + grouped price/carat analysis
# ---------------------------------------------------------------
numeric_corr = df[["price", "carat", "depth", "table", "x", "y", "z"]].corr()["price"].sort_values(ascending=False)
print("\n--- Correlation with price ---")
print(numeric_corr)

cut_price = df.groupby("cut", observed=True)["price_per_carat"].mean().reindex(cut_order)
color_price = df.groupby("color", observed=True)["price_per_carat"].mean().reindex(color_order)
clarity_price = df.groupby("clarity", observed=True)["price_per_carat"].mean().reindex(clarity_order)

# ---------------------------------------------------------------
# 4. OUTLIER DETECTION (IQR method on price_per_carat within carat bins)
# ---------------------------------------------------------------
df["carat_bin"] = pd.cut(df["carat"], bins=[0, 0.5, 1, 1.5, 2, 5], labels=["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2+"])

def flag_outliers(group):
    q1, q3 = group["price_per_carat"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (group["price_per_carat"] < lower) | (group["price_per_carat"] > upper)

df["is_outlier"] = df.groupby("carat_bin", observed=True, group_keys=False).apply(flag_outliers)
n_outliers = df["is_outlier"].sum()
print(f"\nPricing outliers flagged (IQR method, within carat bands): {n_outliers} ({n_outliers/len(df)*100:.2f}%)")

# ---------------------------------------------------------------
# 5. PREDICTIVE MODEL: interpretable linear regression on log(price)
# ---------------------------------------------------------------
model_df = df.copy()
enc = OrdinalEncoder(categories=[cut_order, color_order, clarity_order])
model_df[["cut_enc", "color_enc", "clarity_enc"]] = enc.fit_transform(
    model_df[["cut", "color", "clarity"]]
)
model_df["log_price"] = np.log(model_df["price"])
model_df["log_carat"] = np.log(model_df["carat"])

features = ["log_carat", "cut_enc", "color_enc", "clarity_enc", "depth", "table"]
X = model_df[features]
y = model_df["log_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
r2 = r2_score(y_test, pred)
mae_pct = np.mean(np.abs((np.exp(pred) - np.exp(y_test)) / np.exp(y_test))) * 100

print(f"\n--- Price Prediction Model (log-linear regression) ---")
print(f"R^2 on held-out test set: {r2:.3f}")
print(f"Mean absolute % error: {mae_pct:.1f}%")
coef_impact = pd.Series(lr.coef_, index=features).sort_values(key=abs, ascending=False)
print("Feature impact (standardized log-scale coefficients):")
print(coef_impact)

# ---------------------------------------------------------------
# 6. STATIC VISUALS (matplotlib/seaborn) — good for a report/slide
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.scatterplot(data=df.sample(4000, random_state=1), x="carat", y="price", hue="cut",
                 palette="viridis", alpha=0.5, s=15, ax=axes[0, 0])
axes[0, 0].set_title("Price vs Carat (colored by Cut) — non-linear premium at scale")

sns.barplot(x=cut_price.index, y=cut_price.values, ax=axes[0, 1], palette="crest")
axes[0, 1].set_title("Avg Price/Carat by Cut Quality")
axes[0, 1].set_ylabel("$ per carat")

sns.heatmap(numeric_corr.to_frame(), annot=True, cmap="coolwarm", center=0, ax=axes[1, 0], cbar=False)
axes[1, 0].set_title("Correlation with Price")

segment_summary["pct_of_inventory"].plot(kind="bar", ax=axes[1, 1], color="#2a6f97")
axes[1, 1].set_title("Inventory Share by Price Segment (%)")
axes[1, 1].set_ylabel("% of listings")
plt.setp(axes[1, 1].get_xticklabels(), rotation=20, ha="right")

plt.tight_layout()
plt.savefig(f"{OUT}/eda_summary.png", dpi=150)
print(f"\nSaved static chart panel -> eda_summary.png")

# ---------------------------------------------------------------
# 7. INTERACTIVE DASHBOARD (Plotly, single self-contained HTML)
# ---------------------------------------------------------------
sample = df.sample(6000, random_state=1)

fig_dash = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Price vs Carat by Cut", "Avg $/Carat by Clarity Grade",
        "Inventory Mix by Price Segment", "Avg $/Carat by Color Grade"
    ),
    specs=[[{"type": "scatter"}, {"type": "bar"}], [{"type": "pie"}, {"type": "bar"}]],
)

for c in cut_order:
    sub = sample[sample.cut == c]
    fig_dash.add_trace(
        go.Scatter(x=sub["carat"], y=sub["price"], mode="markers", name=c,
                    marker=dict(size=5, opacity=0.55)),
        row=1, col=1,
    )

fig_dash.add_trace(
    go.Bar(x=list(clarity_price.index), y=clarity_price.values, marker_color="#2a9d8f", showlegend=False),
    row=1, col=2,
)

fig_dash.add_trace(
    go.Pie(labels=segment_summary.index.tolist(), values=segment_summary["count"].tolist(), hole=0.4),
    row=2, col=1,
)

fig_dash.add_trace(
    go.Bar(x=list(color_price.index), y=color_price.values, marker_color="#e76f51", showlegend=False),
    row=2, col=2,
)

fig_dash.update_layout(
    height=850, width=1100,
    title_text="Diamond Pricing Analytics Dashboard — 53K+ Real Transactions",
    legend_title_text="Cut",
)
fig_dash.write_html(f"{OUT}/diamond_dashboard.html")
print("Saved interactive dashboard -> diamond_dashboard.html")

# ---------------------------------------------------------------
# 8. EXPORT CLEANED DATA (for Tableau / Power BI import)
# ---------------------------------------------------------------
df.drop(columns=["is_outlier"]).to_csv(f"{OUT}/diamonds_cleaned.csv", index=False)
print("Saved cleaned dataset -> diamonds_cleaned.csv (ready for Tableau/Power BI import)")

print("\n=== DONE ===")
