#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:22:15 2026

@author: venkateshchandra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

np.random.seed(42)

#Problem statement - Does this new feature increase merchant revenue compared to what would have happened without the feature?

# Parameters
n_merchants_per_geo = 200
geos = ["Geo_A", "Geo_B", "Geo_C", "Geo_D", "Geo_E"]
treated_geo = "Geo_A"
months = np.arange(1, 25)   # 24 months
treatment_start = 13

#Sample data
rows = []

for geo in geos:
    for merchant_id in range(n_merchants_per_geo):
        merchant_size = np.random.normal(10000, 2000)   # monthly baseline revenue
        avg_ticket = np.random.normal(50, 10)
        tenure = np.random.randint(1, 60)

        geo_trend = np.random.normal(50, 10)  # slight geo-specific trend

        for month in months:
            time_trend = 100 * month

            treatment_effect = 0
            if geo == treated_geo and month >= treatment_start:
                treatment_effect = 1500   # true treatment impact

            revenue = (
                merchant_size
                + time_trend
                + geo_trend * month
                + 10 * avg_ticket
                + 20 * tenure
                + treatment_effect
                + np.random.normal(0, 2000)
            )

            rows.append([
                geo,
                merchant_id,
                month,
                revenue,
                merchant_size,
                avg_ticket,
                tenure
            ])

df = pd.DataFrame(rows, columns=[
    "geo", "merchant_id", "month", "revenue",
    "merchant_size", "avg_ticket", "tenure"
])

df

# Treatment indicators
df["post"] = (df["month"] >= treatment_start).astype(int)
df["treated"] = (df["geo"] == treated_geo).astype(int)
df["did"] = df["post"] * df["treated"]

df.head()


#--------Check parallel trends assumptions-------
#We first check pre treatment trends. For DiD to be valid, treated and control groups must move in parallel before month 13.
#Aggregate average revenue per geo per month.


agg = df.groupby(["geo", "month"])["revenue"].mean().reset_index()

# Plot pre period trends
plt.figure()
for geo in geos:
    temp = agg[(agg["geo"] == geo) & (agg["month"] < treatment_start)]
    plt.plot(temp["month"], temp["revenue"], label=geo)

plt.axvline(x=treatment_start, linestyle="--")
plt.legend()
plt.title("Pre Treatment Trends")
plt.xlabel("Month")
plt.ylabel("Average Revenue")
plt.show()

#--------Selecting best control geo-------

from sklearn.linear_model import LinearRegression

def get_slope(data):
    X = data["month"].values.reshape(-1, 1)
    y = data["revenue"].values
    model = LinearRegression().fit(X, y)
    return model.coef_[0]

pre_slopes = {}

for geo in geos:
    temp = agg[(agg["geo"] == geo) & (agg["month"] < treatment_start)]
    pre_slopes[geo] = get_slope(temp)

treated_slope = pre_slopes[treated_geo]

slope_diff = {
    geo: abs(pre_slopes[geo] - treated_slope)
    for geo in geos if geo != treated_geo
}

best_control = min(slope_diff, key=slope_diff.get)

print("Pre slopes:", pre_slopes)
print("Slope differences:", slope_diff)
print("Selected control geo:", best_control)

# Plot pre period trends for best control geo
plt.figure()
for geo in ['Geo_C', treated_geo]:
    temp = agg[(agg["geo"] == geo) & (agg["month"] < treatment_start)]
    plt.plot(temp["month"], temp["revenue"], label=geo)

plt.axvline(x=treatment_start, linestyle="--")
plt.legend()
plt.title("Pre Treatment Trends")
plt.xlabel("Month")
plt.ylabel("Average Revenue")
plt.show()

#--------Running DiD-------


df_sub = df[df["geo"].isin([treated_geo, best_control])].copy()

model = smf.ols(
    "revenue ~ treated + post + did + merchant_size + avg_ticket + tenure + C(month)",
    data=df_sub
).fit(cov_type="cluster", cov_kwds={"groups": df_sub["merchant_id"]})

print(model.summary())

#Model explanation
'''
OLS is used because:

The DiD model is linear in parameters.

We want an interpretable coefficient for the interaction term.

We need statistical inference: standard errors, p values, confidence intervals.

The estimator has a clear causal interpretation under assumptions.

We add controls to reduce bias.
If after adding controls:

• did shrinks a lot
• becomes insignificant

That means part of what looked like treatment effect was actually driven by changing composition.
'''

#--------Visualize treatment effect-------

agg_sub = df_sub.groupby(["geo", "month"])["revenue"].mean().reset_index()

plt.figure()

for geo in [treated_geo, best_control]:
    temp = agg_sub[agg_sub["geo"] == geo]
    plt.plot(temp["month"], temp["revenue"], label=geo)

plt.axvline(x=treatment_start, linestyle="--")
plt.legend()
plt.title("DiD Visual")
plt.xlabel("Month")
plt.ylabel("Average Revenue")
plt.show()


'''
Null hypothesis:

Treatment has no impact
beta_did = 0

Alternative hypothesis:

Treatment has impact
beta_did ≠ 0

If beta_did > 0 and statistically significant:
Treatment increases revenue.

If beta_did < 0 and statistically significant:
Treatment decreases revenue.
'''
#--------Test pre-period interaction-------

df_sub["time_trend"] = df_sub["month"]

pre_test = smf.ols(
    "revenue ~ treated * time_trend + merchant_size + avg_ticket + tenure",
    data=df_sub[df_sub["month"] < treatment_start]
).fit()

print(pre_test.summary())

'''
In pre treatment period only:

revenue ~ treated * time_trend

If treated × time_trend is significant:

It means treated geo had a different slope before treatment.
Parallel trends violated.
DiD invalid.

If it is not significant:

Pre trends are statistically similar.
Parallel trends plausible.

Here, the p value is 0.713 which is very large. So we fail to reject the null that slopes are equal.
There is no statistical evidence that pre trends differ. Parallel trends assumption looks plausible.
'''






