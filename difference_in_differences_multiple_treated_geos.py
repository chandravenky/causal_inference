#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 18:13:20 2026

@author: venkateshchandra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

np.random.seed(42)

# --------------------------
# 1. Simulate panel data
# --------------------------
n_merchants_per_geo = 100
geos = ["Geo_A", "Geo_B", "Geo_C", "Geo_D", "Geo_E"]
treated_geos = ["Geo_A", "Geo_B"]  # multiple treated
months = np.arange(1, 25)
treatment_start = 13

rows = []

for geo in geos:
    for merchant_id in range(n_merchants_per_geo):
        merchant_size = np.random.normal(10000, 2000)
        avg_ticket = np.random.normal(50, 10)
        tenure = np.random.randint(1, 60)
        geo_trend = np.random.normal(50, 10)
        for month in months:
            time_trend = 100 * month
            treatment_effect = 0
            if geo in treated_geos and month >= treatment_start:
                treatment_effect = 1500  # treatment effect
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

# --------------------------
# 2. Treatment indicators
# --------------------------
df["post"] = (df["month"] >= treatment_start).astype(int)
df["treated"] = df["geo"].isin(treated_geos).astype(int)
df["did"] = df["treated"] * df["post"]

# --------------------------
# 3. Optional: pre-trend plot
# --------------------------
agg = df.groupby(["geo", "month"])["revenue"].mean().reset_index()
plt.figure(figsize=(8,5))
for geo in geos:
    temp = agg[agg["geo"]==geo]
    plt.plot(temp["month"], temp["revenue"], label=geo)
plt.axvline(x=treatment_start, linestyle="--", color="black")
plt.title("Revenue Trends Pre & Post Treatment")
plt.xlabel("Month")
plt.ylabel("Avg Revenue")
plt.legend()
plt.show()

# --------------------------
# 4. Run DiD regression with multiple treated geos
# Using unit fixed effects via merchant_id, time fixed effects via month
# --------------------------
model = smf.ols(
    "revenue ~ did + merchant_size + avg_ticket + tenure + C(month) + C(geo)",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["merchant_id"]})

print(model.summary())

'''
DiD estimate is stat sig and gives the average treatment effect across all treated geos
'''


#--------Estimate for one geo--------

#Exclude other treated geo
df_single = df[df['geo'] == "Geo_A"].copy()
df_single['treated'] = 1  # Geo_A is treated
df_single['post'] = (df_single['month'] >= treatment_start).astype(int)
df_single['did'] = df_single['treated'] * df_single['post']

df_control = df[df['geo'].isin(["Geo_C", "Geo_D", "Geo_E"])].copy()
df_single_control = pd.concat([df_single, df_control])

model = smf.ols(
    "revenue ~ did + merchant_size + avg_ticket + tenure + C(month) + C(geo)",
    data=df_single_control
).fit(cov_type="cluster", cov_kwds={"groups": df_single_control["merchant_id"]})

print(model.summary())


#--------Estimate for one geo--------

#Interaction term approach
#In one regression

for geo in treated_geos:
    df[f'did_{geo}'] = ((df['geo']==geo) & (df['month']>=treatment_start)).astype(int)

formula = "revenue ~ " + " + ".join([f'did_{geo}' for geo in treated_geos]) + " + merchant_size + avg_ticket + tenure + C(month) + C(geo)"

model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["merchant_id"]})
print(model.summary())


