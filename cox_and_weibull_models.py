#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 22:14:39 2026

@author: venkateshchandra
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ---------------------------
# 1. Simulate merchant data
# ---------------------------

n = 800

monthly_gmv = np.random.normal(50000, 15000, n)          # monthly payment volume
num_invoices = np.random.normal(120, 40, n)              # invoices per month
avg_invoice_amount = monthly_gmv / np.maximum(num_invoices, 1)
tenure_days = np.random.normal(365, 200, n)              # time already on platform

# baseline churn timing
baseline_time = np.random.exponential(scale=400, size=n)

# risk mechanism
# higher GMV -> lower churn
# more invoices -> lower churn
# higher avg invoice amount -> slightly higher churn
# low tenure -> higher churn

risk = (
    -0.00002 * monthly_gmv
    -0.005 * num_invoices
    +0.0005 * avg_invoice_amount
    -0.001 * tenure_days
)

# generate time to churn
T = baseline_time * np.exp(-risk)

# censoring: some merchants still active
E = np.random.binomial(1, 0.65, n)

df = pd.DataFrame({
    "T": T,
    "E": E,
    "monthly_gmv": monthly_gmv,
    "num_invoices": num_invoices,
    "avg_invoice_amount": avg_invoice_amount,
    "tenure_days": tenure_days
})

# ---------------------------
# 2. Train test split
# ---------------------------

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ---------------------------
# 3. Fit Cox model
# ---------------------------

cph = CoxPHFitter()
cph.fit(train_df, duration_col="T", event_col="E")

cph.print_summary()

print("Train C index:", cph.concordance_index_)
print("Test C index:", cph.score(test_df, scoring_method="concordance_index"))

# ---------------------------
# 4. Kaplan Meier curve - Poppulation level surviva
# ---------------------------

kmf = KaplanMeierFitter()
kmf.fit(df["T"], event_observed=df["E"])

kmf.plot()
plt.title("Merchant Survival Curve: Payments")
plt.xlabel("Days")
plt.ylabel("Probability Still Active")
plt.show()

# ---------------------------
# 5. Predict survival curve for one merchant
# ---------------------------

sample = test_df.iloc[[0]]
surv_func = cph.predict_survival_function(sample)

surv_func.plot()
plt.title("Predicted Survival Curve for One Merchant")
plt.xlabel("Days")
plt.ylabel("Probability Still Active")
plt.show()

# ---------------------------
# 6. Partial effect: GMV impact
# ---------------------------

cph.plot_partial_effects_on_outcome(
    covariates="monthly_gmv",
    values=[20000, 50000, 80000]
)

plt.title("Effect of Monthly GMV on Churn Risk")
plt.show()


#----------Weibull-----------


from lifelines import WeibullAFTFitter

aft = WeibullAFTFitter()
aft.fit(train_df, duration_col="T", event_col="E")
aft.print_summary()


# Predict expected time until churn for each merchant
expected_T = aft.predict_median(test_df)  # median survival time
print(expected_T.head())


aft.predict_survival_function(test_df.iloc[[0]]).plot()
plt.title("Predicted Weibull Survival Curve for One Merchant")
plt.xlabel("Days")
plt.ylabel("Probability Still Active")
plt.show()




