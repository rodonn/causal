# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
import statsmodels.formula.api as smf
import statsmodels.api as sm
from xgboost import XGBRegressor, XGBClassifier

# %%
# Install the package for calling R from python
#%pip install rpy2
## activate R magic
import rpy2
## Install packages in R if needed
#%R install.packages('hdm')
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter

# %% [markdown]
# ## Savings and 401k Plans
# The sample is drawn from the 1991 Survey of Income and Program Participation (SIPP) and consists
# of 9,915 observations. The observational units are household reference persons aged 25-64 and
# spouse if present. Households are included in the sample if at least one person is employed and no
# one is self-employed. The data set was analysed in Chernozhukov and Hansen (2004) and Belloni
# et al. (2014) where further details can be found. They examine the effects of 401(k) plans on
# wealth using data from the Survey of Income and Program Participation using 401(k) eligibility as
# an instrument for 401(k) participation.
# 
# * p401 = participation in 401(k)
# * e401 = eligibility for 401(k)
# * a401 = 401(k) assets
# * tw = total wealth (in US $)
# * tfa = financial assets (in US $)
# * net_tfa = net financial assets (in US $)
# * nifa = non-401k financial assets (in US $)
# * net_nifa = net non-401k financial assets
# * net_n401 = net non-401(k) assets (in US $)
# * ira = individual retirement account (IRA)
# * inc = income (in US $)
# * age = age
# * fsize = family size
# * marr = married
# * pira = participation in IRA
# * db = defined benefit pension
# * hown = home owner
# * educ = education (in years)
# * male = male
# * twoearn = two earners
# * nohs, = hs, smcol, col dummies for education: no high-school, high-school, some college, college
# * hmort = home mortage (in US $)
# * hequity = home equity (in US $)
# * hval = home value (in US $)

# %%
# Load dataset from R package and return as a pandas dataframe
with localconverter(ro.default_converter + pandas2ri.converter):
    df = ro.conversion.rpy2py(rpy2.robjects.packages.PackageData('hdm').fetch('pension')['pension'])
df

# %%
T = 'e401'
Y = 'net_tfa'
X = ['age', 'inc', 'fsize', 'educ', 'pira', 'hown', 'marr', 'db', 'twoearn']

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.kdeplot(df, x="net_tfa", hue="e401", fill=True, ax=ax)
ax.set_xlim(-50000, 100000)
ax.set_xlabel("Net Total Family Assets")
ax.set_title("Distribution of Net Total Family Assets by 401(k) Eligibility");
#sns.displot(df, x="net_tfa", hue="e401", kind="hist", fill=True, log_scale=(False, True) )

# %%
df['e401_c'] = df['e401'].astype('category')
sns.catplot(data=df, y="e401_c", x="net_tfa", kind="violin", height=5, aspect=3)
#sns.catplot(data=df, x="e401", y="net_tfa", alpha=0.1)

# %%
smf.ols('net_tfa ~ e401', data=df).fit().summary().tables[1]

# %%
# Fit a linear regression model with controls for the other potential confounders
smf.ols('net_tfa ~ e401 + age + inc + fsize + educ + pira + hown + marr + db + twoearn', data=df).fit().summary().tables[1]

# %% [markdown]
# ## Train a single ML model to predict outcomes

# %%
# Train a single ML model
single_model = XGBRegressor()
single_model.fit(df[X + [T]], df[Y])
# Predict the outcome for both possible treatment values
X0 = df[X + [T]].copy(); X0[T] = 0
X1 = df[X + [T]].copy(); X1[T] = 1
df['y0'] = single_model.predict(X0)
df['y1'] = single_model.predict(X1)
# Compute the difference in outcomes
df['y_diff'] = df['y1'] - df['y0'];

# %%
sns.kdeplot(df, x="y_diff")
plt.xlabel("Difference in Predicted Outcomes")
plt.title("Distribution of Predicted Treatment Effects");
df['y_diff'].describe()

# %% [markdown]
# ## Train Treatment and Outcome ML models, without cross fitting

# %%
# Fit model to predict T
treatment_model = XGBClassifier()
treatment_model.fit(df[X], df[T])
T_pred = treatment_model.predict_proba(df[X])[:,1]
T_resid = df[T] - T_pred

# Plot Y against the T residuals
#sns.regplot(data=df, x=T_resid, y=Y, scatter_kws={'alpha': 0.1}, line_kws={'color': 'red'});

# %%
# Fit model to predict Y
outcome_model = XGBRegressor()
outcome_model.fit(df[X], df[Y])
Y_pred = outcome_model.predict(df[X])
Y_resid = df[Y] - Y_pred

# Plot Y residuals against the T residuals
#sns.regplot(x=T_resid, y=Y_resid, scatter_kws={'alpha': 0.1}, line_kws={'color': 'red'});

# %%
# Fit linear model to predict Y_resid based on T_residual
sm.OLS(Y_resid, T_resid).fit().summary().tables[1]

# %% [markdown]
# # Train Treatment and Outcome ML models using Cross-splitting

# %%
# Cross_val_predict splits the data into 5 parts and then returns out-of-sample predictions for each part (trained on the other 4 parts)

# Fit model to predict T
treatment_model = XGBClassifier()
T_pred = cross_val_predict(treatment_model, df[X], df[T], cv=5, method='predict_proba')[:, 1]
T_resid = df[T] - T_pred

# Fit model to predict Y
outcome_model = XGBRegressor()
Y_pred = cross_val_predict(outcome_model, df[X], df[Y], cv=5)
Y_resid = df[Y] - Y_pred

# %%
# Fit linear regression on the residuals
sm.OLS(Y_resid, T_resid).fit().summary().tables[1]

# %% [markdown]
# # Using the EconML Package

# %%
from econml.dml import LinearDML

# %%
model = LinearDML(
    model_y=XGBRegressor(),
    model_t=XGBClassifier(),
    discrete_treatment=True,
    cv=5,
)
model.fit(df[Y], df[T], X=None, W=df[X]);

# %%
model.summary()

# %%
model.intercept__interval()


