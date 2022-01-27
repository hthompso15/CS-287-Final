import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score


# directory
directory = 'stats'
destination = 'plot'
if not os.path.exists(destination):
    os.mkdir(destination)

diet_demo = pd.read_csv(os.path.join(directory,"diet_demo_age20plus_merged.csv"), index_col=0, na_values='')

def poly_spline(df, formula):
    model_poly = smf.ols(formula=formula, data=df)
    result_spline = model_poly.fit()
    #print(result_spline.summary())
    return result_spline


def partialResidualPlot(model, df, outcome, feature, ax):
    y_pred = model.predict(df)
    copy_df = df.copy()
    for c in copy_df.columns:
        if c == feature:
            continue
        copy_df[c] = 0.0
    feature_prediction = model.predict(copy_df)
    results = pd.DataFrame({
        'feature': df[feature],
        'residual': df[outcome] - y_pred,
        'ypartial': feature_prediction - model.params[0],
    })
    results = results.sort_values(by=['feature'])
    #smoothed = sm.nonparametric.lowess(results.ypartial, results.feature, frac=1/6)
    if(feature == 'year'):
        ax.scatter(results.feature, results.ypartial + results.residual,alpha = 0.1, s=1.5, c = df.income, cmap='viridis')
    else:
        ax.scatter(results.feature, results.ypartial + results.residual,alpha = 0.2, s=1.5, c='blue')
    #ax.plot(smoothed[:, 0], smoothed[:, 1], color='black')
    ax.plot(results.feature, results.ypartial, color='r')
    ax.set_xlabel(feature)
    ax.set_ylabel(f'Residual + {feature} contribution')
    return ax


# MODEL 1: Shannon_diet vs. berry and income
# build mini df
shannonD_berry_inc = [diet_demo['Shannon_diet'],diet_demo['berry_crop_6yr_prior'],diet_demo['INDFMPIR'],diet_demo['SDDSRVYR']]
shannonD_b_inc_h = ['shannon_diet','berry_crop','income','year']
shannonD_berry_inc = pd.concat(shannonD_berry_inc,axis=1,keys=shannonD_b_inc_h).dropna()

# train- test split
#print(len(shannonD_berry_inc))
train_df_1 = shannonD_berry_inc.sample(frac=0.8, random_state=1)
#print(len(train_df_1))
test_df_1 = shannonD_berry_inc[~shannonD_berry_inc.isin(train_df_1)].dropna(how = 'all')
#print(len(test_df_1))

# formula 1
f_sd_bi = ('shannon_diet ~ bs(income,df=3,degree=3) + ' + 'berry_crop + year')


# MODEL 2: Shannon_diet vs. (age,year,berry,income)
shannon_diet_aybi =  [diet_demo['Shannon_diet'],diet_demo['SDDSRVYR'],diet_demo['berry_crop_6yr_prior'],diet_demo['INDFMPIR'], diet_demo['Shannon_flavor']]
shannon_aybi_h = ['shannon_diet','year','berry_crop','income', 'shannon_flavor']
shannon_diet_aybi = pd.concat(shannon_diet_aybi,axis=1,keys=shannon_aybi_h).dropna()

# train- test split
train_df_2 = shannon_diet_aybi.sample(frac=0.8, random_state=1)
test_df_2 = shannon_diet_aybi[~shannon_diet_aybi.isin(train_df_2)].dropna(how = 'all')

f_sd_aybi = ('shannon_diet ~ bs(shannon_flavor,df=3,degree=3) + ' + 'year + berry_crop + income')

# Fitting
shannon_berr_inc_M = poly_spline(train_df_1,f_sd_bi)
shannon_diet_aybi_M = poly_spline(train_df_2,f_sd_aybi)

# full model
shannon_berr_inc_M = poly_spline(shannonD_berry_inc,f_sd_bi)
shannon_diet_aybi_M = poly_spline(shannon_diet_aybi,f_sd_aybi)


# make testing prediction
fitted = shannon_berr_inc_M.predict(test_df_1)
RMSE = np.sqrt(mean_squared_error(test_df_1['shannon_diet'], fitted))
r2 = r2_score(test_df_1['shannon_diet'], fitted)

print("model 1:")
print(f'RMSE: {RMSE:.4f}')
print(f'r2: {r2:.4f}')

fitted = shannon_diet_aybi_M.predict(test_df_2)
RMSE = np.sqrt(mean_squared_error(test_df_2['shannon_diet'], fitted))
r2 = r2_score(test_df_2['shannon_diet'], fitted)

print("model 2:")
print(f'RMSE: {RMSE:.4f}')
print(f'r2: {r2:.4f}')

#Plotting
polyfig, axs = plt.subplots(1,2,figsize=(10.5,4))
partialResidualPlot(shannon_berr_inc_M,shannonD_berry_inc,'shannon_diet','income',axs[0])
partialResidualPlot(shannon_diet_aybi_M,shannon_diet_aybi,'shannon_diet','shannon_flavor',axs[1])
plt.tight_layout()
plt.savefig(os.path.join(destination, 'model1+2.png'), dpi=300, bbox_inches='tight')


#x_lim = np.linspace(shannon_diet_aybi[['income']].min(), shannon_diet_aybi[['income']].max(), 100)
#fig, ax = plt.subplots(figsize=(8, 6))
#fit_100 = shannon_berr_inc_M.predict(x_lim)

#ax.scatter(shannon_diet_aybi['income'], shannon_diet_aybi['shannon_diet'], facecolor='None', edgecolor='k', alpha=0.1)
#plt.plot(x_lim, fit_100, color='y', linewidth = 1.5, label='Specifying 3 knots, knots = (15, 30, 45)')


# predictions = shannon_berr_inc_M.get_prediction(shannon_diet_aybi)
# df_predictions = predictions.summary_frame()
# df_predictions['income'] = shannon_diet_aybi.income.values

#fig = sm.graphics.abline_plot(model_results=shannon_berr_inc_M)
#sns.scatterplot(x='income', y='shannon_diet', data=shannon_diet_aybi, label='data', ax=ax)
# sns.lineplot(x='income', y='mean', data=df_predictions, label='fit', ax=ax)
# ax.fill_between(
#     x=df_predictions.income,
#     y1=df_predictions.mean_ci_lower,
#     y2=df_predictions.mean_ci_upper,
#     #color=sns_c[3],
#     alpha=0.5,
#     label='Confidence Interval (mean_pred)'
# )
#plt.savefig(os.path.join(destination, 'model1.png'), dpi=300, bbox_inches='tight')



# ax.plot(shannon_diet_aybi['income'], shannon_diet_aybi['shannon_diet'], "o", label="data")
# #ax.plot(x, res.fittedvalues, "r--.", label="OLS")
# plt.plot(df_predictions['mean'], color='crimson')
# plt.fill_between(df_predictions.index, df_predictions.mean_ci_lower, df_predictions.mean_ci_upper, alpha=.1, color='crimson')
# plt.fill_between(df_predictions.index, df_predictions.obs_ci_lower, df_predictions.obs_ci_upper, alpha=.1, color='crimson')
# #ax.legend(loc="best")
# plt.savefig(os.path.join(destination, 'model1.png'), dpi=300, bbox_inches='tight')

#st, data, ss2 = summary_table(shannon_berr_inc_M, alpha=0.05)
#predictions.summary_frame(alpha=0.05)


#Plotting
#polyfig, axs = plt.subplots(1,2,figsize=(10,5))
#partialResidualPlot(shannon_berr_inc_M,shannonD_berry_inc,'shannon_diet','year',axs[0])
#partialResidualPlot(shannon_diet_aybi_M,shannon_diet_aybi,'shannon_diet','income',axs[1])