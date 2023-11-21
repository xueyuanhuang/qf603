import pandas as pd;
import numpy as np;
import statsmodels.api as sm;
import statsmodels.stats.api as sms;
import statsmodels.discrete.discrete_model as smdiscrete
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


pd.set_option('use_inf_as_na', True)

meta_df = pd.read_csv("stockmetadata.csv")
fdata_df = pd.read_csv("corpfund.csv")
fdata_df = fdata_df[fdata_df['dimension']=='ARQ']
fdata_df['datekey'] = pd.to_datetime(fdata_df['datekey'])
df_left = pd.merge(fdata_df, meta_df, on='ticker', how='left')
df_left = df_left.set_index('datekey')


industrydummies = pd.get_dummies(df_left['sicsector'])
industrydummies.sum()		#purely for exploring the data, has no other purpose
industrydummies.describe()      #purely for exploring the data, has no other purpose

data_w_dummies = pd.concat([df_left,industrydummies], axis=1)

data_w_dummies.drop(['Wholesale Trade'], inplace=True, axis=1)	#drop 1 dummy variable
data_w_dummies['epratio'] = data_w_dummies['eps']/data_w_dummies['price']	#generate dependent variable
data_w_dummies['operatingmargin'] = data_w_dummies['opinc'] / data_w_dummies['revenue']


#initial analysis
result = sm.OLS(data_w_dummies['epratio'], sm.add_constant(data_w_dummies[['operatingmargin']]), missing='drop').fit()
result.summary()
result = sm.OLS(data_w_dummies['epratio'], sm.add_constant(data_w_dummies[['operatingmargin', 'Agriculture Forestry And Fishing', 'Construction','Finance Insurance And Real Estate', 'Manufacturing', 'Mining','Retail Trade', 'Services','Transportation Communications Electric Gas And Sanitary Service']]), missing='drop').fit()
result.summary()
data_w_dummies['lnoperatingmargin'] = np.log(data_w_dummies['operatingmargin'])
result = sm.OLS(data_w_dummies['epratio'], sm.add_constant(data_w_dummies[['lnoperatingmargin']]), missing='drop').fit()
result.summary()
data_w_dummies['lnepratio'] = np.log(data_w_dummies['epratio'])
result = sm.OLS(data_w_dummies['lnepratio'], sm.add_constant(data_w_dummies[['lnoperatingmargin']]), missing='drop').fit()
result.summary()

#with dummy variables
result = sm.OLS(data_w_dummies['lnepratio'], sm.add_constant(data_w_dummies[['lnoperatingmargin', 'Agriculture Forestry And Fishing', 'Construction','Finance Insurance And Real Estate', 'Manufacturing', 'Mining','Retail Trade', 'Services','Transportation Communications Electric Gas And Sanitary Service']]), missing='drop').fit()
result.summary()

#clustering standard variables
data_w_dummies.dropna(subset = ['lnepratio', 'lnoperatingmargin'], inplace=True)	#because of a bug in python where fillna is not working perfectly
#note that we can cannot cluster by str variables in python, hence using siccode instead of sicsector
result = sm.OLS(data_w_dummies['lnepratio'], sm.add_constant(data_w_dummies[['lnoperatingmargin', 'Agriculture Forestry And Fishing', 'Construction','Finance Insurance And Real Estate', 'Manufacturing', 'Mining','Retail Trade', 'Services','Transportation Communications Electric Gas And Sanitary Service']]), missing='drop').fit(cov_type='cluster', cov_kwds={'groups': data_w_dummies['siccode']})
result.summary()

#generate categorical variable for probit/logit analysis
data_w_dummies['paydividend'] = data_w_dummies['dps']>0
data_w_dummies['paydividend']	#purely for describing data
data_w_dummies['paydividend'].mean()	#purely for describing data
data_w_dummies['paydividend'] = data_w_dummies['paydividend'].astype(int)	#formatting the data for estimation models

#try using OLS anyway
result = sm.OLS(data_w_dummies['paydividend'], sm.add_constant(data_w_dummies[['lnoperatingmargin', 'Agriculture Forestry And Fishing', 'Construction','Finance Insurance And Real Estate', 'Manufacturing', 'Mining','Retail Trade', 'Services','Transportation Communications Electric Gas And Sanitary Service']]), missing='drop').fit(cov_type='cluster', cov_kwds={'groups': data_w_dummies['siccode']})
result.summary()

#use logit instead
result = smdiscrete.Logit(data_w_dummies['paydividend'], sm.add_constant(data_w_dummies[['lnoperatingmargin', 'Agriculture Forestry And Fishing', 'Construction','Finance Insurance And Real Estate', 'Manufacturing', 'Mining','Retail Trade', 'Services','Transportation Communications Electric Gas And Sanitary Service']]), missing='drop').fit(cov_type='cluster', cov_kwds={'groups': data_w_dummies['siccode']})
result.summary()


###PCA#####
from sklearn.decomposition import PCA
data = data_w_dummies  #renaming the variable for easier typing
numerator = ['cashneq', 'debt', 'ebit', 'ebt', 'eps', 'equity', 'fcf', 'gp', 'inventory', 'liabilities', 'payables', 'receivables', 'tangibles', 'workingcapital']
denominator = ['assets', 'revenue']
featureslist = []
for n in numerator:
    for d in denominator:
        tag = n+'_'+d
        data[tag] = np.log(data[n]/data[d])
        featureslist.append(tag)

data.dropna(subset=featureslist, inplace=True)
features = data.loc[:, featureslist].values
from sklearn.preprocessing import StandardScaler
features = StandardScaler().fit_transform(features)
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
principal_components = pca.fit_transform(features)
principal_components
pca.explained_variance_ratio_
pc_df = pd.DataFrame(principal_components)
pc_df.corr()
pc_df.columns = ['PC1', 'PC2', 'PC3', 'PC4']
pc_df.index = data.index
data_merge = pd.concat([data, pc_df], axis=1)
result = sm.OLS(data_merge['lnepratio'], sm.add_constant(data_merge[featureslist]), missing='drop').fit(cov_type='cluster', cov_kwds={'groups': data_w_dummies['siccode']})
print(result.summary())
result = sm.OLS(data_merge['lnepratio'], sm.add_constant(data_merge[['PC1', 'PC2', 'PC3', 'PC4']]), missing='drop').fit(cov_type='cluster', cov_kwds={'groups': data_w_dummies['siccode']})
print(result.summary())

#lets figure out what 'PC1' means
featureslist.append('PC1')
data_merge[featureslist].corr()


#####INTERACTION VARIABLES###
dummies = industrydummies.columns.values
dummies = dummies[1:len(dummies)-1]
dlist = []
for ind in dummies:
    tag = 'lnoperatingmargin'+'_'+ind
    data[tag] = data['lnoperatingmargin']*(data[ind])
    dlist.append(tag)
    dlist.append(ind)
dlist.append('lnoperatingmargin')
result = sm.OLS(data['lnepratio'], sm.add_constant(data[dlist]), missing='drop').fit(cov_type='cluster', cov_kwds={'groups': data_w_dummies['siccode']})
result.summary()



######SQUARED VARIABLES######
#We hypothesize that a stock's valuation has a concave relationship to its asset tangiblility - why?
data['tangibles_assets_normal'] = data['tangibles']/data['assets']
data['tangibles_assets_sq'] = data['tangibles_assets_normal']*data['tangibles_assets_normal']
result = sm.OLS(data['epratio'], sm.add_constant(data[['tangibles_assets_normal', 'tangibles_assets_sq']]), missing='drop').fit(cov_type='cluster', cov_kwds={'groups': data_w_dummies['siccode']})
print(result.summary())
result = sm.OLS(data['epratio'], sm.add_constant(data[['tangibles_assets_normal']]), missing='drop').fit(cov_type='cluster', cov_kwds={'groups': data_w_dummies['siccode']})
print(result.summary())

##LAG INDEPENDENT VARIABLES
data_merge.index=[data_merge['ticker'], data_merge.index]
datalag1 = data_merge.groupby(level=0).shift(1)
dataset = pd.concat([data_merge['lnepratio'], datalag1[['PC1', 'PC2', 'PC3', 'PC4']]], axis=1)
dataset.dropna(inplace=True)
result = sm.OLS(dataset['lnepratio'], sm.add_constant(dataset[['PC1', 'PC2', 'PC3', 'PC4']]), missing='drop').fit()
print(result.summary())

#F-TESTS
from statsmodels.formula.api import ols
formula = 'lnepratio ~ tangibles_assets_normal + tangibles_assets_sq + cashneq_assets + gp_assets + gp_revenue + fcf_assets'
result = ols(formula, data).fit()
hypothesis1 = '(tangibles_assets_normal = cashneq_assets), tangibles_assets_sq = -0.5'
f_test = result.f_test(hypothesis1)
print(f_test)
hypothesis2 = '(gp_assets = gp_revenue)'
f_Test2 = result.f_test(hypothesis2)
print(f_Test2)

#deliberately get an insignificant F-test result so we can see what that looks like
formula = 'epratio ~ tangibles_assets_normal + tangibles_assets_sq + cashneq_assets + gp_assets + gp_revenue + fcf_assets'
result = ols(formula, data).fit()
f_Test2 = result.f_test(hypothesis2)
print(f_Test2)


#Durbin Watson and other diagnostics
#we are just repeating these estimations
result = sm.OLS(data_w_dummies['epratio'], sm.add_constant(data_w_dummies[['operatingmargin', 'Agriculture Forestry And Fishing', 'Construction','Finance Insurance And Real Estate', 'Manufacturing', 'Mining','Retail Trade', 'Services','Transportation Communications Electric Gas And Sanitary Service']]), missing='drop').fit()
result.summary()
result = sm.OLS(data_w_dummies['lnepratio'], sm.add_constant(data_w_dummies[['lnoperatingmargin', 'Agriculture Forestry And Fishing', 'Construction','Finance Insurance And Real Estate', 'Manufacturing', 'Mining','Retail Trade', 'Services','Transportation Communications Electric Gas And Sanitary Service']]), missing='drop').fit()
result.summary()


#BRESUCH-GODFREY
result = sm.OLS(data_w_dummies['lnepratio'], sm.add_constant(data_w_dummies[['lnoperatingmargin', 'Agriculture Forestry And Fishing', 'Construction','Finance Insurance And Real Estate', 'Manufacturing', 'Mining','Retail Trade', 'Services','Transportation Communications Electric Gas And Sanitary Service']]), missing='drop').fit()
result.summary()
from statsmodels.stats.diagnostic import acorr_breusch_godfrey as bg
bg(result)
bg(result, nlags=5)


#GOLDFIELD-QUANDT test
from statsmodels.compat import lzip
result = sm.OLS(data_w_dummies['lnepratio'], sm.add_constant(data_w_dummies[['lnoperatingmargin', 'Agriculture Forestry And Fishing', 'Construction','Finance Insurance And Real Estate', 'Manufacturing', 'Mining','Retail Trade', 'Services','Transportation Communications Electric Gas And Sanitary Service']]), missing='drop').fit()
result.summary()
GQ = sms.het_goldfeldquandt(result.resid, result.model.exog)
lzip(['Fstat', 'pval'], GQ)

#WHITE test
from statsmodels.stats.diagnostic import het_white
result = sm.OLS(data_w_dummies['lnepratio'], sm.add_constant(data_w_dummies[['lnoperatingmargin', 'Agriculture Forestry And Fishing', 'Construction','Finance Insurance And Real Estate', 'Manufacturing', 'Mining','Retail Trade', 'Services','Transportation Communications Electric Gas And Sanitary Service']]), missing='drop').fit()
result.summary()
wtest = het_white(result.resid, result.model.exog)
labels = ['Lagrange Multiplier statistic:', 'LM test\'s p-value:', 'F-statistic:', 'F-test\'s p-value:']
lzip(labels, wtest)

#Effect of additional logarithm filter on heterogeneity White's test results
data_w_dummies['lnlnepratio'] = np.log(data_w_dummies['lnepratio'] + 1)
data_w_dummies['lnlnoperatingmargin'] = np.log(data_w_dummies['lnoperatingmargin'] + 1)
result = sm.OLS(data_w_dummies['lnlnepratio'], sm.add_constant(data_w_dummies[['lnlnoperatingmargin', 'Agriculture Forestry And Fishing', 'Construction','Finance Insurance And Real Estate', 'Manufacturing', 'Mining','Retail Trade', 'Services','Transportation Communications Electric Gas And Sanitary Service']]), missing='drop').fit()
result.summary()
wtest = het_white(result.resid, result.model.exog)
labels = ['Lagrange Multiplier statistic:', 'LM test\'s p-value:', 'F-statistic:', 'F-test\'s p-value:']
lzip(labels, wtest)


#Ramsey's RESET test
from statsmodels.stats.diagnostic import linear_reset as lr
result = sm.OLS(data_w_dummies['lnepratio'], sm.add_constant(data_w_dummies[['lnoperatingmargin', 'Agriculture Forestry And Fishing', 'Construction','Finance Insurance And Real Estate', 'Manufacturing', 'Mining','Retail Trade', 'Services','Transportation Communications Electric Gas And Sanitary Service']]), missing='drop').fit()
lr(result, power = 3)

#Try to improve result on RESET test
data_w_dummies['lnom_sq'] = data_w_dummies['lnoperatingmargin']**2
data_w_dummies['lnom_cu'] = data_w_dummies['lnoperatingmargin']**3
result = sm.OLS(data_w_dummies['lnepratio'], sm.add_constant(data_w_dummies[['lnoperatingmargin', 'lnom_sq', 'lnom_cu', 'Agriculture Forestry And Fishing', 'Construction','Finance Insurance And Real Estate', 'Manufacturing', 'Mining','Retail Trade', 'Services','Transportation Communications Electric Gas And Sanitary Service']]), missing='drop').fit()
lr(result, power = 3)




