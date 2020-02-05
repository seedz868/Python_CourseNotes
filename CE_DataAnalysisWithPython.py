
# change the name of the index for drive_wheels_count 
drive_wheels_counts.index.name = 'drive-wheels'

	              value_counts
drive-wheels	
fwd	            118
rwd	            75
4wd	            8


Week 3 

  Descriptive Stats

df.describe()  
df.describe(include=['object'])  
df['drive-wheels'].value_counts() # not df[['']]
df['drive-wheels'].value_counts().to_frame


  
column1_counts=df['column1'].values_counts()
column1_counts.rename(columns={'name_before':'name_after'}, inplace=True)
column1_counts.index.name = 'name_before

    Box Plots

sns.boxplot(x='col1', y='col2', data=df)

y=df['col2']
x=df['col1']
plt.scatter(x,y) 

plt.title('my Title')
plt.xlabel('ddd')
plt.ylabel('fff')


  Grouping data
  
df_A = df[['col1_nominal','col2_nominal','price']]
df_B = df_A.groupby(['col1_nominal','col2_nominal'], as_index=False ).mean()

grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.get_group('4wd')['price'] # obtain the values of the method group



  Pivot Tables
  
makes it more readable

df_pivot = df_B.pivot(index= 'col1_nominal', columns='col1_nomina2')
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0

Heat Maps

plt.pcolor(df_pivot, cmap='RdBBu')
plt.colorbar()
plt.show()

  Correlation

sns.regplot(x='', y='', data=df)
plt.ylim(0,) # lower limit for y axis

#eveything as correlated to price, sorted:
df.corr()['price'].sort_values()

Pearson
  scipy?
pearson_coef, p_value = stats.pearsonr[df?['horsepower'],df['price']]

  ANOVA analysis of variance

df_anova=df[['make','price']]
grouped_anova=df_anova.groupby(['make'])

anova_results = stats.f_oneway
  (
    grouped_anova.get_group('honda')['price'] , 
    grouped_anova.get_group('subaru')['price']
  )

f_val, p_val = stats.f_oneway
  (
    grouped_test2.get_group('fwd')['price'], 
    grouped_test2.get_group('rwd')['price'], 
    grouped_test2.get_group('4wd')['price']
  )
        
Week 4 - 

	Model Development

from sklearn.linear_model import LinearRegression

lm=LinearRegression()
X = df[['highway-mpg']]
Y = df['price']

lm.fit(X,Y)
Yhat = lm.predict(xnew)
lm.intercept_
lm.coef_

Z= df[['a','c','d','s']]
lm.fit(Z, df['price'])
#To obtain a prediction Yhat from some value x1
Yhat = lm.predict(Xnew)


	Model Evaluation
	
regression plot

import seaborn as sns

----
width = 12
height = 10
plt.figure(figsize=(width, height))
----
sns.regplot(x="aaa", y="price", data=df)
plt.ylim(0,)

residual plot

import seabord as sns

sns.residplot(df['aasd'],df['price'])

Distribution Plot - predicted value vs actual value

import seaborn as sns
ax1 = sns.distplot(df["price"], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values", ax=ax1)

	Polynpomial Regression	
transform data into a polynomial

	
f=np.polyfit(x,y,3)
p=np.poly1d(f) # to get the actual expression

from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
x_polly=pr.fit_transform(x[['aaa','ddd']], include_bias=False) # transform features

pr=PolynomialFeatures(degree=2)
pr.fit_transform([1,2], include_bias=False)

	Pre Processing
from sklearn.preprocessing import StandardScalar
SCALE=StandardScalar()
SCALE.fit(x_data[['horsepower','highwayMPG']])
x_scale=SCALE.transform(x_data[['horsepower','highwayMPG']])

	Pipelines
Steps to getting a prediction:
	Normzn > Poly transform > Linear Reg

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScalar
from sklearn.pipeline import Pipeline

a LIST of TUPLES

Input = [
	 ('scale',StandardScaler()), 
	 ('polynomial',PolynomialFeatures(degree=2)),
	 ('model',LinearRegression())
	]

pipe=Pipeline(Input)
pipe.train(X['horsepower','curbweight','enginesize','highway'], y)
?pipe.fit(X['horsepower','curbweight','enginesize','highway'], y)
#this trains the model

yhat = pipe.predict(X[['horsepower','curbweight','enginesize','highway']])
#get a prediction

	Mean Sq Error
from sklearn.metrics import mean_squared_error
mean_sq_error(df['price'], Y_predict_simple_fit)

	coeff of determination
Rsqared = 1- (MSE of fit regression line / MSE of average of data )

X=df[['highwayMPG']]
Y=df['price']
lm.fit(X,Y)
lm.score(X,Y) == Rsquared (how much of the price is explained by the model)

	Preiction and decision making
import numpy as np

new_input=np.arange(1,101,1).reshape(-1,1)

Week 5
	Model evaluation and refinement
	
Train Test Split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, =  train_test_split(x_data, y_data, test_size=0.3, random_state=0)

Cross Validation 

from sklearn.model_selection import cross_val_score
scores=cross_val_score(lr,x_data,y_data,cv=3)
lr is a linear regression model = outputs 3 Rsquared values

Cross val PREDICT

from sklearn.model_selection import cross_val_prediict
yhat=cross_val_predict(lr2e, x_data, y_data, cv=3)
putputs the actua lpredict values for each Rsquared



	Overfitting

rSQU_test=[]
order = [1,2,3,4]
for n in order:
	pr=PolynomialFeatures(degree=n)
	x_train_pr=pr.fit_transform(x_train[['horsepower']])
	x_test_pr =pr.fit_transform(x_test[['horsepower'v ]])
	
	lr.fit(x_train_pr,y_train)
	
	rSQU_test.append(lr.score(x_test_pr, y_test))
	

	Ridge Regression
from sklearn.linear_model import Ridge

Rigemodel=Ridge(alpha=0.1)
Rigemodel.fit(X,y)

Yhat = Rigemodel.predict(X)

	Grid Search
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

parameters1=[{'alpha':[0.001,0.01,0.1,1.0,10,100,1000]}]

RR = Ridge()
Grid1 = GridSearchCV(RR, parameters1,cv=4)

Grid1.fit(x_data[['horsepower','curbweight','engineSize','highwayMPG']], y_data)

Grid1.best_estimator_

scores = Grid1.cv_results_
score['mean_test_score']




