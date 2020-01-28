
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
X=df[[]]
Y=df[[]]

lm.fit(X,Y)
Yhat = lm.predict(xnew)
lm.intercept_
lm.coef_

Z= df[['a','c','d','s']]
lm.fit(Z, df['price'])
Yhat = lm.predict(Xnew)

	Model Evaluation
	
regression plot

import seaborn as sns

sns.regplot(x="aaa", y="price", data=df)
plt.ylim(0,)

residual plot

import seabord as sns

sns.residplot(df['aasd'],df['eee'])









