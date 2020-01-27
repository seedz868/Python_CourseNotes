

Week 3 

  Descriptive Stats

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
  
df_A = df['col1_nominal','col2_nominal','price']
df_B = df_A.groupby(['col1_nominal','col2_nominal'], as_index=False ).mean()

Pivot method makes it more readable

df_pivot = df_B.pivot(index= 'col1_nominal', columns='col1_nomina2')

Heat Maps

plt.pcolor(df_pivot, cmap='RdBBu')
plt.colorbar()
plt.show()

  Correlation

sns.regplot(x='', y='', data=df)
plt.ylim(0,)

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


