

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
df_B = df_A.groupby(  ).mean()
