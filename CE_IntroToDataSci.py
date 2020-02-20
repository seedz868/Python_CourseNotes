Week1 


sales_record = {
'price': 3.24,
'num_items': 4,
'person': 'Chris'}

sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'

print(sales_statement.format(sales_record['person'],
                             sales_record['num_items'],
                             sales_record['price'],
                             sales_record['num_items']*sales_record['price']))

with open('mpg.csv') as csvfile:
    mpg = list(csv.DictReader(csvfile))
    
mpg[:3] # The first three dictionaries in our list.
sum(float(d['cty']) for d in mpg) / len(mpg)

# Use `set` to return the unique values for the number of cylinders
cylinders = set(d['cyl'] for d in mpg)
cylinders
    {'4', '5', '6', '8'}
 
---
store1 = [10.00, 11.00, 12.34, 2.34, 0.11]
store2 = [9.00, 11.10, 12.34, 2.01, -0.11]
cheapest = map(min, store1, store2)
cheapest
    <map at 0x7f04a37514e0>
for item in cheapest:
  print(item)
    9.0
    11.0
    12.34
    2.01
    -0.11
    
my_function = lambda a, b, c : a + b
my_function(1, 2, 3)
  3

  List comprehension

my_list = [number for number in range(0,1000) if number % 2 == 0]
my_list
  [0,
   2,
   4,
   6,
   8,
   10,
   12,
   14...]
 
  NumPy
np.array
np.arange
VOI.reshape
np.linspace
VOI.resize
np.ones
np.zeros
np.eye
np.diag
np.repeat
np.vstack
np.hstack
VIO.argmax()
VOI.argmin()

# start @ 5th element from the end, count backwards by 2 until @ beginning.
s[-5::-2]

# Complete copy to memory
r_copy = r.copy()
r_copy

# Use `zip` to iterate over multiple iterables.
for i, j in zip(test, test2):
    print(i,'+',j,'=',i+j)

Week 2

pd.series(VOI_list)

import numpy as np
np.nan == None
np.nan == np.nan
np.isnan(np.nan)

s = pd.Series({'Archery': 'Bhutan', 'Golf': 'Scotland', 'Sumo': 'Japan', 'Taekwondo': 'South Korea'})
s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])

------------------------
Archery           Bhutan
Golf            Scotland
Sumo               Japan
Taekwondo    South Korea
------------------------
dtype: object

> s.iloc[3]  / s[3]
'South Korea'
> s.loc['Golf'] / s['Golf']
'Scotland'

#this creates a big series of random numbers
s = pd.Series(np.random.randint(0,1000,10000))

-----

original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(
                                    ['Australia','Barbados','Pakistan','England'], 
                              index=['Cricket','Cricket','Cricket','Cricket']
                                    )

RENAME
  df.rename(columns={col:'Gold' + col[4:]}, inplace=True)

WHERE STATEMENT
  only_gold = df.where(df['Gold'] > 0)
 
len(df[
    (df['Gold'] > 0) 
    | 
    (df['Gold.1'] > 0)
    ])

df[
    (df['Gold.1'] > 0) 
    & 
    (df['Gold']  == 0)
  ]

INDEXING a DATAFRAME

df['country'] = df.index # make a series that is your index, to be safe
df = df.set_index('Gold') # set the index to the Gold column
df = df.reset_index()
df = df.set_index(['STNAME', 'CTYNAME'])
df.loc[[ # looking up 2 values with this new set of indices
      ('Michigan', 'Washtenaw County'),      
      ('Michigan', 'Wayne County')
      ]]
df = df.set_index('time')
df = df.sort_index()
df = df.fillna(method='ffill')

df['SUMLEV'].unique()

MISSING VALUES

df = df.set_index('time')
df = df.sort_index()
df = df.reset_index()
df = df.set_index(['time', 'user'])
df = df.fillna(method='ffill')


Week 3

Merge data
#make inital data frame
df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])
# add 3 columns
df['Date'] = ['December 1', 'January 1', 'mid-May']
df['Delivered'] = True
df['Feedback'] = ['Positive', None, 'Negative']

	     |Cost	Item Purchased	Name	Date	      Delivered	Feedback
------------------------------------------------------------------
Store 1|22.5	Sponge	        Chris	December 1	True	    Positive
Store 1|2.5	  Kitty Litter	  Kevyn	January 1	  True	    None
Store 2|5.0	  Spoon	          Filip	mid-May	    True	    Negative

adf = df.reset_index()
adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'})
#updating the date column, but specifying which to update and how
#Store number is called index cuz it WAS the index
 |index	  Cost	Item Purchased	Name	Date	      Delivered	Feedback
--------------------------------------------------------------------
0|Store 1	22.5	Sponge	        Chris	December 1	True	    Positive
1|Store 1	2.5	  Kitty Litter	  Kevyn	NaN	        True	    None
2|Store 2	5.0	  Spoon	          Filip	mid-May	    True	    Negative

staff_df = staff_df.set_index('Name')
student_df = student_df.set_index('Name')

pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)
                                   'inner'
                                   'left'
                                   'right'
pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')

Idiomatic

(df.where(df['SUMLEV']==50)
   .dropna()
   .set_index(['STNAME','CTYNAME'])
   .rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'}))
# Bracket allow you to write code in multiple lines

# Applying a function to some or all the rows of a data frame,
import numpy as np
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    return pd.Series({'min': np.min(data), 'max': np.max(data)})
	# This returns only the new cols, plus indices
# OR #  This returns the original DF, plus the new calculations
    row['max'] = np.max(data)
    row['min'] = np.min(data)
    return row

df.apply(min_max, axis=1) # normall a 0 for rows but a 1 here
# OR #
rows = ['POPESTIMATE2010',
        'POPESTIMATE2011',
        'POPESTIMATE2012',
        'POPESTIMATE2013',
        'POPESTIMATE2014',
        'POPESTIMATE2015']
df.apply(lambda x: np.max(x[rows]), axis=1)
# similar to first example


Group By
for group, frame in df.groupby('STNAME'): # kinda like enumerate
    avg = np.average(frame['CENSUS2010POP'])
    print('Counties in state ' + group + ' have an average population of ' + str(avg))

# prolly not useful but 

df = df.set_index('STNAME')
def fun(item): # break up depending on 1st letter of STNAME
    if item[0]<'M':
        return 0
    if item[0]<'Q':
        return 1
    return 2

for group, frame in df.groupby(fun): # pass in a function
    print('There are ' + str(len(frame)) + ' records in group ' + str(group) + ' for processing.')

# running average or sum over some cols.
(df.groupby('STNAME')
   .agg({'CENSUS2010POP': np.average }
)
(df.set_index('STNAME')
   .groupby(level=0)['CENSUS2010POP']  # not sure what level=0 means
   .agg({'avg': np.average, 'sum': np.sum})
)
# all the calculation over all the cols
(df.set_index('STNAME')
   .groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
   .agg({'avg': np.average, 'sum': np.sum})
)
#specific calculation over specific cols
(df.set_index('STNAME')
   .groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
   .agg({'POPESTIMATE2010': np.average, 'POPESTIMATE2011': np.sum})
)

 
 
Scales 


Pivot tables






