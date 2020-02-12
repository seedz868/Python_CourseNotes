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

