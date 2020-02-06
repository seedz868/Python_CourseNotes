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
 
