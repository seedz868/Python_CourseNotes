
# Python Classes and Inheritance
# https://www.coursera.org/learn/python-classes-inheritance/home/week/1

class Point():
    pass

point1 = Point()
point2 = Point()

point1.x = 5
point2.x = 10


print (point1.x)
print (point2.x)


a method is like a function that belongs to a class
Methods ALWAYS have at least 1 argument, self


class Point():
    def getX(self):
        return self.x

point1 = Point()
point2 = Point()

point1.x = 5
point2.x = 10

print(point1.getX())



class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

point1 = Point(5,10)
point2 = Point(1,2)


class Point:

    def __init_(self, initX, initY):
        self.x = initX
        self.y = inteY

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distanceFromOrigin(self):
        return ((self.x**2) + (self.y**2)) ** 0.5

    def __str__(self):
        return 'Point ({}, {})'.format(self.x, self.y)

p = Point(5,6)
print (p)
p2 = Point(6,7)
print (p2)

class Point:
    ...

    def __add__(self, otherPoint):
        return Point(self.x + otherPoint.x, self.y + otherPoint.y)

    def __sub__
        return Point(self.x - otherPoint.x, self.y - otherPoint.y)

print (p1 + p2)
print (p1 - p2)





class Point:

    def __init_(self, initX, initY):
        self.x = initX
        self.y = inteY

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distanceFromOrigin(self):
        return ((self.x**2) + (self.y**2)) ** 0.5

    def __str__(self):
        return 'Point ({}, {})'.format(self.x, self.y)

    def halfwaymid(self, target):
        mx = (self.x + target.x) / 2
        my = (self.y + target.y) / 2
        return Point(mx,my)


mid = p.halfway(q)
print (mid)

print (mid.getX())
print (mid.getY())



class Fruit():
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def sort_priority(self):
        return self.price


L = [ Fruit('adasd',10),
Fruit('eere',20),
Fruit('ccddfdf',32)
]

for f in sorted(L, key = Fruit.sort_priority):
for f in sorted(L, key = lambda x: x.sort_priority()):

    print (f.name)

print()


Inheriting

CURRENT_YUEAR=2019
class Persiopn:
    def __init__(self, name, year_born):
        self.nam,e = name
        self.year_born = year_born
    def getAge(self):
        return CURRENT_YEAR - self.year_born
    def __str__(self):
        return '{} ({})'.format(self.name, self.getAge())


class Student:
    def __init__(self, name, year_born):
#        self.name = name
#        self.year_boirn =- yuear_born
        Person.__init__(self, name, year_)born)
        self.nowledge =- 0
    def study)self):
        self.knowledge +=1


###### Inheritance

class Book:
    def __init__(self, title, authjor):
        self.title = title
        self.author = author

    def __str__(self):
        return '"{}" by {}'.format(self,title, self.author)

class PaperBook(Book):
    def __init__(self, title, author numPages):
            Book.__init__(self, title, author)
            self.numPages = numPages

class EBook(Book):
    def __init__(self, title, author,  size):
        Book.__init__(self, title, author)
        self.size = size

class Library:
    def __init__(self):
        self.books = []
    def addBook(self, book):
        self.
    def getNumBooks(self):
        return len(self.book)



myBook = EBook('The odyys', 'Homer', 2)
myPaperBook = PaperBook('The Odyssy','Homer', 500)

aadl = Librar()
aadl.addBook(myBook)
aadl.addBook(mtyPaperBook)


print (aadl,getNumBooks())



class Dog(Pet):
    sounds = []

    def feed(self):
        Pet.feed(self)  #callls part class feed first, then adds a modifier
        print ("arf")


# testing

p = Point(3,4)
p.move(-2,3)

test.testEqual(p.x,1)
test.testEqual(p.y,7)


test.testEqual(square(3),9)

### Exceotion handling

Syntactic
Runtime
Semantic

items = ['a','b']

try:
    print ('a')      # runs
    third = items[2] # breaks code block
    print ('b')      # does NOT run
except:
    print ('something went wrong') #runs
    third = False                  #runs
print ('I want this to run')       #runs


if else IF you know what might go wrong
try except IF u dont know and want to be more general


try:
    x = 10/0
    th = item[34]
    print ('a')
except ZeroDivisionError:
    print ()
except IndexError:
    print ()
except NameError:
    print ()
