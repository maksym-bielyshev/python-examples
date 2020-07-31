

import math
list = range(5)
def root(x):
    return math.sqrt(x)
for y in list:
    if root(y) % 2 == 0:
        print(y)
# 0 4

def outer_func(msg):
    def inner_func():
        print(f"message is {msg}")
    return inner_func
var = outer_func("hello world!")
var()
# message is hello world!

a = (8, -1, 3)
a.sort()
print(a[1])
# AttributeError: 'tuple' object has no attribute 'sort'

def func(x):
    x[0] = ['def']
    x[1] = ['abc']
    return id(x)
q = ['abc', 'def']
print(id(q)==func(q))
# True

a = "eggs"
b = a
a = "spam"
print(b)
# eggs

def func1(x):
    return x + 1
def func2(x, y = 2):
    return x + y
arr = [func1, func2]
n = 0
for f in arr:
    n += f(n)
print(n)
# 4

lst = [].append(5)
print(lst)
# None

def square_of_x(x,y):
    return x*2
print(square_of_x(y=2, x=3))
# 6

a={0:1, 1:2}
_sum = 0
for b in a: _sum += b
print(_sum + b)
# 2

b = 3
a = 1
for i in range(b):
    a = a + a * i
print(a)
# 6

def test():
    print("x")
    return test

test()()()()
# xxxx

def bar():
    x = 1
    def foo():
        print('x' in globals(), 'x' in locals())
    foo()
bar()
# False False

nums = range(5)
a = 1
for i in nums[1:]:
    a *= i
print(a)
# 24

a = [1,2,3]
x,y,z = a
print(x*z+4%3)
# 4

d = {'s':'o', 'l':'o', 'l': 'e', 'a': 'r', 'n':'!'}
print(len(d))
# 4

a = [1,2,3]
b = a
a = a+[4]
print(len(b))
# 3

a = 6
b = 4
if((b%a)==1):
    print(a)
else:
    print(b)
# 4

arr = [0,1,1,0]
for val in arr:
    if val == 0:
        arr[val] = 1
print(arr)
# [1, 1, 1, 0]

a = 1 % 213
b = 1 % 215
print(a == abs(b))
# True

import numpy as np
a = np.array([[1,2], [3,4], [5,6]])
print(a[:,1])
# [2 4 6]

try:
    s = ['a', 'c']
    n = [0, 2]
    s[1:1] = 'b'
    n[1:1] = 1
except:
    pass
print(s[1], n[1])
# b 2

def myAdd(x = 2, y = 4, *args):
    while (x < y):
        for ar in range(len(args)):
            return (x + y + args[ar + 2])
            break
print(myAdd(3,4,5,6,9))
# 16

class MyClass:
    def __init__(self, n):
        self.x = n
    def set_y(self):
        self.y = self.x * 2
obj = MyClass(2)
print(obj.x + obj.y)
# AttributeError: 'MyClass' object has no attribute 'y'

import numpy as np
ar = np.ones((3,2))
print(ar.ndim)
# 2

def factorial(x):
    w = 1
    for i in range(1, x + 1):
        w = w * i
    return w
print(factorial(5))
# 120

a = [8, 0, 4, 6, 1]
m = len(a) // 2
b = a[:m] + a[m:]
print(a == b)
# True

a = "abcd"
b = "abc"
func = (lambda s:s[1:]) or (lambda s:s[:-1])
print((func(b)))
# bc

if 1 & 0:
    print('a')
else:
    print('b')
# b

func = lambda x: x%3 == 0
l = filter(func, range(10))
l = list(l)
print(len(l)+sum(l)+max(l))
# 31

a = [0, 1, 2, 3, 4, 5, 6]
b = a[::-3]
b.sort()
if (b == a[::3]):
    print(1)
else:
    print(0)
# 1

x = [2, 3, 4]
y = x
print(y is x)
# True

print(abs(3 + 4j))
# 5.0

a = [1, 2, ":", 3]
del a[:]
print(a)
# []

A = [1,2]
B = [3,4]
C = A.extend(B)
print(C[3])
# TypeError: 'NoneType' object is not subscriptable

a = 1
b = a + 1
a, b = 2, a + 1
print(a, b)
# 2 2

def guess(list1):
    list2 = list1
    list2.append(4)
    list1 = [5, 6, 7]


list1 = [1, 2, 3]
guess(list1)
print(sum(list1))
# 10

class Myclass:
    def __init__(self):
        self.n = 0
a = Myclass()
b = a
b.n = 1
print(a.n)
# 1

a = [1,2]
a.append(a)
print(a != a[2][2][2])
# False

f = [1, 2, 4, 6, 8]
print(str(f[2] -f[4] % 3) * 3)
# 222

a = 148
b = a * a + a
print(b % a < b / a)
# True

arr = [0, 1, 2, 3, 4, 5]
sum = 0
for n in arr:
    sum = sum + n
    if (n == 3):
        arr.insert(n, 10)
    if (sum > 15):
        break
print(sum)
# 18

list = [1, 2, 3]
list2 = [4, 5, 6]
"""
for i in list2:
    list.append(i)
"""
print(len(list))
# 3

x = 10
def addTwo(x):
    return x + 2
n = addTwo(4)
print(x)
# 10

import re
pattern = r"(?P<le>Solo)((le)arn)"
match = re.match(pattern, "Sololearn")
if match:
    print(match.group("le"))
else:
    print("Sololearn")
# Solo

list = [0, 1, 2, 3, 4, 5]
print(list[:1:-1][1])
# 4

class FirstClass:
    n = 4
    def __init__(self, n):
        n = n//4
a = FirstClass(8)
print(a.n)
# 4

a = [1,2,3,5,8,13]
print(a[2:2])
# []

import numpy as np
arr = np.zeros((2,4))
print(arr.size)
# >>> 8

def f(q, mylist=[]):
    mylist = mylist + q
    return mylist
a = [1]
print(f(a))
a = [2]
print(f(a))
# >>> [1] [2]


def f(m: str, v: str) -> str:
    return f.__annotations__
print(f('a', 'b')['m'])
# >>> <class 'str'>


def fac(c):
    if c > 0:
        return c * fac(c - 1)
    else:
        return 1
print(fac(4))
# >>> 24

x = 2
y = "2"
print(not x == y and y == x)
# >>> False


class A(int):
    pass

print(A(9) + A(9))
# >>> 18

List = []
List.append(1)
List.extend(2)
print(List)
# >>> TypeError: 'int' object is not iterable


def f(x=0):
    y = x

    def g(y):
        print(y)
    print(x)
    return g
f()(1)
# >>> 0 1

a = [3, 4, 2]
b = sorted(a)
b.insert(0, 1)
print(a[0])
# >>> 3

g = 64 // 3
f = type(g) == int
h = ['float', 'str', 'int']
print(f'The type of g is {h[-f]}')
# >>> The type of g is int


def count(n=0):
    while True:
        yield (yield n)
        n += 1
a = count()
next(a)
print(a.send(5), end="")
print(a.send("a"))
# >>> 51

list1 = [1, 2]
list2 = list1
list3 = list2[:]
a = list2 is list1
b = list3 is list1
print(a, b)
# >>> True False

a = 257
b = 257
print(a == b and a is b)
# >>> True

print(int(2 / 2 * 2 - 2 + 2 % 2))
# >>> 0

*x, y = [1, 2], [3, 4], [5, 6]
print(list(zip(*x + [y]))[1][1])
# >>> 4


def function(x):
    if x == 1:
        return x
    else:
        x -= 1
        function(x)
x = 2
print(function(x), x)
# >>> None 2

my_list = [3, 5, 6, 7, 8, 1]
for i in range(3):
    my_list.pop(i)
print(my_list)
# >>> [5, 7, 1]

num = 5
sum = 0
while True:
    if num >= 1:
        sum += num
        num -= 1
    else:
        break
print(sum)
# >>> 15

x = [[0], [1]]
print(len('a'.join(list(map(str, x)))))
# >>> 7

a = [1, 2]
b = {1, 2}
c = (2, 1)
print(a == list(b))
print(tuple(b) == c)
print(b == set(c))
print(tuple(a) == c)
# >>> True False True False

dict = {1: 15, 5: 10, 4: 26}
sum = 0
for value in dict:
    sum = sum + value
print(sum)
# >>> 10

if 1 & 2:
    print(1)
else:
    print(2)
# >>> 2

mixed = ".s:;olq[olw::e/a#@rn"
for char in ".@#:;/[qw":
    mixed = mixed.replace(char, '')
print(mixed)
# >>> sololearn

a = [x - 2 for x in range(-1, 5)]
val = lambda x: x + 3
b = [val(x) for x in a]
print(abs(max(a) - max(b)))
# >>> 3


def f(n, v):
    n = len(v)
    v.append(n)
n = 0
v = [8, 0, 4, 6]
f(n, v)
print(n, sum(v))
# >>> 0 22

name, age = "Jane", 22
print("Her name is {}. She is {}.".format(name, age))
# >>> Her name is Jane. She is 22.

a = {1, 2, 3}
a.add(0)
print(len(a))
# >>> 4

seq = 2
list_1 = [1, 1, 4]
for i in range(1, len(list_1)):
    h = list_1[i] - list_1[i - 1]
    if h != seq:
        list_1[i] = list_1[i - 1] + seq
print(list_1)
# >>> [1, 3, 5]

x = [0, 1, 2, 3, 6]
x.append(x[:])
print(len(x))
# >>> 6


def func(item):
    return len(item)
_list = ['Java', 'Python', 'C', 'R']
print(max(_list, key=func))
# >>> Python


def multiplying(x):
    j = range(1, x + 1)
    return eval('*'.join([str(i) for i in j]))
print(multiplying(3))
# >>> 6

arr = [1, 2, 3, 4, 3, 2, 1]
a = list(map(lambda x: x % 2 == 0, arr))
if a[3] == True:
    print('c')
else:
    print('d')
# >>> c


def foo(x):
    x = ['def', 'abc']
    return id(x)
q = ['def', 'abc']
print(id(q) == foo(q))
# >>> False

arr = [2, 1, "a", 3]
arr.sort()
print(arr)
# >>> TypeError: '<' not supported between instances of 'str' and 'int'

from functools import reduce
numbers = [1, 2, 3, 4]
sum = reduce(lambda x, y: x + y, numbers)
print(sum)
# >>> 10


class A:

    @staticmethod
    def sample(self):
        print('static method')

    @classmethod
    def sample(self):
        print('class method')
A.sample()
# >>> 'class method'

print('python'[::-1].endswith('p'))
# >>> True

t = (n for n in range(5))
print(type(t))
# >>> <class 'generator'>


def f(n=9):
    print(n)
f(5)
# >>> 5


class Myclass:

    def __init__(self, n=1):
        self.__n = n

    def val_n(self):
        return self.__n
obj = Myclass(2)
obj.__n = 3
print(obj.val_n())
# >>> 2

arr = [4, 3, 1, 2]
arr[0], arr[arr[0] - 1] = arr[arr[0] - 1], arr[0]
print(arr)
# >>> [2, 4, 1, 2]

a = [0, 1, 4, 9, 16, 25]
for v in a:
    if not a.index(v) ** 2 != v:
        print("True")
        break
    else:
        print("False")
        break
# Output: True


class A:
    x = [1, 2, 3]


class B:
    y = A()
    z = 2
obj = B()
if obj.y.x.index(obj.z) == 1:
    print(True)
else:
    print(False)
# Output: True

print(int('0011', 2))
# Output: 3

s = "Hello"
print(*s)
# Output: H e l l o

x = True
y = False
z = False
if not x or y:
    print(1)
elif not x or not y and z:
    print(2)
else:
    print(3)
# Output: 3

lst = [34, 67, 34]
if lst.sort() == sorted(lst):
    print('true')
else:
    print('false')
# Output: false

print(5 | 3)
# Output: 7

A = [[]] * 3
A[0].append(3)
print(A)
# Output: [[3], [3], [3]]


class Myclass:

    def __init__(self, val):
        self.__n = val
obj = Myclass(0)
obj.__n = 1
print(obj._Myclass__n)
# Output: 0


def make(a):
    return a + a


def func(a):
    b = '*' * 10
    if make(a) == b:
        print("True")
    else:
        print("False")
a = '*' * 5
func(a)
# Output: True

x = (0, 1, 2)
[a, b, c] = x
print(a + b + c)
# Output: 4

import numpy
arr = numpy.arra([[1, 2, 3], [4, 5, 6]])
arr = arr.reshape(3, 2)
print(arr[1][1])
# Output: 4


def func(n):
    y = '*'.join(str(x) for x in range(1, n, 2))
    return eval(y)

print(func(7))
# Output: 15

a = [0]


def pick_last(deck=a):
    val = deck[-1]
    deck.append(val + 1)
    return val
pick_last()
print(pick_last())
# Output: 1


def func(a, L=[]):
    L.append(a)
    return L
print(func(1), func(3))
# Output: [1, 3] [1, 3]

import bisect
students = [(5, "Ben"), (20, "Sam")]
bisect.insort(students, (25, "Bob"))
bisect.insort(students, (1, "Tom"))
print(students[0][1])
# Output: Tom

a = [0, 1, 2, 3]
for a[-1] in a:
    print(a[-1], end=' ')
# Output: 0 1 2 2


class Parent(object):
    x = 1


class Child1 (Parent):
    pass


class Child2(Parent):
    pass

Child1.x = 2
Parent.x = 3
print(Parent.x)
print(Child1.x)
print(Child2.x)
# Output: 3 2 3


class A:
    x = 1

    def __add__(self, obj):
        if isinstance(obj, A):
            return self.x + obj.x
        return "False"


class B(A):
    x = 2
a = A()
b = B()
print(a + b)
# Output: 3


def f(values, arr=[]):
    for i in values:
        if i % 2 == 0:
            arr.append(i)
    return len(arr)
print(f(range(4)) + f(range(5)))
# Output: 7


def my_func(x, y, z):
    x = y - z
    z = y % x
    y = x + y
    n = x + y**z
    print(n)
my_func(9, 4, 2)
# Output: 3

a = [2, 3, 1, 0]
b = [3, 1, 0, 2]
c = []
for i in a:
    if i == b[i]:
        c.append(i)
    else:
        continue
print(len(c))
# Output: 1

r = ((3 + 3 * 9) // 10 - 4) * 5 + 5
print(r / 1)
# Output: 0.0

a = [1, 2, 3, 4, 5]
for n in a:
    a.remove(n)
print(a)
# Output: [2, 4]

numbers = list(range(5, 20, 2))
value = (numbers[0]) + (numbers[5])
print(value)
# Output: 20


def count_to_5():
    for i in range(1, 6):
        yield(i)
c = count_to_5()
n = 0
for i in c:
    n += 1
for i in c:
    n -= 1
print(n)
# Output: 5

my_dict = {"Bill ": "Gates", "Steve": "Jobs"}
L = []
for i, j in my_dict.items():
    L.append(i)
    L.append(j)
print(len(L))
# Output: 4

x, y, z = 2, 3, 1
print('x:{0},y:{1},z:{2}'.format(z, x, y))
# Output: x:1,y:2,z:3

data = {-1, 1, 2}


def analyze(data):
    myData = data or '0'
    ans = [int(d) for d in myData]
    ans.sort(key=lambda x: (abs(x), -x))
    print(ans[0])
analyze(data)
# Output: 1

d = {x: x**2 for x in range(1, 10, 3)}
print(d[7])
# Output: 49

a = []
a.append(a)
print(a)
# Output: [[...]]

s = 'think'
s = ''.join(sorted(list(s)[:4]))
print(s)
# Output: hint

a = [0, 1, 2, 3]
for a[-1] in a:
    print(a[-1], end=' ')
# Output: 0 1 2 2

col = []
sum = 0
for a in range(10):
    col.append(sum)
    sum += a
print(col[5])
# Output: 10

print('hello' == "hello")
# Output: True

print(False and False or True)
# Output: True

print((True or False) and False)
# Output: False

print(True or False and False)
# Output: True

A = [1, 2, 3, 4, 5, 6, 7]
G = iter(A)
next(G)
for num in G:
    print(num, end=' ')
    next(G)
    next(G)
# Output: 2 5


class Class:
    n = 3

    def __init__(self, n):
        n = n
a = Class(5)
print(a.n)
# Output: 3

a = [2, 1, 2, 4]
a[1:].remove(2)
print(sum(a))
# Output: 9


def f(n):
    if n == 1:
        return '0'
    else:
        return n * f(n - 1)
print(int(f(5) == 120))
# Output: 0

str = 'sololearn' * 2
s = map(len, str.split())
print(sum(s))
# Output: 18

a = 3
b = 5
c = 6
print((a + c) - a * b + b % a)
# Output: -4

a = 0
while a <= 10:
    a = a + 2
    if (a % 4 == 0):
        print(a, end=' ')
# Output: 4 8 12

arr = ([4, 1], [3, 6])
fa = lambda x, y = 0: x + y
for n in arr:
    print(fa(*n), end='')
# Output: 59

x = (0, 1, 2)
[a, b, c] = x
print(a + b + c)
# Output: 3


def fun():
    for x in range(10):
        yield(x)
x = fun()
for a in range(5):
    next(x)
print(next(x))
# Output: 5

x = [1, 2, 3]
nx = [i**2 for i in x if i % 2 == 0]
print(nx)
# Output: [4]

x = [0, [2, 4, 6], [13, 11, 9]]
y = x[2] + x[1] * 2
print(y[x[1][0]])
# Output: 9

a = [1, 2, 3, 4, 5]
s = 0
for i in a[:3]:
    for j in a[3:]:
        s += 1
print(s)
# Output: 6

y = [x for x in range(10) if x // 3 == 2]
print(sum(y))
# Output: 21

print(False == 0, False == None, None or False, None == 0)
# Output: True False False False

x = str(212.33)
print(x[1])
# Output: 1


def func(x):
    x = 1
func(a)
print(a)
# Output: 0

x = []


def af(b):
    x.append(b)
    return x


def ab(f):
    f += '1'
    return f
print(len(af(ab([1, 0]))))
# Output: 1

a = [1, 2, 3, 4, 12, 13]
for i in a:
    if i % 2 == 1:
        a.append(0)
print(len(a))
# Output: 9

print(*[i for i in range(6) if i % 2 == 0])
# Output: 0 2 4

test = []
for i in range(2):
    test.append(100)
    test.append(3)
print(test[1])
# Output: 3

list = ["(", 1, 2]
index = 2
list.insert(index, ",")
print(*list, end=")")
# Output: ( 1, 2 )


def swap(x, y):
    x = x + y
    y = x - y
    x = x - y
    return(x, y)
print(swap(5, 9))
# Output: (9, 5)

a = 5,
b = 6, 7
c = a + b
print(c)
# Output: 5, 6, 7


class malicious(int):

    def __init__(self, value):
        self = value

    def __mul__(self, x):
        return self - x

    def __add__(self, x):
        return self // x
a = malicious(5)
print((a + 2) + a * 2)
# Output: 5

a = list(range(0, 100))


def function(x):
    a.pop(x - 1):  # invalid syntax
        return(a)
for i in range(0, 100):
    n = 101 - i
    function(n)
print(len(a))
# Output: SyntaxError: invalid syntax

x = [[0] * 2] * 2
x[0][0] = 1
print(x[0][0] + x[1][0])
# Output: 2

a = [3, 5, 7, 9]
b = a
a[1] = 8
print(b[1])
# Output: 8

string = "banana"
x = ''.join(set(string[::2]))
y = (lambda z: z * (z + 2))(len(string))
print(x, y, sep='k')
# Output: bnk48

string = "edakrak"
list = range(6)
if(type(list) == list):
    print(string[::2], string[1::2], sep='')
else:
    print(string[::-1])
# Output: karkade

person = {"Name": "Sarah", "Age": "18"}
person["Name"] = "Sia"
person["Age"] = "18+"
if "18+" not in person:
    print("a")
else:
    print("b")
# Output: a


def updateTable():
    table = ['a', 'd']
print('table')
# Output: table


def r(min, max):
    z = min
    y = max
    if z != max:
        pass
    return("%s%i") % (str(z), int(y))
print(r(1, 2))
# Output: 12

x = 1
word = 'python'
y = ''
for i in word:
    y = y + str(x)
    x += 1
print(y)
# Output: 123456

values = [1, 2, 1, 3]
nums = set(values)


def checkit(num):
    if num in nums:
        return True
    else:
        return False
for i in filter(checkit, values):
    print(i, end='')
# Output: 1213

tens = [(20, 60), (10, 40), (20, 30)]
a = sorted(tens)
print(a)
# Output: [(10, 40), (20, 30), (20, 60)]

print(True, True, True == (True, True, True))
# Output: True True False

list = range(5)
print(type(list))
# Output: <class 'range'>

s = []
a = 0
for i in s:
    s.append(0)
print(len(s))
# Output: 0

x = 35
y = 27
z = 8
r = (44 - x / z % y) * (y - y) * (x * y * x - z)
print(int(r))
# Output: 0

import re
sent = "hi, how, are ! you doing"
count = 0
list1 = filter(None, re.split("[,!]+", sent))
for i in list1:
    count += 1
print(count)
# Output: 4

print(2**0, 0**0, 0**1, 2**1)
# Output: 1 1 0 2

a = [1]
b = [1]
print(a is b)
# Output: False

x = 25 % 5
y = 4**2 // 5
print((x + y)**2)
# Output: 9

a = [1, 2, 3]
b = 3 * a
b.remove(2)
print(b.index(2) * b.count(2))
# Output: 6

arr = {0, 1, 2, 3, 4}
x = arr[len(arr) - 1]
y = x % 4
print(y)
# Output: TypeError: 'set' object does not support indexing


class L(list):
    n = 0

    def __init__(self):
        L.n += 1
a = L()
b = L()
try:
    c = a + b
    print(a.n, end="")
    print(c.n)
except:
    print(0)
# Output: 20

print(chr(ord('A')))
# Output: A

a = 'herro solo'
a[2] = 'l'
print(a[2])
# Output: TypeError: 'str' object does not support item assignment

a = 1
b = 3
if(a <= 5 and b < 9):
    if (a == 4):
        b = 20
    a += 1
    b -= 1
if(b <= 20):
    a = -5
print(a)
# Output: -5


def recf(lst, s=0):
    if len(lst) == 0:
        return s
    else:
        s += lst.pop()
        return recf(lst, s)
s = [5 for i in range(10)]
print(recf(s))
# Output: 50

a = list()
b = dict()
c = dict()
for n in range(3):
    a.append(n)
    b[n] = a
    c[n] = list(a)
print(b == c)
# Output: False

listOne = [1, 3, 5, 7, 9]
listTwo = [2, 5, 4, 3, 9]
for i in range(len(listOne)):
    if listOne[i] == listTwo[i]:
        print(listOne[i] * i)
# Output: 36

a = [1, 2, 3, 4, 5]
print(a[-1:None:-2])
# Output: [5, 3, 1]


class Myclass:
    n = 0

    def change(self, val):
        n = val
obj = Myclass()
obj.change(1)
print(obj.n)
# Output: 0

print('y', end="=")
x, y = 7, 9
y *= x
y %= 8
print(y)
# Output: y=7

lst = [34, 67, 34]
if lst.sort() == sorted(lst):
    print('true')
else:
    print('false')
# Output: false


def fun(x):
    if x == 1:
        return 1
    else:
        return x * fun(x - 1)
print(fun(5))
# Output: 120


def meth():
    g = [2, 3, 4, 5, 7]
    for i in g:
        if i % 2 == 1:
            yield i
for i in meth():
    print(i)
    break
# Output: 3

x = 5


def foo():
    print(x)
    pass
foo()
# Output: 5

d = {'A': 3, 'B': 8}
for n in range(2, 12, 2):
    d['A'] += n
    d['B'] -= n
print(d['A'] + d['B'])
# Output: 11


def write(lst, text):
    lst.append(text)
a = 50
b = a ** 2
c = [a]
if(b % a == 0):
    write(c, b)
else:
    write(c, a)
print(c[-1])
# Output: 2500

x = 5
print([y // 2 for y in range(6)][x])
# Output: 2

x = 100.1205
y = str(x)[6]
print('{:.{}f}'.format(x, y))
# Output: 100

a = []
b = [a, a, a]
for x in b:
    n = len(x)
    x.append(n)
print(b[0])
# Output: [0, 1, 2]
