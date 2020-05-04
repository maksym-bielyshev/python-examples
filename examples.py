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
