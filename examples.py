# What is the output of this code?
def recf(lst, s=0):
    if len(lst) == 0:
        return s
    else:
        s += lst.pop()
        return recf(lst, s)
s = [5 for i in range(10)]
print(recf(s))

# Output: 50

# What is the output of this code?
a = list()
b = dict()
c = dict()
for n in range(3):
    a.append(n)
    b[n] = a
    c[n] = list(a)
print(b == c)

# Output: False

# What is the output of this code?
listOne = [1, 3, 5, 7, 9]
listTwo = [2, 5, 4, 3, 9]
for i in range(len(listOne)):
    if listOne[i] == listTwo[i]:
        print(listOne[i] * i)

# Output: 36

# What is the output of this code?
a = [1, 2, 3, 4, 5]
print(a[-1:None:-2])

# Output: [5, 3, 1]

# What is the output of this code?


class Myclass:
    n = 0

    def change(self, val):
        n = val
obj = Myclass()
obj.change(1)
print(obj.n)

# Output: 0

# What is the output of this code?
print('y', end="=")
x, y = 7, 9
y *= x
y %= 8
print(y)

# Output: y=7

# What is the output of this code?
lst = [34, 67, 34]
if lst.sort() == sorted(lst):
    print('true')
else:
    print('false')

# Output: false

# What is the output of this code?


def fun(x):
    if x == 1:
        return 1
    else:
        return x * fun(x - 1)
print(fun(5))

# Output: 120

# What is the output of this code?


def meth():
    g = [2, 3, 4, 5, 7]
    for i in g:
        if i % 2 == 1:
            yield i
for i in meth():
    print(i)
    break

# Output: 3

# What is the output of this code?
x = 5


def foo():
    print(x)
    pass
foo()

# Output: 5

# What is the output of this code?
d = {'A': 3, 'B': 8}
for n in range(2, 12, 2):
    d['A'] += n
    d['B'] -= n
print(d['A'] + d['B'])

# Output: 11

# What is the output of this code?


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

# What is the output of this code?
x = 5
print([y // 2 for y in range(6)][x])

# Output: 2

# What is the output of this code?
x = 100.1205
y = str(x)[6]
print('{:.{}f}'.format(x, y))

# Output: 100

# What is the output of this code?
a = []
b = [a, a, a]
for x in b:
    n = len(x)
    x.append(n)
print(b[0])

# Output: [0, 1, 2]
