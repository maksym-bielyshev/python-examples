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
