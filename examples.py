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
