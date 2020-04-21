def meth():
    g=[2,3,4,5,7]
    for i in g:
        if i%2==1:
            yield i
for i in meth():
    print(i)
    break