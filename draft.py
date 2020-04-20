d = {'A': 3, 'B':8}
for n in range(2,12,2):
    d['A'] += n
    d['B'] -= n
print(d['A'] + d['B'])