import os


f = open('fiber_description.txt','r')
for line in f.readlines():
    item = line.split('  ')[0].replace(' ','').replace('\n','')
    new = item.split(':')
    new = ':'.join(['\''+new[1]+'\'', new[0]]) + ','
    line = new + line[len(new):]
    if '(' in line:
        idx = line.index('(')
        line = line[:idx].replace('\n','') + '#' + line[idx:].replace('\n','')
    print(line)
f.close()
