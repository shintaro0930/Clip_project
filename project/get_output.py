import re

# input_text = input("input:")

with open('output.txt') as f:
    lines:list = f.readlines()

line_list = []
prob_list = []
line_list_list = []
count = 0
for line in lines:
    count += 1
    if(re.search('[=]+', line)):
        continue
    line = line.rstrip()
    prob = line
    line = line.split(':')[0]
    line_list.append(line)
    prob = prob.split(' ')[-1]
    prob_list.append(prob)
    print(f'str: {line}')
    print(f'prob: {prob}')
    if(count / 3 == 0):
        line_list_list.append(line_list)

for out in line_list_list:
    print("入ってない")
    print(out)