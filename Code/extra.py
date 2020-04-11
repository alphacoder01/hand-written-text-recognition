from parameters import letters
import numpy as np

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

Train = True
files_names = []
lines = []
        # print(self.n , "Image Loading Start...")
if Train:
    line_path = "G:/Ateva_project/My_project/train_list.txt"
else:
    line_path = "G:/Ateva_project/My_project/val_list.txt"

with open(line_path,'r') as f:
    for line in f:
        lineSplit = line.strip().split('\t')
        linePath = lineSplit[0]
        name = linePath.split('/')[-1]
        text = lineSplit[1]
        lines.append((name,text))

# # print(lines)
# img_file = 'a06-008-05.png'
# idx = ([x[0] for x in lines].index(img_file))
# print(lines[idx][1])
Y_data = np.ones([16, 100])
hello = []
for i in range(16):
    text = lines[i][1]
    labels = text_to_labels(text)
    for no in range(len(labels)):
        Y_data[i][no] = int(labels[no])
print(Y_data)
