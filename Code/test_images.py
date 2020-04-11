import cv2
import os
import shutil
import random
def truncateLabel( text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
    cost = 0
    for i in range(len(text)):
        if i != 0 and text[i] == text[i - 1]:
            cost += 2
        else:
            cost += 1
        if cost > maxTextLen:
            return text[:i]
    return text

allfiles = []
filePath = 'G:/Ateva_Project/My_project/data/'
f = open(filePath+'lines.txt')
chars = set()
bad_samples = []
bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
for line in f:
    if not line or line[0] == '#':
        continue
    lineSplit = line.strip().split(' ')
    fileNameSplit = lineSplit[0].split('-')
    fileName = filePath + 'lines/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' +lineSplit[0] + '.png'
    gtText_list = lineSplit[8].split('|')
    gtText = truncateLabel(' '.join(gtText_list), 100)
    # print(gtText)
    allfiles.append((fileName,gtText))


print(allfiles[0])
random.shuffle(allfiles)
print(len(allfiles))
train_files = allfiles[:12686]
val_files = allfiles[12686:]


with open('val_list.txt','w') as f:
    for file in val_files:
        f.write("{}\t{}\n".format(file[0],file[1]))

with open('train_list.txt','w') as f:
    for file in train_files:
        f.write("{}\t{}\n".format(file[0],file[1]))


for file in train_files:
    shutil.move(file[0],"G:/Ateva_project/My_project/data/train/")
print('Done for train files')

for file in val_files:
    shutil.move(file[0], "G:/Ateva_project/My_project/data/val/")
print('Done for val files')

    # gtText_list = lineSplit[8].split('|')
    # gtText = truncateLabel(' '.join(gtText_list), 100)
    # print(gtText)
    # chars = (set(list(gtText)))
    # # print(chars)

    # if not os.path.getsize(fileName):
    #     bad_samples.append(lineSplit[0] + '.png')
    #     continue

    # if set(bad_samples) != set(bad_samples_reference):
    #     print("Warning, damaged images found:", bad_samples)
    #     print("Damaged images expected:", bad_samples_reference)



