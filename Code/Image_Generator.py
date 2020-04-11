import cv2
import os, random
import numpy as np
from parameters import letters,max_text_len

def labels_to_text(labels):
    return list(map(lambda x: letters[int(x)],labels))
# print(labels_to_text("0123456"))

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))


class TextImageGenerator:
    def __init__(self, img_dirpath, img_w, img_h, batch_size, downsample_factor, max_text_len=max_text_len):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # image dir path
        self.img_dir = os.listdir(self.img_dirpath)     # images list
        self.n = len(self.img_dir)                      # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []

    def truncateLabel( self, text, maxTextLen):
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

    def build_data(self,Train=True):
        files_names = []
        lines = []
        print(self.n , "Image Loading Start...")
        if Train:
            line_path = "../data/train_list.txt"
        else:
            line_path = "../data/val_list.txt"

        with open(line_path,'r') as f:
            for line in f:
                lineSplit = line.strip().split('\t')
                linePath = lineSplit[0]
                name = linePath.split('/')[-1]
                text = lineSplit[1]
                lines.append((name,text))

        bad_samples = []
        bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']

        for i, img_file in enumerate(self.img_dir):
            img = cv2.imread(self.img_dirpath+img_file,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(self.img_w,self.img_h))
            img = img.astype(np.float32)
            img = (img/255.0)*2.0 - 1.0
            self.imgs[i,:,:] = img

            if not os.path.getsize(self.img_dirpath+img_file):
                bad_samples.append(lineSplit[0] + '.png')
                continue

            idx = ([x[0] for x in lines].index(img_file))
            gtText = lines[idx][1]
            self.texts.append(gtText)
        print(len(self.texts)==self.n)
        print(self.n, "Image Loading Finish")
            
    
    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]] , self.texts[self.indexes[self.cur_index]]

    
    def next_batch(self):
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h,1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size,1))*(self.img_w//self.downsample_factor-2)
            label_length = np.zeros((self.batch_size,1))

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img,-1)
                X_data[i] = img
                
                label_list = text_to_labels(text)
                for no in range(len(label_list)):
                    Y_data[i][no] = label_list[no]
                
                label_length[i] = len(text)

            inputs={
                'the_input' : X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length' : label_length 
            }

            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs,outputs)