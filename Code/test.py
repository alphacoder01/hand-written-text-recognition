import cv2
import itertools,os, time
import numpy as np
from Model import get_Model
from parameters import letters
import argparse
from tensorflow.python.keras import backend as K
K.set_learning_phase(0)
import shutil


def decode_label(out):
        out_best = list(np.argmax(out[0, 2:], 1))
        # print((out_best))
        out_best = [k for k, g in itertools.groupby(out_best)]
        out_str=''
        for i in out_best:
            if i<len(letters):
                out_str += letters[i]
        return out_str


parser = argparse.ArgumentParser()
parser.add_argument('-w','--weight',help='weight file directory',type=str)
parser.add_argument('-t','--test_image',help='Test image Directory',type=str,default='../data/check_image/')
args = parser.parse_args()

model = get_Model(training=False)

try:
    model.load_weights(args.weight)
    print("...Using Saved Weights...")
except:
    raise Exception("...No Weights found...")

test_dir = args.test_image
test_imgs = os.listdir(args.test_image)

start = time.time()

#########
# Loop to extract lines from the input images
# #########
for test_img in test_imgs:
    image = cv2.imread(test_dir+test_img,cv2.IMREAD_GRAYSCALE)
    # for i in range(0,image.shape[1],64):
    #     cv2.line(image,(0,i),(image.shape[0],i),(0,0,255),2)
    # cv2.imshow('hello',image)
    # cv2.waitKey(0)

    # res = cv2.resize(image,(800,800),interpolation=cv2.INTER_LINEAR)
    ret,thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi = image[y:y+h, x:x+w]
        if h<w and h>20 and w>20:  #i.e draw only horizontal recatangles.
            cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
            cv2.imwrite('../data/Sentences/'+test_img+str(i)+'.png',roi)
            cv2.imwrite('../data/tempSentences/'+test_img+str(i)+'.png',roi)

    cv2.imshow('marked',image)
    cv2.waitKey(0)

################
#Loop to predict output
###############
    line_dir = '../data/tempSentences/'
    test_lines = os.listdir(line_dir)
    f = open(test_img[:-4]+'.txt','w+')
# f = open('hello'+'.txt','w+')

    for test_img in test_lines:
        img = cv2.imread(line_dir+test_img,cv2.IMREAD_GRAYSCALE)

        image_pred = img.astype(np.float32)
        img_pred = cv2.resize(image_pred,(800,64),interpolation=cv2.INTER_LINEAR)
        img_pred = (img_pred/255.0)*2.0 - 1.0
        img_pred = img_pred.T
        img_pred = np.expand_dims(img_pred,axis=-1)
        img_pred = np.expand_dims(img_pred, axis =0)

        net_out_value = model.predict(img_pred)
        pred_texts = decode_label(net_out_value)

        print(pred_texts)

        f.write('{}\n'.format(pred_texts))
    f.close()

    filelist = [ f for f in os.listdir('../data/tempSentences') if f.endswith(".png") ]
    for f in filelist:
        os.remove(os.path.join('../data/tempSentences', f))
end = time.time()
print('...DONE in {} seconds...'.format(end-start))


