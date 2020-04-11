import cv2
import itertools,os, time
import numpy as np
from Model import get_Model
from parameters import letters
import argparse
from tensorflow.python.keras import backend as K
K.set_learning_phase(0)



def decode_label(out):
        out_best = list(np.argmax(out[0, 2:], 1))
        # print((out_best))
        out_best = [k for k, g in itertools.groupby(out_best)]
        out_str=''
        for i in out_best:
            if i<len(letters):
                out_str += letters[i]
            # print(out_str)
        return out_str

parser = argparse.ArgumentParser()
parser.add_argument('-w','--weight',help='weight file directory',type=str)
parser.add_argument('-t','--test_image',help='Test image Directory',type=str,default='G:/Ateva_project/My_project/data/check_image/')
args = parser.parse_args()

model = get_Model(training=False)

try:
    model.load_weights(args.weight)
    print("...Using Saved Weights...")
except:
    raise Exception("...No Weights found...")

test_dir = args.test_image
test_imgs = os.listdir(args.test_image)
total = 0
acc = 0
letter_total =0
letter_acc = 0

start = time.time()

for test_img in test_imgs:
    img = cv2.imread(test_dir+test_img,cv2.IMREAD_GRAYSCALE)

    image_pred = img.astype(np.float32)
    img_pred = cv2.resize(image_pred,(800,64),interpolation=cv2.INTER_LINEAR)
    img_pred = (img_pred/255.0)*2.0 - 1.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred,axis=-1)
    img_pred = np.expand_dims(img_pred, axis =0)

    net_out_value = model.predict(img_pred)
    pred_texts = decode_label(net_out_value)

    print(pred_texts)