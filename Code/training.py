import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import SGD,Adadelta,Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from Model import get_Model
# from DataLoader import Batch, DataLoader, FilePaths
from Image_Generator import TextImageGenerator
from parameters import *
K.set_learning_phase(0)

model = get_Model(training=True)

try:
    model.load_weights('../Pretrained_weights/Model--20--12.879.hdf5')
    print("...Previous Weights found...")
except:
    print('...Using New Weights...')
    pass


train_file_path = '../data/train/'
tiger_train = TextImageGenerator(train_file_path,img_w,img_h,batch_size,downsample_factor)
print('Building Data')
tiger_train.build_data(Train=True)
print('Done Building Data')

valid_file_path = '../data/val/'
tiger_val = TextImageGenerator(valid_file_path,img_w,img_h,val_batch_size,downsample_factor)
tiger_val.build_data(Train=False)


# sgd = SGD(lr=0.02,
#               decay=1e-6,
#               momentum=0.9,
#               nesterov=True)
ada = Adam(lr=0.001)
early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience = 4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='Model--{epoch:02d}--{val_loss:.3f}.hdf5',monitor='loss',verbose=1, mode ='min', period=1)

model.compile(loss={'ctc':lambda y_true, y_pred,:y_pred}, optimizer=ada)
print(model.summary())
print('Training network')
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch = int(tiger_train.n/batch_size),
                    epochs=20,
                    callbacks=[checkpoint],
                    validation_data=tiger_val.next_batch(),
                    validation_steps=int(tiger_val.n/val_batch_size)
                    )


