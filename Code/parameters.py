CHAR_VECTOR = " \"!#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
letters = [letter for letter in CHAR_VECTOR]
num_classes = len(letters)+1

img_w, img_h = 800, 64

# Network parameters
batch_size = 8
val_batch_size = 8

downsample_factor = 4
max_text_len = 100