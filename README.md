# Hand written text recognition
---

Using the `IAM-lines dataset`. A CRNN(VGG+LSTM) has been trained using `keras` to recognize the text in the input image once segmented lines are obtained using computer-vision algorithm.
The above model achieves ~1.7% character error rate.

![Segmented lines from input image](https://raw.githubusercontent.com/alphacoder01/hand-writting-recognition/master/images/a01-000u.png)

### Dependencies:

1. Tensorflow 1.14
2. Keras 2.3.1
3. OpenCV 4.0
4. Numpy

### How to use:

Place the input images in `data/check_image` and run the following command in the `Code` directory.
`python test.py --weight=../Pretrained_weights/Model--20--12.879.hdf5`
This will generate one text file for each of the input image in the `check_image` directory containing the text.

### Note:
The Model works very accurately on pre-segmented lines, but the segmentation algorithm needs more refinement!!

### Future work:

Better line segmentation algorithm to be deployed, inorder to get better results.
