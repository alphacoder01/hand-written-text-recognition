# Hand written text recognition
---

Using the `IAM-lines dataset`. A CRNN(VGG+LSTM) has been trained using `keras` to recognize the text in the input image once segmented lines are obtained using computer-vision algorithm.
The above model achieves ~1.7% character error rate.

#### Segmented Lines from input Image
![Segmented lines from input image](https://raw.githubusercontent.com/alphacoder01/hand-writting-recognition/master/images/a01-000u.png)

#### ROI from input image
![ROI input image_1](https://raw.githubusercontent.com/alphacoder01/hand-writting-recognition/master/images/a01-000u-00.png)
#### Another Roi
![ROI input image_2](https://raw.githubusercontent.com/alphacoder01/hand-writting-recognition/master/images/a01-000u-01.png)

#### CMD Output 
![CMD SCR](https://raw.githubusercontent.com/alphacoder01/hand-writting-recognition/master/images/cmd_scr.png)

#### Final output of the Model in the text file
![final output](https://raw.githubusercontent.com/alphacoder01/hand-writting-recognition/master/images/final_op.png) 

### Dependencies:

1. Tensorflow_gpu 1.14
2. Keras 2.3.1
3. OpenCV 4.0
4. Numpy

### Dataset
A modified version of the IAM lines dataset is used, which can be found here:
`https://www.kaggle.com/ashish2001/iam-dataset-modified`

### How to use:

Place the input images in `data/check_image` and run the following command in the `Code` directory.
`python test.py --weight=../Pretrained_weights/Model--20--12.879.hdf5`
This will generate one text file for each of the input image in the `Code` directory containing the text, and will have the segmented lines stored in `data/Sentences/`

### Future work:

Better line segmentation algorithm to be deployed, inorder to get better results.

## UPDATE:

Using the code available at [this](https://github.com/Samir55/Image2Lines) repository, More refined line segmentation was possible.
Although the current version contains code written in c++. I'll post a python based version soon!

**Steps to run the C++ code available at the said repository in Windows 10**
- Install Opencv 4.x in the default path.
- Add the path `C:\opencv\build\x64\vc14\bin` to the system's PATH and user's Path environment variable.
- Install Visual Studio 2017 community edition and select `Desktop development with C++` workload and under individual components select the follwing (if you want minimal installation else go for full installation):
1. Windows 10 SDK(10.0.15063) for Desktop C++
2. Windows 10 SDK(10.0.15063) for UWP: C#, VB, JS
3. Windows 10 SDK(10.0.15063) for UWP: C++

- Now, create a new project as C++ console application and add delete the .cpp file generated.
- Right Click on the Source Files in Solution Explorer to add the files at the [repository](https://github.com/Samir55/Image2Lines/tree/master/src).
- Select the proper platform in the Dropdown in the top-middle `x64` will work for windows 10.
- Open the project properties go to
1. `C/C++ > General >` Add the following path `C:\opencv\build\include` to the Additional Include  Directories.
2. `Linker > General` Add the path `C:\opencv\buld\x64\vc14\lib` to the Additional Library Directories.
3. `Linker > Input` Add the .lib file at the end of Additional Dependencies seperated by a semicolon. The file will be located at `C:\opencv\build\x64\vc14\lib\opencv_world430d.lib`, and Apply the changes.

**NOTE:: There are few changes the the files in-order to work properly**

1. In main.cpp replace the first line as `#include"LineSegmentation.hpp"` to `#include"LineSegmentation.h"`.
2. In LineSegmentation.h comment the line `#include <cv.h>`, and replace `#include<opencv/cv.hpp>` to `#include<opencv2/opencv.hpp>`.

If evertything was done properly you could build the project and run it.

