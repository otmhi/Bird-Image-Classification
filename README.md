
## Object recognition and computer vision 2018/2019

### Assignment 3: Image classification 

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18/assignment3/bird_dataset.zip). The test image labels are not provided.

### Preprocessing 
Before beginning any training, run the following command :
```bash
python preprocess.py
```
This will create the cropped images of birds with [YOLO v3](https://github.com/eriklindernoren/PyTorch-YOLOv3) to help the models learn better. Once this is done, you can run the `main.py` script to train and validate your model .

#### Final Training 

Once you're happy with your model, you can retrain your model on all the samples ( train and validation dataset ) using the :
```bash
python final_train.py
```

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file]
```

To reproduce the performances on the leaderboard, you should run the final_train.py script with --epochs 40.
and then take the `model_40.pth`and run the evaluate.py script.

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.

#### Acknowledgments
Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.\\
This solution is a part of the 3rd assignment for Object recognition and computer vision 2018/2019 MVA Class [Jean Ponce](https://www.di.ens.fr/~ponce/), [Ivan Laptev](https://www.di.ens.fr/~laptev/), [Cordelia Schmid](http://lear.inrialpes.fr/~schmid/) and [Josef Sivic](https://www.di.ens.fr/~josef/), adapted from https://github.com/willowsierra/recvis18_a3. \\
link to the class : https://www.di.ens.fr/willow/teaching/recvis18/

