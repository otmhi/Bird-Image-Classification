import os
import sys
import warnings
warnings.filterwarnings("ignore")

if not os.path.isdir('./PyTorch-YOLOv3'):
    os.system('git clone https://github.com/eriklindernoren/PyTorch-YOLOv3')
os.chdir('PyTorch-YOLOv3/weights/')
if not os.path.isfile('yolov3.weights'):
    os.system('wget https://pjreddie.com/media/files/yolov3.weights')
os.chdir('..')
sys.path.insert(0, './')


from models import *
from utils.utils import *
from utils.datasets import *


import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='bird_dataset', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--tol', type=int, default=15, help='bounding box tolerance')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
parser.add_argument('--pad', type=bool, default=False, help='wether to pad the image to keep its ratio or not')
opt = parser.parse_args()
print(opt)

if not os.path.isdir('../cropped_bird_dataset'):

    print('copying the bird dataset to create the cropped dataset')
    os.system('cp -r ../bird_dataset/ ../cropped_bird_dataset')
    cuda = torch.cuda.is_available() and opt.use_cuda


    # Set up model
    model = Darknet(opt.config_path, img_size=opt.img_size)
    model.load_weights(opt.weights_path)

    if cuda:
        model.cuda()

    model.eval() # Set in evaluation mode

    data_transforms = transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)),
                                          transforms.ToTensor()])
    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, sample

    datasets.ImageFolder.__getitem__ = __getitem__

    classes = load_classes(opt.class_path) # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def make_square(im, min_size=224, fill_color=(0, 0, 0, 0)):
        x, y = im.size
        ratio = min(min_size/x, min_size/y)
        x, y = int(ratio*x), int(ratio*y)
        im = im.resize((x,y), Image.ANTIALIAS)
        new_im = Image.new('RGB', (min_size, min_size), fill_color)
        new_im.paste(im, ((min_size - x) // 2, (min_size - y) // 2))
        return new_im

    print('beginning detection')

    for split in {'train' , 'val', 'test'}:
        print('beginning detection for '+split )
        data_loader = DataLoader(
                    datasets.ImageFolder('../'+opt.image_folder+'/'+split+'_images',
                                         transform=data_transforms),
                    batch_size=opt.batch_size, shuffle=False, num_workers=1)


        imgs = []           # Stores image paths
        img_detections = [] # Stores detections for each image index



        for img_paths, input_imgs in tqdm(data_loader):
            img_paths = img_paths[0]
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)


            #print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

            # Save image and detections
            imgs.append(img_paths)
            img_detections.extend(detections)

        print ('\nSaving ' + split+' images:')
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            img = np.array(Image.open(path))

            # The amount of padding that was added
            pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
            # Image height and width after padding is removed
            unpad_h = opt.img_size - pad_y
            unpad_w = opt.img_size - pad_x

            # Draw bounding boxes and labels of detections
            if detections is not None:
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    if  classes[int(cls_pred)] == 'bird':
                        # Rescale coordinates to original dimensions
                        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                        x_start, y_start = int(x1), int(y1)
                        x_final, y_final = int(x_start + box_w), int(y_start + box_h)
                        cropped = img[max(0, y_start-opt.tol):min(img.shape[0], y_final+opt.tol), 
                                      max(0, x_start-opt.tol):min(img.shape[1], x_final+opt.tol)]
                        if opt.pad : 
                            im = Image.fromarray(cropped)
                            cropped = make_square(im)
                        if split != 'test':
                            plt.imsave(path[:3]+'cropped_'+path[3:-4]+'_cropped'+str(opt.tol)+path[-4:], np.array(cropped))
                        else : 
                            plt.imsave(path[:3]+'cropped_'+path[3:], np.array(cropped))

else : 
    print('cropped data exists already')