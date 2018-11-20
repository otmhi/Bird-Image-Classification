from PIL import Image, ImageFilter
import numpy as np

import torchvision.transforms as transforms

def random_blur(img, p = 0.7):
    a = np.random.uniform(size=1)
    if (a> p).all(): return img.filter(ImageFilter.GaussianBlur(radius=0.2))
    return img


data_transforms = {
    'train': transforms.Compose([
        transforms.transforms.Resize((224,224)),
        #transforms.transforms.Lambda(random_blur),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomVerticalFlip(0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


