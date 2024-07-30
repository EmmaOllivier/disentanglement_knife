import numpy as np
from torch.autograd import Variable
import PIL.Image as Image
import torch
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
import pandas as pd



def del_extra_keys(model_par_dir):
    model_par_dict = torch.load(model_par_dir)
    model_par_dict_clone = model_par_dict.copy()
    for key, value in model_par_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_par_dict[key]
    
    return model_par_dict


class data_augm:
    def __init__(self,resolution):
        self.H_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.Jitter = torchvision.transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.5,1.5),saturation=(0.5,1.5),hue=(-0.1,0.1))
        self.resize = torchvision.transforms.Resize((resolution,resolution),antialias=True)

    def transform(self,x):
        x = self.resize(x)
        x = self.H_flip(x)
        x = self.Jitter(x)
        x = x/255
        return x

class data_adapt:
    def __init__(self,resolution):
        self.resize = torchvision.transforms.Resize((resolution,resolution),antialias=True)

    def transform(self,x):
        x = self.resize(x)
        x = x/255
        return x
    
