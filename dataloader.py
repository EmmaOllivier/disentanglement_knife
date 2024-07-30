import os,glob
import pandas as pd
import torch
import torchvision
import random
import numpy as np
from tqdm import tqdm


class Dataset_Biovid_image_binary_class(torch.utils.data.Dataset):
    def __init__(self,PATH_IMG,PATH_ANOT,transform = lambda x: x,IDs = None,set_type = None,nb_image = None,preload=False,loaded_resolution=224):
        self.PATH_IMG = PATH_IMG
        self.PATH_ANOT = PATH_ANOT
        self.transform = transform
        self.nb_image = nb_image
        self.set_type = set_type
        self.IDs = IDs
        self.preload = preload

        self.reset()

        dataframe = pd.read_csv(self.PATH_ANOT)

        N = len(dataframe)
        self.resize  = torchvision.transforms.Resize((loaded_resolution,loaded_resolution),antialias=True)
        self.img_loaded = torch.zeros((N,) + (3,loaded_resolution,loaded_resolution),dtype=torch.uint8)
        self.dic_image = {}
        if preload:
            loader = tqdm(dataframe.index)
            for i,p in enumerate(loader):
                img = torchvision.io.read_image(self.dataframe.loc[p,'path'])
                self.dic_image[p] = i
                img = self.resize(img)
                self.img_loaded[i]=img
        
    def __len__(self):
        return len(self.index_link)
    
    def reset(self):
        dataframe = pd.read_csv(self.PATH_ANOT)

        self.dataframe = dataframe


        self.dic_ID = {d : i for i,d in enumerate(set(self.dataframe['ID']))}
        if not self.nb_image is None:
            self.index_link = list(self.dataframe.sample(self.nb_image).index)
        else:
            self.index_link = list(self.dataframe.index)


    def __getitem__(self,idx):
        index = self.index_link[idx]
        try :
            i = self.dic_image[index]
        except:
            i = len(self.dic_image)
            self.dic_image[index] = i
            try:
                self.img_loaded[i] = self.resize(torchvision.io.read_image(self.dataframe.loc[index,'path']))
            except :
                print(self.dataframe.loc[index,'path'])
        
        img_tensor = self.img_loaded[i]
        img_tensor = self.transform(img_tensor)

        pain_tensor = self.dataframe.loc[index,'pain']
        ID_tensor = self.dic_ID[self.dataframe.loc[index,'ID']]
        video_tensor = self.dataframe.loc[index,'id_video']
        
        return img_tensor, pain_tensor,ID_tensor, video_tensor

class Dataset_Biovid_image_class(torch.utils.data.Dataset):
    def __init__(self,PATH_IMG,PATH_ANOT,transform = lambda x: x,IDs = None,set_type = None,nb_image = None,nb_fold=1,preload=False,loaded_resolution=224):
        self.PATH_IMG = PATH_IMG
        self.PATH_ANOT = PATH_ANOT
        self.transform = transform
        self.nb_image = nb_image
        self.set_type = set_type
        self.IDs = IDs
        self.preload = preload
        self.reset()
        q_fold = 40//nb_fold
        self.fold = [[j for j in range(i*q_fold,(i+1)*q_fold)] for i in range(nb_fold)]

        dataframe = pd.read_csv(self.PATH_ANOT)
        N = len(dataframe)
        self.resize  = torchvision.transforms.Resize((loaded_resolution,loaded_resolution),antialias=True)
        self.img_loaded = torch.zeros((N,) + (3,loaded_resolution,loaded_resolution),dtype=torch.uint8)
        self.dic_image = {}
        if preload:
            loader = tqdm(dataframe.index)
            for i,p in enumerate(loader):
                img = torchvision.io.read_image(self.dataframe.loc[p,'path'])
                self.dic_image[p] = i
                img = self.resize(img)
                self.img_loaded[i]=img
        
    def __len__(self):
        return len(self.index_link)
    
    def reset(self,fold=None,keep=False):
        dataframe = pd.read_csv(self.PATH_ANOT)
        self.dataframe = dataframe
        if not fold is None and fold < len(self.fold) and fold >=0:
            if not keep:
                self.dataframe = self.dataframe[~ self.dataframe['id_video'].isin(self.fold[fold])]
            else:
                self.dataframe = self.dataframe[self.dataframe['id_video'].isin(self.fold[fold])]

        self.dic_ID = {d : i for i,d in enumerate(set(self.dataframe['ID']))}
        if not self.nb_image is None:
            self.index_link = list(self.dataframe.sample(self.nb_image).index)
        else:
            self.index_link = list(self.dataframe.index)


    def __getitem__(self,idx):
        index = self.index_link[idx]
        try :
            i = self.dic_image[index]
        except:
            i = len(self.dic_image)
            self.dic_image[index] = i
            try:
                self.img_loaded[i] = self.resize(torchvision.io.read_image(self.dataframe.loc[index,'path']))
            except :
                print(self.dataframe.loc[index,'path'])
        
        img_tensor = self.img_loaded[i]
        img_tensor = self.transform(img_tensor)

        pain_tensor = self.dataframe.loc[index,'pain']
        ID_tensor = self.dic_ID[self.dataframe.loc[index,'ID']]
        video_tensor = self.dataframe.loc[index,'id_video']
        
        return img_tensor, pain_tensor,ID_tensor, video_tensor

class Dataset_Biovid_image(torch.utils.data.Dataset):
    def __init__(self,PATH_IMG,PATH_ANOT,transform = lambda x: x,IDs = None,set_type = None,nb_image = None,loaded_resolution=224):
        self.PATH_IMG = PATH_IMG
        self.PATH_ANOT = PATH_ANOT
        self.transform = transform
        self.nb_image = nb_image
        self.set_type = set_type
        self.IDs = IDs

        self.reset()
        dataframe = pd.read_csv(self.PATH_ANOT)
        N = len(dataframe)
        self.resize  = torchvision.transforms.Resize((loaded_resolution,loaded_resolution),antialias=True)
        self.img_loaded = torch.zeros((N,) + (3,loaded_resolution,loaded_resolution),dtype=torch.uint8)
        self.dic_image = {}
        
        
        
    def __len__(self):
        return len(self.index_link)
    
    def reset(self):
        dataframe = pd.read_csv(self.PATH_ANOT)

        self.dataframe = dataframe
        self.dataframe['seed'] = np.random.randint(1000,size=(len(self.dataframe),))
        if not self.nb_image is None:
            self.index_link = list(self.dataframe.sample(self.nb_image).index)
        else:
            self.index_link = list(self.dataframe.index)


    def __getitem__(self,idx):
        index = self.index_link[idx]
        path = self.PATH_IMG+'/'+self.dataframe.loc[index,'path'][14:]
        glob_path = glob.glob(path)
        random.seed(int(self.dataframe.loc[index,'seed']))
        self.dic_ID = {d : i for i,d in enumerate(set(self.dataframe['ID']))}

        try:
            path = glob_path[random.randint(0,len(glob_path)-1)]
        except:
            print(path)
        
        
        try :
            i = self.dic_image[index]
        except:
            i = len(self.dic_image)
            self.dic_image[index] = i
            try:
                self.img_loaded[i] = self.resize(torchvision.io.read_image(path))
            except :
                print(self.PATH_IMG+self.dataframe.loc[index,'path'][14:])
        
        img_tensor = self.img_loaded[i]
        img_tensor = self.transform(img_tensor)

        pain_tensor = self.dataframe.loc[index,'pain']
        ID_tensor = self.dic_ID[self.dataframe.loc[index,'ID']]
        video_tensor = self.dataframe.loc[index,'id_video']
        
        return img_tensor, pain_tensor,ID_tensor,video_tensor


