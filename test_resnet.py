

import os
import time
import torch
from torch import nn, optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchvision.models import resnet18, ResNet18_Weights
import model
import numpy as np
import pandas as pd
import dataloader
import utils
import PIL.Image as Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

device="cuda"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_deterministic(seed=0):
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

BACKBONE = "resnet18"

RESOLUTION = 112
    
def test_per_subject():
    seed=0

    make_deterministic(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    save_path="../models/"
    Biovid_img_all = '../Biovid/sub_red_classes_img/'
    tr_test = utils.data_adapt(RESOLUTION)

    biovid_annot_test = '../test_set.csv'
    dataset_test = dataloader.Dataset_Biovid_image_binary_class(Biovid_img_all,biovid_annot_test,transform = tr_test.transform,IDs = None,nb_image = None,preload=False)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    loss_affect = torch.nn.BCELoss(reduction='sum')

    encoder=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    fc_expression = model.Classif(1,False).to(device)

    encoder.load_state_dict(torch.load(save_path+'...pt'))
    fc_expression.load_state_dict(torch.load(save_path+'...pt'))


    encoder.eval()
    fc_expression.eval()
    loss_task_tot_val = 0
    elem_sum_val = 0
    true_response_affect_val  =0
    accuracy_list=[]
    current_subject=-1
    loop_test = tqdm(test_loader,colour='GREEN')
    for pack in loop_test:
        
        img_tensor = pack[0].to(device)
        pain_tensor = pack[1].float().to(device)
        ID_tensor = pack[2].to(device)
        if (current_subject==ID_tensor or current_subject==-1):
            current_subject=ID_tensor
            elem_sum_val += img_tensor.shape[0]

            with torch.no_grad():
                encoded_img  = encoder(img_tensor)
                output =fc_expression(encoded_img)
                loss_task_affect_val = loss_affect(output,pain_tensor)
                loss_task_tot_val += float(loss_task_affect_val) 
                true_response_affect_val +=  float(torch.sum(output.round() == pain_tensor))

            acc = true_response_affect_val/elem_sum_val*100
            loop_test.set_postfix(accuracy_pain=acc)
            

        elif(current_subject!=ID_tensor):
            print("new")
            current_subject=ID_tensor
            accuracy_list.append(acc)

            loss_task_tot_val = 0
            elem_sum_val = 0
            true_response_affect_val=0

            elem_sum_val += img_tensor.shape[0]

            with torch.no_grad():
                encoded_img  = encoder(img_tensor)
                output =fc_expression(encoded_img)
                loss_task_affect_val = loss_affect(output,pain_tensor)
                loss_task_tot_val += float(loss_task_affect_val) 
                true_response_affect_val +=  float(torch.sum(output.round() == pain_tensor))

            acc = true_response_affect_val/elem_sum_val*100
            loop_test.set_postfix(accuracy_pain=acc)
    
    return accuracy_list
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test():
    seed=0

    make_deterministic(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    bs = 20
    save_path="../models/"
    Biovid_img_all = '../Biovid/sub_red_classes_img/'
    tr_test = utils.data_adapt(RESOLUTION)

    biovid_annot_test = '../test_set.csv'
    dataset_test = dataloader.Dataset_Biovid_image_binary_class(Biovid_img_all,biovid_annot_test,transform = tr_test.transform,IDs = None,nb_image = None,preload=False)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=bs, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    encoder=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    print(count_parameters(encoder))
    
    fc_expression = model.Classif(1,False).to(device)

    encoder.load_state_dict(torch.load(save_path+'...pt'))
    fc_expression.load_state_dict(torch.load(save_path+'....pt'))

    print('---- testing ----')
    encoder.eval()
    fc_expression.eval()

    elem_sum_val = 0
    elem_pain = 0
    elem_no_pain = 0
    true_response_affect=0
    true_response_pain=0
    true_response_no_pain=0
    loop_test = tqdm(test_loader,colour='GREEN')
    for pack in loop_test:
        img_tensor = pack[0].to(device)

        pain_tensor = pack[1].float().to(device)
        ID_tensor = pack[2].to(device)
            
        with torch.no_grad():
            
            encoded_img  = encoder(img_tensor)
            output =fc_expression(encoded_img)

            true_response_affect += float(torch.sum(output.round() == pain_tensor))
            true_response_pain += float(torch.sum((output.round() == pain_tensor) & (pain_tensor==torch.ones(img_tensor.shape[0]).to(device))))
            true_response_no_pain += float(torch.sum((output.round() == pain_tensor) & (pain_tensor==torch.zeros(img_tensor.shape[0]).to(device))))

            elem_pain += float(torch.sum(pain_tensor==torch.ones(img_tensor.shape[0]).to(device)))
            elem_no_pain += float(torch.sum(pain_tensor==torch.zeros(img_tensor.shape[0]).to(device)))


        elem_sum_val += img_tensor.shape[0]
        loop_test.set_description(f"Test")
        loop_test.set_postfix(accuracy=true_response_affect/elem_sum_val*100)
    print("accuracy : "+str(true_response_affect/elem_sum_val*100))
    print("accuracy pain: "+str(true_response_pain/elem_pain*100))
    print("accuracy no pain: "+str(true_response_no_pain/elem_no_pain*100))
    print('end')


def test_per_video_softmax_output():

    seed=0

    make_deterministic(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    fold=1

    save_path="../models/"+str(fold)+"/"
    Biovid_img_all = '../Biovid/sub_red_classes_img/'

    tr_test = utils.data_adapt(RESOLUTION)

    biovid_annot_test = '../test_set.csv'


    dataset_test = dataloader.Dataset_Biovid_image_binary_class(Biovid_img_all,biovid_annot_test,transform = tr_test.transform,IDs = None,nb_image = None,preload=False)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    dic_log = {'accuracy':[],'threshold':[]}


    encoder=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    fc_expression = model.Classif(1,False).to(device)


    encoder.load_state_dict(torch.load(save_path+'...pt'))
    fc_expression.load_state_dict(torch.load(save_path+'...pt'))

    threshold = [round(x * 0.05, 2) for x in range(0, 22)]
    #threshold=[0.25]

    for t in threshold:

        encoder.eval()
        fc_expression.eval()

        pre_list = []
        GT_list = []

        val_ce = 0
        video_results=[]
        video_output=[]
        current_video=-1
        loop_test = tqdm(test_loader ,colour='GREEN')
        for i, (batch_val_x, batch_val_y, batch_val_id, batch_val_video) in enumerate(loop_test):
            if(batch_val_video.data[0] != current_video and current_video!=-1):
                video_mean=sum(video_output)/len(video_output)
                if(video_mean>=t):
                    reg_video=[1]*len(video_results)
                else:
                    reg_video=[0]*len(video_results)
                pre_list = np.hstack((pre_list, reg_video))
                val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
                loop_test.set_postfix(accuracy_pain=val_acc*100)
                current_video=batch_val_video.data[0]
                video_results=[]
                video_output=[]
            elif(batch_val_video.data[0] != current_video):
                current_video=batch_val_video.data[0]

            GT_list = np.hstack((GT_list, batch_val_y.numpy()))
            batch_val_x = Variable(batch_val_x).to(device)
            batch_val_y = Variable(batch_val_y).to(device)
            batch_val_y_np = batch_val_y.data.cpu().numpy()

            batch_fea = encoder(batch_val_x)
            batch_p = fc_expression(batch_fea)
            batch_fea_np = batch_p.data.cpu().numpy()

            batch_results = batch_p.cpu().data.numpy().round()
            video_results=np.hstack((video_results, batch_results))
            batch_output = batch_p.cpu().data.numpy()

            video_output = np.hstack((video_output, batch_output))
            
            if(i==len(test_loader)-1):
                video_mean=sum(video_results)/len(video_results)
                if(video_mean>t):
                    reg_video=[1]*len(video_results)
                else:
                    reg_video=[0]*len(video_results)
                pre_list = np.hstack((pre_list, reg_video))
                val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
                loop_test.set_postfix(accuracy_pain=val_acc*100)

        val_acc_pain = (np.sum(((GT_list != pre_list) & (GT_list == 1 )).astype(float)) / (np.sum((GT_list == 1).astype(float))))
        print(val_acc_pain)

        val_acc_no_pain = (np.sum(((GT_list != pre_list) & (GT_list == 0 )).astype(float)) / (np.sum((GT_list == 0).astype(float))))
        print(val_acc_no_pain)

        val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
        val_ce = val_ce / i
        print(val_acc)


        dic_log['threshold'].append(t)
        dic_log['accuracy'].append(val_acc)

        dataframe = pd.DataFrame(dic_log)
        dataframe.to_csv(save_path+"result_per_video_threshold.csv")

        
if __name__ == '__main__':
    test_per_video_softmax_output()

    
    
