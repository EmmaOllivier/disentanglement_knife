import os
import torch
import pandas as pd
import torchvision
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from utils import data_augm,data_adapt
from knife import KNIFE
from dataloader import Dataset_Biovid_image_binary_class
import model
import torch
import random
from tqdm import tqdm
from train_resnet import train

os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

device = "cuda"



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



Biovid_img_all = '/export/livia/home/vision/Eollivier/Biovid/'


BATCH_SIZE = 200
RESOLUTION = 112
nb_ID = 49


LEARNING_RATE = 0.01
LEARNING_RATE_FINETUNE = 0.000005
EPOCH_PRETRAIN = 10
EPOCH_FINETUNE = 15
FOLD=5
LAMBDA = [round(i*0.05,2) for i in range(44)]


arg_MI= {'zd_dim':1000, 'zc_dim':49,'hidden_state':100, 'layers':3, 'nb_mixture':10,'tri':False}

seed=0

make_deterministic(seed)

g = torch.Generator()
g.manual_seed(seed)


tr = data_augm(RESOLUTION)
tr_test = data_adapt(RESOLUTION)
tr_size = torchvision.transforms.Resize((RESOLUTION,RESOLUTION),antialias=True)


for fold in range(1, FOLD+1):


    print(f"Fold {fold}")
    print("-------")

    save_path = "../models/"+str(fold)+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    biovid_annot_train="../train"+str(fold)+".csv"
    biovid_annot_val="../valid"+str(fold)+".csv"

    Biovid_img_all = "../Biovid/sub_red_classes_img/"

    save_log_name='result_pretraining.csv'
    

    dataset_train = Dataset_Biovid_image_binary_class(Biovid_img_all,biovid_annot_train,transform = tr.transform,IDs = None,nb_image = None,preload=False)
    loader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=BATCH_SIZE, shuffle=True,
                                                num_workers=4,drop_last = True, worker_init_fn=seed_worker, generator=g) 

    # Validation
    dataset_test = Dataset_Biovid_image_binary_class(Biovid_img_all,biovid_annot_val,transform = tr_test.transform,IDs = None,nb_image = None,preload=False)
    loader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=BATCH_SIZE,
                                                num_workers=4, worker_init_fn=seed_worker, generator=g)


    encoder=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    fc_expression = model.Classif(1,False).to(device)
    fc_ID = model.Classif(nb_ID).to(device)

    MI = KNIFE(**arg_MI).to(device)

    optimizer = torch.optim.Adam(list(fc_expression.parameters())+list(fc_ID.parameters()),lr=LEARNING_RATE)
    optimizer_MI = torch.optim.Adam(list(MI.parameters()),lr=0.01)

    loss_BCE = torch.nn.BCELoss(reduction='sum')
    loss_CE = torch.nn.CrossEntropyLoss(reduction='sum')
    dic_log = {'loss_CE_train':[],'loss_CE_val':[],'loss_acc_train':[],'loss_acc_val':[],'loss_acc_ID_val': [],'MI':[]}

    for epoch in range(EPOCH_PRETRAIN):
        dataset_train.reset()
        loss_task_tot = 0
        elem_sum = 0
        true_response_affect = 0
        true_response_ID = 0
        MI_loss_tot = 0
        encoder.train()
        fc_expression.train()
        fc_ID.train()
        loop_train = tqdm(loader_train,colour='BLUE')
        for i,pack in enumerate(loop_train):

            img_tensor = pack[0].to(device)
            pain_tensor = pack[1].float().to(device)
            ID_tensor = pack[2].to(device)
            ID_tensor_one_hot = torch.nn.functional.one_hot(ID_tensor,49).float()
            with torch.no_grad():
                encoded_img  = encoder(img_tensor)

            # UPDATE MI
            loss_MI = MI.loss(encoded_img.detach(),ID_tensor_one_hot)
            optimizer_MI.zero_grad()
            loss_MI.backward()
            optimizer_MI.step()
            MI_loss_tot += float(loss_MI)
            
            # TASK Affect
            output = fc_expression(encoded_img)
            loss_task_affect = loss_BCE(output,pain_tensor) 
            true_response_affect += float(torch.sum(output.round() == pain_tensor))

            # Task ID
            output = fc_ID(encoded_img.detach())
            loss_task_ID = loss_CE(output,ID_tensor) 
            true_response_ID += float(torch.sum(output.max(dim=-1)[1] == ID_tensor))

            loss =  loss_task_affect + loss_task_ID
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elem_sum += img_tensor.shape[0]
            loss_task_tot += float(loss_task_affect)

            loop_train.set_description(f"Epoch [{epoch}/{EPOCH_PRETRAIN}] training")
            loop_train.set_postfix(loss_task = loss_task_tot/elem_sum,accuracy_pain=true_response_affect/elem_sum*100,accuracy_ID=true_response_ID/elem_sum*100,MI=MI_loss_tot/elem_sum)
        
        encoder.eval()
        fc_expression.eval()
        fc_ID.eval()

        loss_task_val = 0
        elem_sum_val = 0
        true_response_affect_val  =0
        true_response_ID_val = 0
        loop_test = tqdm(loader_test,colour='GREEN')
        for pack in loop_test:
            img_tensor = pack[0].to(device)
            pain_tensor = pack[1].float().to(device)
            ID_tensor = pack[2].to(device)
                
            with torch.no_grad():
                encoded_img  = encoder(img_tensor)
                # TASK Affect
                output =fc_expression(encoded_img)
                loss_task_affect_val = loss_BCE(output,pain_tensor) 
                true_response_affect_val += float(torch.sum(output.round() == pain_tensor))

                # Task ID
                output = fc_ID(encoded_img.detach())
                loss_task_ID_val = loss_CE(output,ID_tensor) 
                true_response_ID_val += float(torch.sum(output.max(dim=-1)[1] == ID_tensor))

            elem_sum_val += img_tensor.shape[0]
            loss_task_val += float(loss_task_affect_val)
            loop_test.set_description(f"Test")
            loop_test.set_postfix(loss_task = loss_task_val/elem_sum_val,accuracy_pain=true_response_affect_val/elem_sum_val*100,accuracy_ID=true_response_ID_val/elem_sum_val*100)

        dic_log['loss_CE_train'].append(loss_task_tot/elem_sum)
        dic_log['MI'].append(MI_loss_tot/elem_sum)
        dic_log['loss_acc_train'].append(true_response_affect/elem_sum*100)
        dic_log['loss_CE_val'].append(loss_task_val/elem_sum_val)
        dic_log['loss_acc_val'].append(true_response_affect_val/elem_sum_val*100)
        dic_log['loss_acc_ID_val'].append(true_response_ID_val/elem_sum_val*100)
        dataframe = pd.DataFrame(dic_log)
        dataframe.to_csv(save_path+save_log_name)
        torch.save(encoder.state_dict(),save_path+'encoder_pretrained.pt')
        torch.save(fc_ID.state_dict(),save_path+'ID_pretrained.pt')
        torch.save(fc_expression.state_dict(),save_path+'Affect_pretrained.pt')
        torch.save(MI.state_dict(),save_path+'MI_pretrained.pt')


    encoder=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    model_affect = model.Classif(1,False).to(device)
    model_ID = model.Classif(nb_ID).to(device)
    MI = KNIFE(**arg_MI).to(device)

    LAMBDA=[0,0.4,0.5,0.6,0.7]
    for lamb in LAMBDA:
    
        encoder=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        fc_expression = model.Classif(1,False).to(device)
        fc_ID = model.Classif(nb_ID).to(device)

        MI = KNIFE(**arg_MI).to(device)
        encoder.load_state_dict(torch.load(save_path+'encoder_pretrained.pt'))
        fc_expression.load_state_dict(torch.load(save_path+'Affect_pretrained.pt'))
        fc_ID.load_state_dict(torch.load(save_path+'ID_pretrained.pt'))


        MI.load_state_dict(torch.load(save_path+'MI_pretrained.pt'))

        

        train(encoder,MI,fc_expression,fc_ID,loader_train,loader_test,device,lamb=lamb,EPOCH=EPOCH_FINETUNE,lr=LEARNING_RATE_FINETUNE,save_path=save_path)


