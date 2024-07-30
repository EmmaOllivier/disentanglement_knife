import os
import torch
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
from knife import KNIFE
import torch
from tqdm import tqdm


def train(encoder,MI,fc_expression,fc_ID, loader_train, loader_test,device,lr=0.0001,lr_MI=0.01,lamb=None,EPOCH=10,loss_affect=None,save_path=None, hyperparametre=None, Binary=True):
    

    save_model_name = 'encoder_'+str(lamb)+'lamb.pt'
    save_model_name_ID = 'ID_'+str(lamb)+'lamb.pt'
    save_model_name_Affect = 'Affect_'+str(lamb)+'lamb.pt'
    save_model_name_MI = 'MI_'+str(lamb)+'lamb.pt'
    save_log_name='result'+str(lamb)+'.csv'
    loss_affect = torch.nn.BCELoss(reduction='sum')
    loss_ID = torch.nn.CrossEntropyLoss(reduction='sum')


    optimizer = torch.optim.Adam(list(encoder.parameters())+list(fc_expression.parameters())+list(fc_ID.parameters()),lr=lr)
    optimizer_MI = torch.optim.Adam(list(MI.parameters()),lr=0.01)

    min_loss_val = None
    dic_log = {'loss_CE_train':[],'loss_CE_val':[],'loss_acc_train':[],'loss_acc_val':[],'loss_acc_ID_train': [],'MI':[]}
    count = 0 
    acc_max = 0
    
    for epoch in range(EPOCH):
        
        loader_train.dataset.reset()
        loss_task_tot = 0
        elem_sum = 0
        true_response_affect = 0
        true_response_ID = 0
        MI_loss_tot = 0
        
        encoder.train()
        fc_expression.train()
        fc_ID.train()
        MI.train()
        loop_train = tqdm(loader_train,colour='BLUE')
        for i,pack in enumerate(loop_train):

            img_tensor = pack[0].to(device)
            pain_tensor = pack[1].float().to(device)
            ID_tensor = pack[2].to(device)
            ID_tensor_one_hot = torch.nn.functional.one_hot(ID_tensor,49).float()
            elem_sum += img_tensor.shape[0]

            #Encoding
            encoded_img  = encoder(img_tensor)

            # UPDATE MI
            loss_MI = MI.loss(encoded_img.detach(),ID_tensor_one_hot)
            optimizer_MI.zero_grad()
            loss_MI.backward()
            optimizer_MI.step()
            MI_loss_tot += float(loss_MI)
            
            # TASK Affect
            output = fc_expression(encoded_img)
            loss_task_affect = loss_affect(output,pain_tensor)
            loss_task_tot += float(loss_task_affect) 
            true_response_affect += float(torch.sum(output.round() == pain_tensor))

            # Task ID
            output = fc_ID(encoded_img.detach())
            loss_task_ID = loss_ID(output,ID_tensor) 
            true_response_ID += float(torch.sum(output.max(dim=-1)[1] == ID_tensor))

            loss =  loss_task_ID + loss_task_affect + lamb*MI.I(encoded_img,ID_tensor_one_hot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop_train.set_description(f"Epoch [{epoch}/{EPOCH}] training")
            loop_train.set_postfix(loss_task = loss_task_tot/elem_sum,accuracy_pain=true_response_affect/elem_sum*100,accuracy_ID=true_response_ID/elem_sum*100,MI=MI_loss_tot/elem_sum)
        
        encoder.eval()
        fc_expression.eval()
        fc_ID.eval()
        MI.eval()

        loss_task_val = 0
        elem_sum_val = 0
        true_response_affect_val  =0
        true_response_ID_val = 0
        loop_test = tqdm(loader_test,colour='GREEN')
        for pack in loop_test:
            img_tensor = pack[0].to(device)
            pain_tensor = pack[1].float().to(device)
            ID_tensor = pack[2].to(device)

            elem_sum_val += img_tensor.shape[0]

            with torch.no_grad():
                #Encoding
                encoded_img  = encoder(img_tensor)

                # TASK Affect
                output = fc_expression(encoded_img)
                loss_task_affect_val = loss_affect(output,pain_tensor) 
                true_response_affect_val += float(torch.sum(output.round() == pain_tensor))
                loss_task_val += float(loss_task_affect_val)

                # Task ID
                output = fc_ID(encoded_img.detach())
                loss_task_ID_val = loss_ID(output,ID_tensor) 
                true_response_ID_val += float(torch.sum(output.max(dim=-1)[1] == ID_tensor))


            loop_test.set_description(f"Test lambda {lamb}")

            loop_test.set_postfix(loss_task = loss_task_val/elem_sum_val,accuracy_pain=true_response_affect_val/elem_sum_val*100,accuracy_ID=true_response_ID_val/elem_sum_val*100)
 
        dic_log['loss_CE_train'].append(loss_task_tot/elem_sum)
        dic_log['MI'].append(MI_loss_tot/elem_sum)
        dic_log['loss_acc_train'].append(true_response_affect/elem_sum*100)
        dic_log['loss_CE_val'].append(loss_task_val/elem_sum_val)
        dic_log['loss_acc_val'].append(true_response_affect_val/elem_sum_val*100)
        dic_log['loss_acc_ID_train'].append(true_response_ID/elem_sum*100)


        acc=true_response_affect_val/elem_sum_val*100

        if acc > acc_max :

            acc_max = acc
            
            torch.save(encoder.state_dict(),save_path+save_model_name)
            torch.save(fc_ID.state_dict(),save_path+save_model_name_ID)
            torch.save(fc_expression.state_dict(),save_path+save_model_name_Affect)
            torch.save(MI.state_dict(),save_path+save_model_name_MI)

        dataframe = pd.DataFrame(dic_log)
        dataframe.to_csv(save_path+save_log_name)







