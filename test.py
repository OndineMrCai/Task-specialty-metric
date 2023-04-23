import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def torch_cov(input_vec): 
    """compute the covariance matrix(vector-horizontal, X.T*X)"""   
    x = input_vec- torch.mean(input_vec,axis = 0)
    cov_matrix = torch.mm(x.T, x) / x.shape[0] 
    return cov_matrix


def compute_metric(hiddenstate_dict, device):
    """"compute task-specialty metric in the paper"""
    Groupmean_matrix = torch.cat([torch.mean(matrix.to(device), dim = 0,keepdim = True) for matrix in hiddenstate_dict.values()], dim = 0) # compute hy(l)
    Betweenclass_matrix = torch_cov(Groupmean_matrix) # compute between-class variability
    Withinclass_matirx = torch.zeros(Betweenclass_matrix.size()).to(device) 
    for matrix in hiddenstate_dict.values():
        Withinclass_matirx = Withinclass_matirx + torch_cov(matrix.to(device)) / len(hiddenstate_dict)
        del matrix #compute within-class variability
    metric = torch.trace(torch.mm(Withinclass_matirx, torch.linalg.pinv(Betweenclass_matrix))) / len(hiddenstate_dict) #compute the metirc
    del Withinclass_matirx, Betweenclass_matrix
    for vector in hiddenstate_vectors.values():
        del vector
    torch.cuda.empty_cache()
    return metric

tokenizer = RobertaTokenizer.from_pretrained('/home/csc/.cache/huggingface/hub/models--roberta-large/snapshots/716877d372b884cad6d419d828bac6c85b3b18d9')
model = RobertaModel.from_pretrained('/home/csc/.cache/huggingface/hub/models--roberta-large/snapshots/716877d372b884cad6d419d828bac6c85b3b18d9', output_hidden_states= True)

train_data = []
train_label = []
with open('data/train.tsv', 'r') as f:
    for line in f:
        line = line.strip('\n').split('\t')  
        train_data = train_data + [line[3]]
        train_label = train_label + [line [1]]

def creat_empty_dict():
    All_layer_dict = []
    for i in range(25):
        All_layer_dict = All_layer_dict + [{'0':'','1':''}]
    return All_layer_dict

def hidden_state(train_data, train_label, All_layer_dict, device):
    for i in range(len(train_data)):
        encoded_input = tokenizer(train_data[i], return_tensors='pt')
        with torch.no_grad():
            output = model(**encoded_input)
        if train_label[i] == '0':
            if All_layer_dict[0]['0'] == '':
                for j in range(len(output.hidden_states)):
                    All_layer_dict[j]['0'] = torch.mean(output.hidden_states[j],axis=1)
            else:
                for j in range(len(output.hidden_states)):
                    All_layer_dict[j]['0'] = torch.cat((All_layer_dict[j]['0'],torch.mean(output.hidden_states[j],axis=1)),dim = 0)
        elif train_label[i] == '1':
            if All_layer_dict[0]['1'] == '':
                for j in range(len(output.hidden_states)):
                    All_layer_dict[j]['1'] = torch.mean(output.hidden_states[j],axis=1)
            else:
                for j in range(len(output.hidden_states)):
                    All_layer_dict[j]['1'] = torch.cat((All_layer_dict[j]['1'],torch.mean(output.hidden_states[j],axis=1)),dim = 0)
    print('done')
    return All_layer_dict

Note=open('result.txt',mode='w')
numberOdata = len(train_data)
m_list = []
All_layer_dict = creat_empty_dict()


All_layer_dict = hidden_state(train_data, train_label, All_layer_dict, device)


# print(All_layer_dict[0]['0'].shape)
# print(All_layer_dict[0]['1'].shape)
# print(All_layer_dict[0]['0'])
# print(All_layer_dict[0]['1'])
# print(All_layer_dict[20]['0'])
# print(All_layer_dict[20]['1'])
    
for j in range(len(All_layer_dict)):
    m = compute_metric(All_layer_dict[j], device).cpu()
    print(m)
    m_list = m_list + [m] 
    del All_layer_dict[j]['0']
    del All_layer_dict[j]['1']
    torch.cuda.empty_cache()

x = list(range(0,25))
plt.figure()
plt.plot(x,m_list)
plt.show()