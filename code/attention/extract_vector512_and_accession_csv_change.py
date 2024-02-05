import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from topk.svm import SmoothTop1SVM
            

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = True, k_sample=8, n_classes=2,
        instance_loss_fn= SmoothTop1SVM(n_classes = 2), subtyping=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)
        print(A)# softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h)
        print(A.shape)
        print(h.shape)
        print(M.shape) 
        logits = self.classifiers(M)
        print(logits)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        print(Y_hat)
        Y_prob = F.softmax(logits, dim = 1)
        print(Y_prob)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict, M

class CLAM_MB(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = True, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
import os
import torch
from torch.autograd import Variable

## 변동하고싶을때 이거
z=0

# CLAM 모델 정의 및 가중치 로드
model = CLAM_SB()
state_dict = torch.load(f'/home/minkyoon/first/CLAM/results/remission_for_muiltimodal/task_1_tumor_vs_normal_CLAM_50_s1/s_{z}_checkpoint.pt')
model.load_state_dict(state_dict) 

# 모델을 평가 모드로 설정
model.eval()

# 디바이스 설정
device = torch.device("cpu")
model = model.to(device)

# 디렉토리 내 모든 pt 파일을 순회
directory = '/home/minkyoon/first/testclam/feature/unifeature/pt_files'
output_directory = f'/home/minkyoon/crohn/csv/clam/remission/manyfold/fold{z}'
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(directory):
    if filename.endswith('.pt'):
        # 데이터 로드
        data = torch.load(os.path.join(directory, filename))

        # 디바이스 설정
        data = data.to(device)

        # 데이터를 모델에 전달하고 attention score를 얻습니다.
        with torch.no_grad():
            _, _, _, A_raw, _,M = model(data)

        # 512 vector를 추출
        vector = M.squeeze(0)

        # 결과를 새로운 pt 파일로 저장
        torch.save(vector, os.path.join(output_directory, filename))

print("Vector extraction and saving completed.")



import pandas as pd
import os
import shutil

z=0

# CSV 파일을 불러와서 'serial_number'와 'accession_number'를 매핑하는 dictionary를 만듭니다.
mapping_df = pd.read_csv('/home/minkyoon/crohn/csv/original_csv/serial_accession_match.csv')
serial_to_accession = mapping_df.set_index('serial_number')['accession_number'].to_dict()

output_directory=f'/home/minkyoon/crohn/csv/clam/remission/manyfold/fold{z}'
# 원본 파일 경로
source_dir = output_directory

# 대상 디렉토리가 존재하지 않으면 만듭니다.
target_dir = f'/home/minkyoon/crohn/csv/clam/remission/manyfold/fold{z}_accession'
os.makedirs(target_dir, exist_ok=True)

## 991 serial_number가없음 991 그래서 날아간듯
# 원본 디렉토리에서 모든 파일을 순회합니다.
for filename in os.listdir(source_dir):
    # 파일인지 확인합니다.
    if os.path.isfile(os.path.join(source_dir, filename)):
        # 파일명에서 확장자를 제외한 부분을 serial number로 사용합니다.
        serial_number = int(os.path.splitext(filename)[0])

        # 만약 serial number가 매핑 dictionary에 있으면, 새로운 파일명을 생성합니다.
        if serial_number in serial_to_accession:
            new_filename = str(int(serial_to_accession[serial_number])) + '.pt'
            
            # 파일을 새 위치에 복사합니다.
            shutil.copy(os.path.join(source_dir, filename), os.path.join(target_dir, new_filename))







import pandas as pd
import numpy as np

# CSV 파일을 불러와서 'serial_number'와 'accession_number'를 매핑하는 dictionary를 만듭니다.
mapping_df = pd.read_csv('/home/minkyoon/crohn/csv/original_csv/serial_accession_match.csv')
serial_to_accession = mapping_df.set_index('serial_number')['accession_number'].to_dict()

# 원본 파일 경로
source_file = f'/home/minkyoon/first/CLAM/splits/task_1_tumor_vs_normal_relapse_downsampling_100/splits_{z}.csv'

# 데이터 불러오기
data_df = pd.read_csv(source_file)

# 각 column에 대해 serial_number를 accession_number로 변경
for column in ['train', 'val', 'test']:
    data_df[column] = data_df[column].map(serial_to_accession)
    


# 결과를 CSV 파일로 저장
data_df.to_csv(f'/home/minkyoon/crohn/csv/clam/relapse/downsampling/clam_vecotr_512/accession_csv/Fold{z}.csv', index=False,float_format='%.0f')



#a=torch.load('/home/minkyoon/crohn/csv/clam/remission/512vectorpt/1179.pt')


#a.shape