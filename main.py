import torch
from model_clam import  MultimodalModel, CLAM_mre, CLAM_endo
from torch.utils.data import Dataset, DataLoader
from dataset import PTFilesDataset
import os
import pandas as pd
from utils.topk.svm import SmoothTop1SVM
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np

num_epochs=100
device='cuda:3'

def create_datasets_for_fold(split_file, pt_directory, mre_directory, label_csv, tabular_csv, mre_endo_csv):
    splits = pd.read_csv(split_file)
    train_ids = splits['train'].dropna().astype(int).tolist()
    val_ids = splits['val'].dropna().astype(int).tolist()
    test_ids = splits['test'].dropna().astype(int).tolist()

    train_dataset = PTFilesDataset(pt_directory, mre_directory, label_csv, tabular_csv, train_ids, mre_endo_csv)
    val_dataset = PTFilesDataset(pt_directory, mre_directory, label_csv, tabular_csv, val_ids, mre_endo_csv)
    test_dataset = PTFilesDataset(pt_directory, mre_directory, label_csv, tabular_csv, test_ids, mre_endo_csv)

    return train_dataset, val_dataset, test_dataset


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # 데이터 로드
        endo_data, mre_data, tabular_data, labels = batch
        
        mre_data=mre_data.to(device)
        endo_data=endo_data.to(device)
        tabular_data=tabular_data.to(device)
        labels=labels.to(device)
        
    
        

        # 모델 실행
        logits, results_dict_endo, results_dict_mre = model(endo_data, mre_data, tabular_data, label=labels)
        loss = criterion(logits, labels)
        instance_loss_endo = results_dict_endo['instance_loss']
        instance_loss_mre = results_dict_mre['instance_loss']
        
        
  
        
        bag_weight=0.7
        
        total_loss = bag_weight*loss + (1-bag_weight)*instance_loss_endo + 0.3*instance_loss_mre
        

        # 역전파
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        
    return 

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            endo_data, mre_data, tabular_data, labels = batch
            
            # 데이터를 device로 옮깁니다.
            mre_data = mre_data.to(device)
            endo_data = endo_data.to(device)
            tabular_data = tabular_data.to(device)
            labels = labels.to(device)

            # 모델 실행
            logits, results_dict_endo, results_dict_mre = model(endo_data, mre_data, tabular_data, label=labels)
            loss = criterion(logits, labels)
            instance_loss_endo = results_dict_endo['instance_loss']
            instance_loss_mre = results_dict_mre['instance_loss']

            bag_weight = 0.7
            total_loss += bag_weight * loss + (1 - bag_weight) * instance_loss_endo + 0.3 * instance_loss_mre

    return total_loss / len(val_loader)

def test(model, test_loader, fold):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            endo_data, mre_data, tabular_data, labels = batch
            
            # 데이터를 device로 옮깁니다.
            mre_data = mre_data.to(device)
            endo_data = endo_data.to(device)
            tabular_data = tabular_data.to(device)
            labels = labels.to(device)

            logits, _, _ = model(endo_data, mre_data, tabular_data, label=labels)
            _, predicted = torch.max(logits, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 계산된 메트릭스를 저장합니다.
    accuracy = accuracy_score(all_labels, all_predictions)
    sensitivity = recall_score(all_labels, all_predictions)
    specificity = recall_score(all_labels, all_predictions, pos_label=0)
    f1 = f1_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    roc = roc_auc_score(all_labels, all_predictions)

    with open(f'test_metrics{fold}_lr_0.001.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Sensitivity: {sensitivity}\n')
        f.write(f'Specificity: {specificity}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write(f'Confusion Matrix: \n{cm}\n')
        f.write(f'ROC AUC Score: {roc}\n')

    return accuracy
    

def main():
    for fold in range(10):  # 10 폴드
        split_file = f'/home/minkyoon/2023_CLAM_MUTLIMODAL/data/10fold_csv/splits_{fold}.csv'
        train_dataset, val_dataset, test_dataset = create_datasets_for_fold(
            split_file,
            pt_directory='/home/minkyoon/CLAM3/data/raw/feature_from_resnet/pt_files',
            mre_directory='/home/data/crohn/mre/feature_from_resnet/pt_files',
            label_csv='/home/minkyoon/2023_CLAM_MUTLIMODAL/data/processed/label.csv',
            tabular_csv='/home/minkyoon/2023_CLAM_MUTLIMODAL/data/processed/remission_under_10.csv',
            mre_endo_csv='/home/minkyoon/2023_CLAM_MUTLIMODAL/data/mre_csv/mre_endo_merged.csv'            
        )

        # 데이터 로더 설정
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 모델 초기화
        instance_loss_fn = SmoothTop1SVM(n_classes = 2)
        endo_model = CLAM_endo()
        mre_model = CLAM_mre()   # MRE 이미지 모델
        model = MultimodalModel(endo_model=endo_model,mre_model=mre_model,instance_loss_fn=instance_loss_fn)
        model=model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()
        


        # 훈련 및 검증 로직
        # 예: 각 에포크마다 훈련 및 검증 수행
        for epoch in range(num_epochs):
            print(f'epoch:{epoch}')
            train(model, train_loader, optimizer, criterion)
            validate(model, val_loader, criterion)

        # 테스트 데이터에 대한 평가
        test_accuracy = test(model, test_loader,fold)

        # 결과 출력 또는 저장
        print(f"Fold {fold}: Test Accuracy = {test_accuracy}")

if __name__ == "__main__":
    main()
