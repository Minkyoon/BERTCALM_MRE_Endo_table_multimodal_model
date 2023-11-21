import torch
from model_clam import CLAM_SB
from torch.utils.data import Dataset, DataLoader
from dataset import PTFilesDataset
import os
import pandas as pd
from utils.topk.svm import SmoothTop1SVM

num_epochs=100
device='cuda:2'

def create_datasets_for_fold(split_file, pt_directory, label_csv, tabular_csv):
    splits = pd.read_csv(split_file)
    train_ids = splits['train'].dropna().astype(int).tolist()
    val_ids = splits['val'].dropna().astype(int).tolist()
    test_ids = splits['test'].dropna().astype(int).tolist()

    train_dataset = PTFilesDataset(pt_directory, label_csv, tabular_csv, train_ids)
    val_dataset = PTFilesDataset(pt_directory, label_csv, tabular_csv, val_ids)
    test_dataset = PTFilesDataset(pt_directory, label_csv, tabular_csv ,test_ids)

    return train_dataset, val_dataset, test_dataset


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # 데이터 로드
        images, tabular_data, labels = batch
        
        images=images.to(device)
        tabular_data=tabular_data.to(device)
        labels=labels.to(device)
    
        

        # 모델 실행
        logits, Y_prob, Y_hat, _, instance_dict = model(images, tabular_data, label=labels)
        loss = criterion(logits, labels)
        instance_loss = instance_dict['instance_loss']
        
        print(loss)
        print(instance_loss)
        
        bag_weight=0.7
        
        total_loss = bag_weight*loss + (1-bag_weight)*instance_loss
        

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
            images, tabular_data, labels  = batch
            images=images.to(device)
            tabular_data=tabular_data.to(device)
            labels=labels.to(device)
        
            

            # 모델 실행
            logits, Y_prob, Y_hat, _, instance_dict = model(images, tabular_data, label=labels)
            loss = criterion(logits, labels)
            instance_loss = instance_dict['instance_loss']
            
            bag_weight=0.7
            
            total_loss = bag_weight*loss + (1-bag_weight)*instance_loss
    return total_loss / len(val_loader)

def test(model, test_loader):
    model.eval()
    correct = 0
    total =0
    with torch.no_grad():
        for batch in test_loader:
            images, tabular_data, labels = batch
            images = images.to(device)
            tabular_data = tabular_data.to(device)
            labels= labels.to(device)
            
            
            logits, Y_prob, Y_hat, _,_ = model(images, tabular_data, label=labels)
            _, predicted=torch.max(logits.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100* correct /total
    return accuracy
    

def main():
    for fold in range(1):  # 10 폴드
        split_file = f'/home/minkyoon/CLAM2/splits/remission_multimodal_stratified_721/splits_{fold}.csv'
        train_dataset, val_dataset, test_dataset = create_datasets_for_fold(
            split_file,
            '/home/minkyoon/CLAM3/data/raw/feature_from_resnet/pt_files',
            '/home/minkyoon/2023_CLAM_MUTLIMODAL/data/processed/label.csv',
            '/home/minkyoon/2023_CLAM_MUTLIMODAL/data/processed/remission_under_10.csv'            
        )

        # 데이터 로더 설정
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 모델 초기화
        instance_loss_fn = SmoothTop1SVM(n_classes = 2)
        model = CLAM_SB(instance_loss_fn=instance_loss_fn)
        model.relocate()
        model.set_classifier(49)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        criterion = torch.nn.CrossEntropyLoss()
        


        # 훈련 및 검증 로직
        # 예: 각 에포크마다 훈련 및 검증 수행
        for epoch in range(num_epochs):
            train(model, train_loader, optimizer, criterion)
            validate(model, val_loader, criterion)

        # 테스트 데이터에 대한 평가
        test_accuracy = test(model, test_loader)

        # 결과 출력 또는 저장
        print(f"Fold {fold}: Test Accuracy = {test_accuracy}")

if __name__ == "__main__":
    main()
