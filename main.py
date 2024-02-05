import torch
from torch.nn.functional import softmax
import torch.nn.functional as F
from model_clam import  MultimodalModel, CLAM_mre, CLAM_endo
from torch.utils.data import Dataset, DataLoader
from dataset import *
import os
import pandas as pd
from utils.topk.svm import SmoothTop1SVM
from utils.utils import set_seeds, FocalLoss
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
import pickle
import random




num_epochs=200
device='cuda:0'

def save_results_to_pkl(test_data, predict_proba, predicted, output_path):
    """
    Save results as a dictionary in pkl format.
    
    Parameters:
    - test_data: DataFrame that contains the test data.
    - predict_proba: 2D array-like where the right column has the probability of label being 1.
    - predicted: 1D array-like, the predicted labels.
    - output_path: The path where to save the pkl file.
    """
    results_dict = {}
    for idx, slide_id in enumerate(test_data['slide_id'].values):
        results_dict[slide_id] = {
            'slide_id': slide_id,
            'prob': predict_proba[idx],
            'label': predicted[idx]
        }
    with open(output_path, 'wb') as file:
        pickle.dump(results_dict, file)

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
        
        #focal loss 용
        labels_one_hot = F.one_hot(labels, num_classes=2).float()
        

        # 모델 실행
        logits, results_dict_endo, results_dict_mre = model(endo_data, mre_data, tabular_data, label=labels)
        #loss = criterion(logits, labels)
        loss = criterion(logits, labels_one_hot)
        instance_loss_endo = results_dict_endo['instance_loss']
        instance_loss_mre = results_dict_mre['instance_loss']
        
        
  
        
        bag_weight=1.7
        
        total_loss = 1.7*loss + 1*instance_loss_endo + 1*instance_loss_mre
        

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
            #focal loss 용
            labels_one_hot = F.one_hot(labels, num_classes=2).float()

            # 모델 실행
            logits, results_dict_endo, results_dict_mre = model(endo_data, mre_data, tabular_data, label=labels)
            #loss = criterion(logits, labels)
            loss = criterion(logits, labels_one_hot)
            instance_loss_endo = results_dict_endo['instance_loss']
            instance_loss_mre = results_dict_mre['instance_loss']

            bag_weight = 0.7
            total_loss +=1.7*loss + 1*instance_loss_endo + 1*instance_loss_mre

    return total_loss / len(val_loader)

def test(model, test_loader, fold, split_file,  results_dir):
    model.eval()
    all_labels = []
    all_predictions = []
    Y_prob = []

    test_ids = pd.read_csv(split_file)['test'].dropna().astype(int).tolist()
    with torch.no_grad():
        for batch in test_loader:
            endo_data, mre_data, tabular_data, labels = batch
            
            # 데이터를 device로 옮깁니다.
            mre_data = mre_data.to(device)
            endo_data = endo_data.to(device)
            tabular_data = tabular_data.to(device)
            labels = labels.to(device)

            logits, _, _ = model(endo_data, mre_data, tabular_data, label=labels)
            probabilities = softmax(logits, dim=1)  # 로짓을 확률로 변환
            Y_prob.extend(probabilities.cpu().numpy())
            _, predicted = torch.max(logits, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())



    # 계산된 메트릭스를 저장합니다.
    predict_proba = [prob[1] for prob in Y_prob] 
    accuracy = accuracy_score(all_labels, all_predictions)
    sensitivity = recall_score(all_labels, all_predictions)
    specificity = recall_score(all_labels, all_predictions, pos_label=0)
    f1 = f1_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    roc = roc_auc_score(all_labels, predict_proba )

    output_path = os.path.join(results_dir, f'test_results_fold_{fold}.txt')

    with open(output_path, 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Sensitivity: {sensitivity}\n')
        f.write(f'Specificity: {specificity}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write(f'Confusion Matrix: \n{cm}\n')
        f.write(f'ROC AUC Score: {roc}\n')

    # Save results to pkl
    test_data = pd.DataFrame({'slide_id': test_ids})
    predict_proba = [prob[1] for prob in Y_prob]  # Assuming Y_prob is probability output
    #predicted = all_predictions
    predicted = all_labels
    output_pkl_path = os.path.join(results_dir, f'test_results_fold_{fold}.pkl')
    save_results_to_pkl(test_data, predict_proba, predicted, output_pkl_path)

    

    return accuracy
    

def main():
    set_seeds(42)
    folder_name = "valid_mlp+0.001lr_bag1.7_instance1_weigted_sample_pretrained”"  # Replace with your desired folder name
    results_dir = f'./result_main/{folder_name}'

    

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)



    for fold in range(10):  # 10 폴드
        split_file = f'/home/minkyoon/2023_CLAM_MUTLIMODAL/data/10fold_csv2/splits_{fold}.csv'
        train_dataset, val_dataset, test_dataset = create_datasets_for_fold(
            split_file,
            pt_directory='/home/minkyoon/CLAM3/data/raw/feature_from_resnet/pt_files',
            mre_directory='/home/data/crohn/mre/feature_from_resnet/pt_files',
            label_csv='/home/minkyoon/2023_CLAM_MUTLIMODAL/data/processed/label.csv',
            tabular_csv='/home/minkyoon/2023_CLAM_MUTLIMODAL/data/processed/remission_under_10.csv',
            mre_endo_csv='/home/minkyoon/2023_CLAM_MUTLIMODAL/data/mre_csv/mre_endo_merged.csv'            
        )

        # 데이터 로더 설정
        train_loader = get_split_loader(train_dataset, training=True, weighted=True)
        #train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 모델 초기화
        focal_loss_fn = FocalLoss(alpha=1, gamma=2.0).to(device)
        cross_loss_fn= torch.nn.CrossEntropyLoss().to(device)
        instance_loss_fn = SmoothTop1SVM(n_classes = 2).to(device)
        endo_model = CLAM_endo()
        mre_model = CLAM_mre()   # MRE 이미지 모델
        model = MultimodalModel(endo_model=endo_model,mre_model=mre_model,instance_loss_fn=instance_loss_fn)
        model=model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = cross_loss_fn
        


        # 훈련 및 검증 로직
        # 예: 각 에포크마다 훈련 및 검증 수행

        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = 30  # Set your patience level

        for epoch in range(num_epochs):
            print(f'epoch:{epoch}')
            train(model, train_loader, optimizer, criterion)
            val_loss = validate(model, val_loader, criterion)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save the model
                best_model_path= os.path.join(results_dir, f'best_model_fold_{fold}.pt')
                torch.save(model.state_dict(), best_model_path)
                print(f'Model saved: Improved validation loss to {best_val_loss}')
            else:
                epochs_no_improve += 1
                print(f'No improvement in validation loss for {epochs_no_improve} epochs')

            # Early stopping
            if epochs_no_improve == patience:
                print('Early stopping triggered')
                break

        # 테스트 데이터에 대한 평가
        model.load_state_dict(torch.load(best_model_path))
        test_accuracy = test(model, test_loader,fold, split_file, results_dir)

        # 결과 출력 또는 저장
        print(f"Fold {fold}: Test Accuracy = {test_accuracy}")

if __name__ == "__main__":
    main()
