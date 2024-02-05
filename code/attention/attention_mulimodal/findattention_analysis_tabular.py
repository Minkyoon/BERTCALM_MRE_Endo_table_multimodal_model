import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from model.model_clam import *   # Assuming clam_model is a module you have
from utils.topk.svm import SmoothTop1SVM
import pandas as pd

device = torch.device("cpu")
# Load the mapping CSV
mapping_df = pd.read_csv('/home/minkyoon/2023_CLAM_MUTLIMODAL/data/mre_csv/mre_endo_merged.csv')
mapping_dict = dict(zip(mapping_df['serial_number_endo'], mapping_df['serial_number_mri']))

# Initialize DataFrame to store image information
image_info_df = pd.DataFrame(columns=['filename', 'score', 'index', 'fold_number', 'serial_number'],
                             dtype=object) 

# Load results
#fold =6
united_score= torch.zeros(1, 49)
mre_sco=torch.zeros(1,1)
endo_sco=torch.zeros(1,1)
total_instances =0

for fold in range(10):

    results_path = f'/home/minkyoon/2023_CLAM_MUTLIMODAL/result_main/dropout+mlp+0.001lr_bag1_weigted_sample/test_results_fold_{fold}.pkl'
    with open(results_path, 'rb') as file:
        results = pickle.load(file)

    # Initialize lists for true positives, true negatives, false positives, false negatives
 
    tp = []
    tn = []
    fp = []
    fn = []

    # Iterate over the dictionary items
    for slide_id, info in results.items():
        # Determine the predicted label (class with highest probability)
        #pred_label = np.argmax(info['prob'])
        pred_label = 1 if info['prob'] > 0.5 else 0
        
        # Compare the predicted and true labels
        if pred_label == info['label']:
            # If they're the same, it's either a true positive or true negative
            if pred_label == 1:
                tp.append(int(slide_id))  # True positive
            else:
                tn.append(int(slide_id))  # True negative
        else:
            # If they're different, it's either a false positive or false negative
            if pred_label == 1:
                fp.append(int(slide_id))  # False positive
            else:
                fn.append(int(slide_id))  # False 
    
    whole=tp+tn+fp+fn
    for i in whole:
        instance_loss_fn = SmoothTop1SVM(n_classes = 2).to(device)
        endo_model = CLAM_endo()
        mre_model = CLAM_mre()   # MRE 이미지 모델
        model = MultimodalModel(endo_model=endo_model,mre_model=mre_model,instance_loss_fn=instance_loss_fn)
        model=model.to(device)
        state_dict_path = f'/home/minkyoon/2023_CLAM_MUTLIMODAL/result_main/dropout+mlp+0.001lr_bag1_weigted_sample/best_model_fold_{fold}.pt'
        model.load_state_dict(torch.load(state_dict_path))
        
        mre_serial_number = mapping_dict.get(i)
        # Load data
        data = torch.load(f'/home/minkyoon/CLAM3/data/raw/feature_from_resnet/pt_files/{i}.pt')
        mre= torch.load(f'/home/data/crohn/mre/feature_from_resnet/pt_files/{mre_serial_number}.pt')
        tabular = pd.read_csv('/home/minkyoon/2023_CLAM_MUTLIMODAL/data/processed/remission_under_10.csv')
        tab = tabular.iloc[:, 1:][tabular['slide_id'] == i].iloc[:, :49].fillna(0)
        tensor = torch.FloatTensor(tab.values)
        
        # Model evaluation
        model.eval()
        # Move model and data to the same device
        
        model = model.to(device)
        data = data.to(device)
        mre=mre.to(device)
        tensor = tensor.to(device)
            
        
        with torch.no_grad():
            _, _, _, A_raw_endo, A_raw_mre, endo_attention, mre_attention, tabular_attention= model(data, mre, tensor)
        
        result_tensor = torch.zeros(1, 49)

        for t in tabular_attention:
            result_tensor += t
            
        united_score +=result_tensor
        mre_sco += mre_attention[0]
        endo_sco += endo_attention[0]
    total_instances += len(whole)
        
united_score=united_score.squeeze()
avg_united_score = united_score / total_instances
avg_endo_sco = endo_sco / total_instances
avg_mre_sco = mre_sco / total_instances
sorted_indices = united_score.argsort(descending=True)
top_indices = sorted_indices[:20].cpu().numpy()
result_list = avg_united_score.tolist()

# 가장 높은 attention score와 해당 인덱스를 출력
print("Top 49 features with their attention scores:")
for i in top_indices:
    feature_name = tab.columns[i]
    feature_values = tab[feature_name]
    print(f"Feature: {feature_name}, Attention Score: {result_list[i]:.4f}, ")
    
 
# Data Preparation
top_features = [tab.columns[i] for i in top_indices]  # Extracting feature names
top_scores = [result_list[i] for i in top_indices]  # Extracting corresponding scores

# Sorting by attention score for better visualization
sorted_indices = np.argsort(top_scores)
sorted_features = [top_features[i] for i in sorted_indices]
sorted_scores = [top_scores[i] for i in sorted_indices]

# # Plotting
# plt.figure(figsize=(12, 10))
# plt.barh(sorted_features, sorted_scores, color='skyblue')
# plt.xlabel('Attention Score')
# plt.ylabel('Feature Name')
# plt.title('Features Ranked by Attention Score')

# # Adding the text labels inside the bar plots
# for index, value in enumerate(sorted_scores):
#     plt.text(value, index, f"{value:.4f}")

# plt.tight_layout()
# plt.savefig("top_features_barplot2.png")
# plt.show()


# Plotting
plt.figure(figsize=(12, 10))
plt.barh(sorted_features, sorted_scores, color='navy')  # Changed to a darker blue color
plt.xlabel('Attention Score')
plt.ylabel('Feature Name')

# Removing the box around the plot
plt.box(False)

# Adding the text labels inside the bar plots
for index, value in enumerate(sorted_scores):
    plt.text(value, index, f"{value:.4f}")

plt.tight_layout()
plt.savefig("top_features_barplot4.png")
plt.show()

print(avg_endo_sco)
print(avg_mre_sco)