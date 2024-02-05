import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from clam_model import CLAM_SB  # Assuming clam_model is a module you have

# Initialize DataFrame to store image information
image_info_df = pd.DataFrame(columns=['filename', 'score', 'index', 'fold_number', 'serial_number'])

# Load results
#fold =6
united_score= torch.zeros(1, 49)
image_sco=torch.zeros(1,1)
for fold in range(10):

    results_path = f'/home/minkyoon/CLAM2/results2/update_multihead1_add_911_230915_forpaper_save_multihead/remission_multimodal_stratified_721_s1/split_{fold}_results.pkl'
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
        pred_label = np.argmax(info['prob'])
        
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
    model = CLAM_SB()
    model.set_classifier(49)
    for i in whole:

        state_dict_path = f'/home/minkyoon/CLAM2/results2/update_multihead1_add_911_230915_forpaper_save_multihead/remission_multimodal_stratified_721_s1/s_{fold}_checkpoint.pt'
        model.load_state_dict(torch.load(state_dict_path))
        
        # Load data
        data = torch.load(f'/home/minkyoon/crohn/csv/clam/remission/csvandpt/{i}.pt')
        tabular = pd.read_csv('/home/minkyoon/CLAM2/data/processed/remission_under_10.csv')
        tab = tabular.iloc[:, 1:][tabular['slide_id'] == i].iloc[:, :49].fillna(0)
        tensor = torch.FloatTensor(tab.values)
        
        # Model evaluation
        model.eval()
        # Move model and data to the same device
        device = torch.device("cpu")
        model = model.to(device)
        data = data.to(device)
        tensor = tensor.to(device)
            
        
        with torch.no_grad():
            _, _, _, _, _, A, score, score_image = model(data, tensor)
        

        
        # Get top and bottom 4 attention scores and indices
        top_scores, top_indices = torch.topk(A, k=4, largest=True, dim=1)
        bottom_scores, bottom_indices = torch.topk(A, k=4, largest=False, dim=1)
        

        result_tensor = torch.zeros(1, 49)

        for t in score:
            result_tensor += t
            
        united_score +=result_tensor
        image_sco +=score_image[0]
        
        
        
        



united_score=united_score.squeeze()
sorted_indices = united_score.argsort(descending=True)
top_indices = sorted_indices[:20].cpu().numpy()
result_list = united_score.tolist()

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

# Plotting
plt.figure(figsize=(12, 10))
plt.barh(sorted_features, sorted_scores, color='skyblue')
plt.xlabel('Attention Score')
plt.ylabel('Feature Name')
plt.title('Features Ranked by Attention Score')

# Adding the text labels inside the bar plots
for index, value in enumerate(sorted_scores):
    plt.text(value, index, f"{value:.4f}")

plt.tight_layout()
plt.savefig("top_features_barplot.png")
plt.show()

print(image_sco)