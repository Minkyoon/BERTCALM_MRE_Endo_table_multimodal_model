import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from clam_model import CLAM_SB  # Assuming clam_model is a module you have
from scipy.stats import skew
# Initialize DataFrame to store image information
image_info_df = pd.DataFrame(columns=['filename', 'score', 'index', 'fold_number', 'serial_number'])

# Load results
#fold =6
for fold in range(10):

    results_path = f'/home/minkyoon/CLAM2/results/update_multihead7_add_911_230823_forpaper/remission_multimodal_stratified_721_s1/split_{fold}_results.pkl'
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
    
    
    for i in tn:
        model = CLAM_SB()
        model.set_classifier(49)
        state_dict_path = f'/home/minkyoon/CLAM2/results/update_multihead7_add_911_230823_forpaper/remission_multimodal_stratified_721_s1/s_{fold}_checkpoint.pt'
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
            _, _, _, _, _, A, _ = model(data, tensor)
        
        # Get top and bottom 4 attention scores and indices
        top_scores, top_indices = torch.topk(A, k=4, largest=True, dim=1)
        bottom_scores, bottom_indices = torch.topk(A, k=4, largest=False, dim=1)
        
        # Calculate statistics for A
        A_numpy = A.cpu().numpy()
        std_dev = np.std(A_numpy).item()
        median_val = np.median(A_numpy).item()
        skewness = skew(A_numpy[0]).item()
        range_val = np.ptp(A_numpy).item()
        quartiles = np.percentile(A_numpy, [25, 50, 75]).tolist()
        
        # Create a title for the bar plot
        title = f"Serial Number: {i}\nStd: {std_dev:.4f}, Median: {median_val:.4f},  Range: {range_val:.4f}, "
        
        # Generate bar plot
        plt.bar(range(len(A_numpy[0])), A_numpy[0])
        plt.title(title)
        plt.xlabel("Index")
        plt.ylabel("Attention Score")
        plt.ylim(0, 1) 
        
        # Save plot
        save_path = f"/home/minkyoon/crohn/for_clam/attention/attention_mulimodal/image/attention/tn_y1/serial_{i}.png"
        plt.savefig(save_path)
        plt.close()
        
        # Create a dictionary with the statistics data you want to add
        stat_data = {
            "serial_number": [i],
            "std_dev": [std_dev],
            "median_val": [median_val],
            "skewness": [skewness],
            "range_val": [range_val],
            "Q1": [quartiles[0]],
            "Q2": [quartiles[1]],
            "Q3": [quartiles[2]]
        }

        # Create a new DataFrame with this data
        new_row_df = pd.DataFrame(stat_data)

        # Concatenate the existing DataFrame with the new row DataFrame
        image_info_df = pd.concat([image_info_df, new_row_df], ignore_index=True)
                
                
  
# Save the statistics DataFrame
image_info_df.to_csv("statistics_tn.csv", index=False)
