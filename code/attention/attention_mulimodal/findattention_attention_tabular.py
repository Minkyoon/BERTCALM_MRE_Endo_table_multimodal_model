from clam_model import *
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create a DataFrame to store the image information
image_info_df = pd.DataFrame(columns=['filename', 'score', 'index', 'fold_number', 'serial_number'])


fold=6
with open(f'/home/minkyoon/CLAM2/results/update_multihead7_add_911_230823_forpaper/remission_multimodal_stratified_721_s1/split_{fold}_results.pkl', 'rb') as file:
    results = pickle.load(file)


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
for i in whole:                
    model = CLAM_SB()
    model.set_classifier(49)
    state_dict = torch.load(f'/home/minkyoon/CLAM2/results/update_multihead7_add_911_230823_forpaper/remission_multimodal_stratified_721_s1/s_{fold}_checkpoint.pt')
    model.load_state_dict(state_dict)    
    serial_num=i
    
    # 데이터 로드
    data=torch.load(f'/home/minkyoon/crohn/csv/clam/remission/csvandpt/{serial_num}.pt')
    tabular=pd.read_csv('/home/minkyoon/CLAM2/data/processed/remission_under_10.csv')
    tab=tabular.iloc[:,1:]
    tab=tab[tab['slide_id']==i]
    tab=tab.iloc[:,:49]
    tab=tab.fillna(0)
    numpy_array = tab.values
    tensor = torch.from_numpy(numpy_array).float()
    
    

    # 모델을 평가 모드로 설정
    model.eval()

    device = torch.device("cpu")
    model = model.to(device)
    data = data.to(device)

    # 데이터를 모델에 전달하고 attention score를 얻습니다.
    with torch.no_grad():
        _, _, _, A_raw, _,A,score= model(data, tensor)


    # 가장 높은 attention score를 가진 instance를 찾습니다.

    top_scores, top_indices = torch.topk(A, k=4, largest=True, dim=1)

    bottom_scores, bottom_indices = torch.topk(A, k=4, largest=False, dim=1)

    sorted_indices = torch.argsort(A, descending=True, dim=1)
    
    
    result_tensor = torch.zeros(1, 49)

# 각 텐서를 더합니다.
    for t in score:
        result_tensor += t
        
    




    
    
## 타블로 데이터    

df=pd.read_csv('/home/minkyoon/CLAM2/data/processed/remission_under_10.csv')


df=df.iloc[:, 1:50]

result_tensor = torch.zeros(1, 49)

# 각 텐서를 더합니다.
for t in score:
    result_tensor += t

# 텐서를 Python 리스트로 변환합니다.
result_list = result_tensor.tolist()[0]

print(result_list)

image_score=torch.tensor(result_list)
sorted_indices = now4.argsort(descending=True)
top_indices = sorted_indices[:10].cpu().numpy()

# 가장 높은 attention score와 해당 인덱스를 출력
print("Top 10 features with their attention scores:")
for i in top_indices:
    feature_name = df.columns[i]
    feature_values = df[feature_name]
    print(f"Feature: {feature_name}, Attention Score: {now4[i]:.4f}, ")
    
 
result_list



now=torch.tensor(result_list)
now2=torch.tensor(result_list)
now3=torch.tensor(result_list)
now4= now+now2+now3