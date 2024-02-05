from clam_model import *
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

fold=5
with open(f'/home/minkyoon/first/CLAM/results/remission_stratified_for_recommend_setting_7,2,1_learningragechange/task_1_tumor_vs_normal_CLAM_50_s1/split_{fold}_results.pkl', 'rb') as file:
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
            
print(tn)
            

    
model = CLAM_SB()
state_dict = torch.load(f'/home/minkyoon/first/CLAM/results/remission_stratified_for_recommend_setting_7,2,1_learningragechange/task_1_tumor_vs_normal_CLAM_50_s1/s_{fold}_checkpoint.pt')
model.load_state_dict(state_dict)    

serial_num=316

# 데이터 로드
data=torch.load(f'/home/minkyoon/crohn/csv/clam/remission/csvandpt/{serial_num}.pt')

# 모델을 평가 모드로 설정
model.eval()

device = torch.device("cpu")
model = model.to(device)
data = data.to(device)

# 데이터를 모델에 전달하고 attention score를 얻습니다.
with torch.no_grad():
    _, _, _, _, _,A = model(data)


# 가장 높은 attention score를 가진 instance를 찾습니다.

top_scores, top_indices = torch.topk(A, k=7, largest=True, dim=1)

bottom_scores, bottom_indices = torch.topk(A, k=7, largest=False, dim=1)



data = pd.read_csv(f'/home/minkyoon/crohn/csv/clam/remission/csvandpt/{serial_num}.csv')

# top_indices를 numpy 배열로 f
top_indices = top_indices.cpu().numpy()

# 각 인덱스에 대한 이미지 표시
for index in top_indices[0]:
    # npy 파일 경로 가져오기
    npy_path = data.iloc[index]['filepath']
    
    # npy 파일 로드
    image = np.load(npy_path)
    
    # 이미지 표시
    plt.imshow(image, cmap='gray')
    plt.title(f"order:{index},serial:")
    plt.show()
    


bottom_indices = bottom_indices.cpu().numpy()

# 각 인덱스에 대한 이미지 표시
for index in bottom_indices[0]:
    # npy 파일 경로 가져오기
    npy_path = data.iloc[index]['filepath']
    
    # npy 파일 로드
    image = np.load(npy_path)
    
    # 이미지 표시
    plt.imshow(image, cmap='gray')
    plt.title(f"order:{index},serial:")
    plt.show()