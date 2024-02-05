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
with open(f'/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting_7,2,1/task_1_tumor_vs_normal_CLAM_50_s1/split_{fold}_results.pkl', 'rb') as file:
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
            


for i in tp:                
    model = CLAM_SB()
    state_dict = torch.load(f'/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting_7,2,1/task_1_tumor_vs_normal_CLAM_50_s1/s_{fold}_checkpoint.pt')
    model.load_state_dict(state_dict)    

    serial_num=i

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

    top_scores, top_indices = torch.topk(A, k=4, largest=True, dim=1)

    bottom_scores, bottom_indices = torch.topk(A, k=4, largest=False, dim=1)



    data = pd.read_csv(f'/home/minkyoon/crohn/csv/clam/remission/csvandpt/{serial_num}.csv')

    # top_indices를 numpy 배열로 변환
    top_indices = top_indices.cpu().numpy()

    # 각 인덱스에 대한 이미지 표시
    for index in top_indices[0][:2]:
        # npy 파일 경로 가져오기
        npy_path = data.iloc[index]['filepath']
        
        # npy 파일 로드
        image = np.load(npy_path)
        
        score_str = "{:.4f}".format(top_scores[0][0].item()) # Score with 4 decimal places
        filename = f'fold{fold}_serial{i}_score{score_str}_index{index}.png'

        
        # 이미지 표시
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.axis('off') # 격자와 축 제거
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # 여백 제거
        fig.savefig(f'./image/tp/{filename}', bbox_inches='tight', pad_inches=0)
        plt.show()
        
        new_row_df = pd.DataFrame({
        'filename': [filename],
        'score': [top_scores[0]],
        'index': [index],
        'fold_number': [fold],
        'serial_number': [i]
        })

    # Append the new row to the existing DataFrame
        image_info_df = pd.concat([image_info_df, new_row_df], ignore_index=True)
        
    print(top_scores, top_indices)
        
    image_info_df.to_csv('./results/image_info_tp.csv', index=False)



    # 각 인덱스에 대한 이미지 표시
    # for index in bottom_indices[0]:
    #     # npy 파일 경로 가져오기
    #     npy_path = data.iloc[index]['filepath']
        
    #     # npy 파일 로드
    #     image = np.load(npy_path)
        
    #     # 이미지 표시
    #     plt.imshow(image, cmap='gray')
    #     plt.show()