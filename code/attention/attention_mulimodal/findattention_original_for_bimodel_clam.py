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
            

tp=[1106]
for i in tp:                
    model = CLAM_SB()
    model.set_classifier(49)
    state_dict = torch.load(f'/home/minkyoon/first/CLAM/results/remission_stratified_for_recommend_setting_7,2,1_learningragechange/task_1_tumor_vs_normal_CLAM_50_s1/s_{fold}_checkpoint.pt')
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

    top_scores, top_indices = torch.topk(A, k=7, largest=True, dim=1)

    bottom_scores, bottom_indices = torch.topk(A, k=7, largest=False, dim=1)

    sorted_indices = torch.argsort(A, descending=True, dim=1)

    data = pd.read_csv(f'/home/minkyoon/crohn/csv/clam/remission/csvandpt/{serial_num}.csv')

    # top_indices를 numpy 배열로 변환
    top_indices = top_indices.cpu().numpy()

    # 각 인덱스에 대한 이미지 표시
    for index in top_indices[0][:7]:
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
        output_dir=f'./image/tn/'
        if not os.path.exists('./image/tn/'):
            os.makedirs(output_dir , exist_ok=True)
        fig.savefig(f'./image/tn/{filename}', bbox_inches='tight', pad_inches=0)
        plt.show()
        
        new_row_df = pd.DataFrame({
        'filename': [filename],
        'score': [top_scores[0:]],
        'index': [index],
        'fold_number': [fold],
        'serial_number': [i]
        })

    # Append the new row to the existing DataFrame
        image_info_df = pd.concat([image_info_df, new_row_df], ignore_index=True)
        
    print(top_scores, top_indices)
        
    image_info_df.to_csv('./results/image_info_tp.csv', index=False)


    bottom_indices = bottom_indices.cpu().numpy()
    for index in bottom_indices[0][:7]:
        # npy 파일 경로 가져오기
        npy_path = data.iloc[index]['filepath']
        
        # npy 파일 로드
        image = np.load(npy_path)
        
        score_str = "{:.4f}".format(bottom_scores[0][0].item()) # Score with 4 decimal places
        filename = f'bottom_fold{fold}_serial{i}_score{score_str}_index{index}.png'

        
        # 이미지 표시
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.axis('off') # 격자와 축 제거
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # 여백 제거
        fig.savefig(f'./image/tn/{filename}', bbox_inches='tight', pad_inches=0)
        plt.show()
        
        new_row_df = pd.DataFrame({
        'filename': [filename],
        'score': [bottom_scores[0:]],
        'index': [index],
        'fold_number': [fold],
        'serial_number': [i]
        })

    # Append the new row to the existing DataFrame
        image_info_df = pd.concat([image_info_df, new_row_df], ignore_index=True)
        
    print(bottom_scores, bottom_indices)
        
    image_info_df.to_csv('./results/bottom_image_info_tp.csv', index=False)
    
    
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