import os
import pandas as pd

# 폴더 경로
folder_path = "/home/jsy/2023_CD_MRE/data/raw/박유랑(2023200145)_MRE"

# 폴더 내의 모든 항목들을 리스트로 가져옴
items = os.listdir(folder_path)

# 데이터 프레임에 들어갈 데이터를 담을 리스트
data = []

# 각 폴더 이름을 분석하여 ID와 serial number를 추출
for item in items:
    if os.path.isdir(os.path.join(folder_path, item)):
        parts = item.split('_')
        serial_number = parts[0]
        ID = parts[1]
        data.append({"ID": ID, "serial_number": serial_number})

# 데이터 프레임 생성
df = pd.DataFrame(data)

# 데이터 프레임을 csv 파일로 저장
csv_file_path = "extracted_data.csv"  # 적절한 경로로 변경하세요
df.to_csv(csv_file_path, index=False)




import pandas as pd

# Load the CSV file
file_path = '/home/minkyoon/2023_CLAM_MUTLIMODAL/data/mre_csv/mre_endo_merged.csv'
data = pd.read_csv(file_path)

# Select the desired columns
selected_data = data[['serial_number_endo', 'label_endo']]

# Save the selected data to a new CSV file
new_file_path = 'endo_csv.csv'
selected_data.to_csv(new_file_path, index=False)

new_file_path



"import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

# 데이터 로드
data = pd.read_csv('/home/minkyoon/2023_CLAM_MUTLIMODAL/code/endo_csv.csv')

# StratifiedKFold 초기화
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 저장할 디렉토리 생성
save_dir = '/home/minkyoon/2023_CLAM_MUTLIMODAL/data/10fold_csv2'
os.makedirs(save_dir, exist_ok=True)

# 각 폴드에 대해 반복
for fold, (train_val_index, test_index) in enumerate(skf.split(data, data['label_endo'])):
    # 트레이닝/검증 세트와 테스트 세트 분할
    train_val_fold = data.iloc[train_val_index]
    test_fold = data.iloc[test_index]

    # 트레이닝/검증 세트를 7:2 비율로 분할
    train_fold = train_val_fold.sample(frac=7/9, random_state=42)  # 약 7:2 비율
    val_fold = train_val_fold.drop(train_fold.index)

    # 각 세트의 시리얼 넘버 추출
    train_serials = train_fold['serial_number_endo'].tolist()
    val_serials = val_fold['serial_number_endo'].tolist()
    test_serials = test_fold['serial_number_endo'].tolist()

    # 최대 길이 계산
    max_length = max(len(train_serials), len(val_serials), len(test_serials))

    # 각 세트의 길이를 최대 길이에 맞추기 위해 None으로 채움
    train_serials.extend([None] * (max_length - len(train_serials)))
    val_serials.extend([None] * (max_length - len(val_serials)))
    test_serials.extend([None] * (max_length - len(test_serials)))

    # 데이터프레임 생성
    fold_df = pd.DataFrame({
        'train': train_serials,
        'val': val_serials,
        'test': test_serials
    })

    # CSV 파일로 저장
    fold_path = os.path.join(save_dir, f'splits_{fold}.csv')
    fold_df.to_csv(fold_path, index=False)

print("10-fold splitting and saving completed.")
"





from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import numpy as np
import pandas as pd
import os

# 데이터 로드
data = pd.read_csv('/home/minkyoon/2023_CLAM_MUTLIMODAL/code/endo_csv.csv')

# StratifiedKFold 및 StratifiedShuffleSplit 초기화
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
sss = StratifiedShuffleSplit(n_splits=1, test_size=2/9, random_state=42) # 훈련:검증 = 7:2

# 저장할 디렉토리 생성
save_dir = '/home/minkyoon/2023_CLAM_MUTLIMODAL/data/10fold_csv2'
os.makedirs(save_dir, exist_ok=True)

# 각 폴드에 대해 반복
for fold, (train_val_index, test_index) in enumerate(skf.split(data, data['label_endo'])):
    # 트레이닝/검증 세트와 테스트 세트 분할
    train_val_fold = data.iloc[train_val_index]
    test_fold = data.iloc[test_index]

    # 훈련 및 검증 세트를 계층화하여 분할
    for train_index, val_index in sss.split(train_val_fold, train_val_fold['label_endo']):
        train_fold = train_val_fold.iloc[train_index]
        val_fold = train_val_fold.iloc[val_index]

    # 시리얼 넘버 추출
    train_serials = train_fold['serial_number_endo'].tolist()
    val_serials = val_fold['serial_number_endo'].tolist()
    test_serials = test_fold['serial_number_endo'].tolist()

    max_length = max(len(train_serials), len(val_serials), len(test_serials))

    # 각 세트의 길이를 최대 길이에 맞추기 위해 None으로 채움
    train_serials.extend([None] * (max_length - len(train_serials)))
    val_serials.extend([None] * (max_length - len(val_serials)))
    test_serials.extend([None] * (max_length - len(test_serials)))

    # 데이터프레임 생성 및 저장
    fold_df = pd.DataFrame({
        'train': train_serials,
        'val': val_serials,
        'test': test_serials
    })

    fold_path = os.path.join(save_dir, f'splits_{fold}.csv')
    fold_df.to_csv(fold_path, index=False)

print("10-fold splitting with stratified train/val splits completed.")
