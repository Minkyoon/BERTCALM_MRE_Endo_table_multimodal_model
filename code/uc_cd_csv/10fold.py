import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np

# 데이터 로드
file_path = '/home/minkyoon/2024_CLAM_MUTLIMODAL_uc_cd/data/processed/Crohn_label.csv'
df = pd.read_csv(file_path)

# 저장할 디렉터리 경로
save_dir = '/home/minkyoon/2024_CLAM_MUTLIMODAL_uc_cd/data/10fold_uc_cd'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 디렉터리가 없으면 생성

# StratifiedKFold 인스턴스 생성 (10 folds)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 각 fold에 대해 실행
for fold, (train_val_idx, test_idx) in enumerate(skf.split(X=df, y=df['Crohn disease'])):
    print(f"Processing Fold {fold+1}")
    # 테스트 세트 설정
    test_set = df.iloc[test_idx]['serial_number'].apply(lambda x: '' if pd.isna(x) else int(x))
    
    # 훈련+검증 세트 준비
    train_val_set = df.iloc[train_val_idx]
    
    # 훈련+검증 세트를 다시 StratifiedKFold를 사용해 분할 (나머지를 7:2로)
    skf_train_val = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
    val_indices = []
    for inner_fold, (train_idx, val_idx) in enumerate(skf_train_val.split(X=train_val_set, y=train_val_set['Crohn disease'])):
        if inner_fold < 2:  # 첫 2개의 fold를 검증 세트로 사용하여 전체 비율을 7:2:1로 조정
            val_indices.extend(train_val_idx[val_idx])
        else:
            break

    # 최종 훈련, 검증 세트 인덱스 정의
    train_indices = list(set(train_val_idx) - set(val_indices))
    train_set = df.iloc[train_indices]['serial_number'].apply(lambda x: '' if pd.isna(x) else int(x))
    val_set = df.iloc[val_indices]['serial_number'].apply(lambda x: '' if pd.isna(x) else int(x))

    # 결과 DataFrame 생성
    max_length = max(len(train_set), len(val_set), len(test_set))
    result_df = pd.DataFrame({
        'train': pd.Series(train_set.values).reindex(range(max_length), fill_value=''),
        'val': pd.Series(val_set.values).reindex(range(max_length), fill_value=''),
        'test': pd.Series(test_set.values).reindex(range(max_length), fill_value='')
    })

    # 결과 저장
    result_df.to_csv(os.path.join(save_dir, f'fold_{fold+1}.csv'), index=False)

print("All folds processed and saved.")

