import pandas as pd

# remission_under_10.csv 파일을 불러옵니다.
remission_df = pd.read_csv('/home/minkyoon/2023_CLAM_MUTLIMODAL/data/processed/remission_under_10.csv')

# 'date' 열을 제거합니다.
remission_df.drop('date', axis=1, inplace=True)

# label.csv 파일을 불러옵니다.
label_df = pd.read_csv('/home/minkyoon/2023_CLAM_MUTLIMODAL/data/processed/label.csv')

# 'slide_id'를 기준으로 'label' 값을 매핑합니다.
label_mapping = dict(zip(label_df['slide_id'], label_df['label']))
remission_df['label'] = remission_df['slide_id'].map(label_mapping)

# 결과를 xgboost_main.csv 파일로 저장합니다.
remission_df.to_csv('/home/minkyoon/xgboost/xgboost_main.csv', index=False)
