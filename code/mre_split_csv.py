import pandas as pd

# mre_endo_merged.csv 파일을 불러옵니다.
mre_endo_df = pd.read_csv('/home/minkyoon/2023_CLAM_MUTLIMODAL/data/mre_csv/mre_endo_merged.csv')

# serial_number_endo를 기준으로 serial_number_mri를 찾는 사전을 생성합니다.
endo_to_mri_mapping = dict(zip(mre_endo_df['serial_number_endo'], mre_endo_df['serial_number_mri']))

# splits 파일들을 처리합니다.
for i in range(10):
    # splits 파일을 불러옵니다.
    split_df = pd.read_csv(f'/home/minkyoon/2023_CLAM_MUTLIMODAL/data/10fold_csv2_copy/splits_{i}.csv')

    # 각 열(train, val, test)에서 serial_number_endo를 serial_number_mri로 변환합니다.
    for col in ['train', 'val', 'test']:
        split_df[col] = split_df[col].map(endo_to_mri_mapping)

    # 변환된 데이터를 새로운 파일로 저장합니다.
    split_df.to_csv(f'/home/minkyoon/2023_CLAM_MUTLIMODAL/data/mre_10fold_csv/splits_{i}.csv', index=False)







## clam csv로 바꾸기


import os

# Directory path
directory_path = "/home/minkyoon/2023_CLAM_MUTLIMODAL/data/mre_10fold_csv"

# List to hold file paths
filepaths = []

# Using os.listdir to get all file names in the directory
for filename in os.listdir(directory_path):
    # Check if the file is a CSV file
    if filename.endswith(".csv"):
        # Adding the full file path to the list
        filepaths.append(os.path.join(directory_path, filename))




for filepath in filepaths:
    # csv 파일 로드
    df = pd.read_csv(filepath)

    # 변경된 csv 파일 저장 (기존 파일 덮어쓰기)
    df.to_csv(filepath, index=False, float_format='%.0f')