import pandas as pd

# CSV 파일 로드
file_path = '/home/minkyoon/2023_CLAM_MUTLIMODAL/code/uc_cd_csv/df_dx_lab_20240110 (1).csv'
df = pd.read_csv(file_path)

# 'Ulcerative colitis', 'Crohn disease', 'IBD', 'date', 'ID' 열을 제외하고 나머지 열 선택
df = df.drop(['Ulcerative colitis', 'Crohn disease', 'IBD', 'date', 'ID'], axis=1)

# 'serial_number' 열을 별도로 저장
serial_number_col = df[['serial_number']]

# 원본 데이터프레임에서 'serial_number' 열을 제거
df.drop('serial_number', axis=1, inplace=True)

# 'serial_number' 열을 데이터프레임의 앞부분에 붙여넣기
df = pd.concat([serial_number_col, df], axis=1)
df=df.iloc[:,:-1]

# 변경된 DataFrame을 새로운 CSV 파일로 저장
save_path = 'newcsv.csv'
df.to_csv(save_path, index=False)

print("File has been processed and saved.")