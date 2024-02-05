import pandas as pd

df=pd.read_csv('/home/minkyoon/2023_CLAM_MUTLIMODAL/code/mre_endo_match/df_dx_lab_20240110 (1).csv')

df=df[['ID','serial_number']]

df.to_csv('endo_id_serial.csv', index=False)



import os
import csv

# 디렉토리 경로 설정
directory_path = "/home/jsy/2023_CD_MRE/data/raw/박유랑(2023200145)_MRE"

# CSV 파일 저장 경로
csv_file_path = "output.csv"

# 결과를 저장할 리스트 초기화
data = []

# 디렉토리 내의 모든 폴더를 순회
for folder_name in os.listdir(directory_path):
    if os.path.isdir(os.path.join(directory_path, folder_name)):
        # 폴더 이름을 '_' 기준으로 분리
        parts = folder_name.split('_')
        if len(parts) >= 3:
            # serial_number와 patient_id 추출
            serial_number = parts[0]
            patient_id = parts[1]
            # 결과 리스트에 추가
            data.append({'MRE_ID': patient_id, 'MRE_serial_number': serial_number})

# CSV 파일로 결과 저장
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['MRE_ID', 'MRE_serial_number'])
    writer.writeheader()
    writer.writerows(data)

print(f"Data saved to {csv_file_path}")



import pandas as pd

# CSV 파일 경로
endo_csv_path = "/home/minkyoon/2023_CLAM_MUTLIMODAL/code/mre_endo_match/endo_id_serial.csv"
mre_csv_path = "/home/minkyoon/2023_CLAM_MUTLIMODAL/code/mre_endo_match/output.csv"

# 결과 파일 경로
merged_csv_path = "/home/minkyoon/2023_CLAM_MUTLIMODAL/code/mre_endo_match/merged_output.csv"

# CSV 파일 읽기
endo_df = pd.read_csv(endo_csv_path)
mre_df = pd.read_csv(mre_csv_path)

# ID를 기준으로 병합
merged_df = pd.merge(endo_df, mre_df, left_on='ENDO_ID', right_on='MRE_ID', how='inner')

# 결과 저장
merged_df.to_csv(merged_csv_path, index=False)

print(f"Merged data saved to {merged_csv_path}")



import pandas as pd

# CSV 파일 경로
endo_csv_path = "/home/minkyoon/2023_CLAM_MUTLIMODAL/code/mre_endo_match/endo_id_serial.csv"
mre_csv_path = "/home/minkyoon/2023_CLAM_MUTLIMODAL/code/mre_endo_match/output.csv"

# 결과 파일 경로
merged_csv_path = "/home/minkyoon/2023_CLAM_MUTLIMODAL/code/mre_endo_match/merged_output2.csv"

# CSV 파일 읽기
endo_df = pd.read_csv(endo_csv_path)
mre_df = pd.read_csv(mre_csv_path)

# ID를 기준으로 병합
merged_df = pd.merge(endo_df, mre_df, left_on='ENDO_ID', right_on='MRE_ID', how='inner')

# MRE_ID별로 MRE_serial_number가 가장 낮은 항목만 남김
filtered_df = merged_df.groupby('MRE_ID').apply(lambda x: x.nsmallest(1, 'MRE_serial_number')).reset_index(drop=True)

# 결과 저장
filtered_df.to_csv(merged_csv_path, index=False)

print(f"Merged and filtered data saved to {merged_csv_path}")


import pandas as pd

# CSV 파일 경로
endo_label_csv_path = "/home/minkyoon/2023_CLAM_MUTLIMODAL/code/mre_endo_match/endo_label.csv"
merged_output_csv_path = "/home/minkyoon/2023_CLAM_MUTLIMODAL/code/mre_endo_match/merged_output2.csv"

# 결과 파일 경로
final_csv_path = "/home/minkyoon/2023_CLAM_MUTLIMODAL/code/mre_endo_match/final_output.csv"

# CSV 파일 읽기
endo_label_df = pd.read_csv(endo_label_csv_path)
merged_output_df = pd.read_csv(merged_output_csv_path)

# 모든 수를 정수로 변환
endo_label_df = endo_label_df.astype(int)
merged_output_df = merged_output_df.astype(int)

# ENDO_serial_number를 기준으로 병합
final_df = pd.merge(merged_output_df, endo_label_df, left_on='ENDO_serial_number', right_on='serial_number', how='left')

# 불필요한 열 제거
final_df.drop('serial_number', axis=1, inplace=True)

# 결과 저장
final_df.to_csv(final_csv_path, index=False)

print(f"Merged and formatted data saved to {final_csv_path}")
