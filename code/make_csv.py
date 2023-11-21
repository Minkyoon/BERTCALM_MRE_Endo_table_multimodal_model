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
