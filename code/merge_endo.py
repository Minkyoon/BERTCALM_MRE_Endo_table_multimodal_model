import pandas as pd

extracted_data_path = "/home/minkyoon/2023_CLAM_MUTLIMODAL/code/edno.csv"
new_output_path = "/home/minkyoon/2023_CLAM_MUTLIMODAL/data/processed/label.csv"


# 데이터 파일 로드
extracted_data_df = pd.read_csv(extracted_data_path)
new_output_df = pd.read_csv(new_output_path)

# 데이터 형식 일치시키기 (예: 모두 문자열로 변환)
extracted_data_df['serial_number'] = extracted_data_df['serial_number'].astype(int)
new_output_df['slide_id'] = new_output_df['slide_id'].astype(int)

# 'ID'와 'slide_id'를 기준으로 두 데이터프레임을 병합 (inner join 사용)
merged_df = pd.merge(extracted_data_df, new_output_df, left_on='serial_number', right_on='slide_id', how='inner')

# 필요없는 'case_id'와 'slide_id' 열을 삭제
merged_df.drop(['case_id', 'slide_id'], axis=1, inplace=True)

# 결과를 새로운 CSV 파일로 저장
merged_df.to_csv("endo22.csv", index=False)
