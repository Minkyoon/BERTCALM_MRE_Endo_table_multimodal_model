import pandas as pd

df=pd.read_csv('/home/minkyoon/2023_CLAM_MUTLIMODAL/code/uc_cd_csv/df_dx_lab_20240110 (1).csv')

df=df[['serial_number','Ulcerative colitis','Crohn disease']]


filtered_df = df[((df['Ulcerative colitis'] == 0.0) & (df['Crohn disease'] == 0.0)) | 
                 ((df['Ulcerative colitis'] == 1.0) & (df['Crohn disease'] == 1.0))]


filtered_df = df[df['Ulcerative colitis'] != df['Crohn disease']]

filtered_df =df[['ID','serial_number', 'Crohn disease']]
filtered_df = filtered_df.astype(int)
filtered_df.to_csv('Crohn_label.csv', index=False)