import pandas as pd



df= pd.read_csv('/home/minkyoon/2023_CLAM_MUTLIMODAL/data/make_multi_csv/accesion_lab_PCDAI_serial_20230628_1.csv')

df2=df[['ID', 'serial_number']]

df2.to_csv('edno.csv', index=False)


