from torch.utils.data import Dataset
import pandas as pd
import torch
import os


#mre 내시경 매칭
def match_mre_endo(mre_endo_csv):
    mre_endo_data = pd.read_csv(mre_endo_csv)
    matched_data = {}
    for index, row in mre_endo_data.iterrows():
        endo_id = row['serial_number_endo']
        mre_id = row['serial_number_mri']
        matched_data[endo_id] = mre_id
    return matched_data



class PTFilesDataset(Dataset):
    def __init__(self, pt_directory, mre_directory, label_csv, tabular_csv, ids=None, mre_endo_csv=None):
        self.pt_directory = pt_directory
        self.labels = pd.read_csv(label_csv)
        self.labels.set_index("slide_id", inplace=True)
        self.tabular_data = pd.read_csv(tabular_csv).drop(columns=['date'])
        self.tabular_data.set_index("slide_id", inplace=True)
        self.tabular_data.fillna(0, inplace=True)
        self.ids = ids
        self.mre_directory = mre_directory
        if mre_endo_csv:
            self.mre_endo_match = match_mre_endo(mre_endo_csv)
        else:
            self.mre_endo_match = {}

    def __len__(self):
        if self.ids is not None:
            return len(self.ids)
        return len(os.listdir(self.pt_directory))

    def __getitem__(self, idx):
        if self.ids is not None:
            slide_id = self.ids[idx]
        else:
            file_name = os.listdir(self.pt_directory)[idx]
            slide_id = int(file_name.split('.')[0])

        label = self.labels.loc[slide_id, "label"]
        file_path = os.path.join(self.pt_directory, f"{slide_id}.pt")
        data = torch.load(file_path)
        
        if slide_id in self.mre_endo_match:
            mre_id = self.mre_endo_match[slide_id]
            mre_file_path = os.path.join(self.mre_directory, f"{mre_id}.pt")
            mre_data = torch.load(mre_file_path)
        # else:
        #     mre_data = torch.zeros_like(data)  # 또는 적절한 기본값
        
        # slide_id 제거를위애 마지막열 제거    
        #tabular_features = self.tabular_data.loc[slide_id].iloc[:-1].values
        tabular_features = self.tabular_data.loc[slide_id].values
        tabular_features = torch.tensor(tabular_features, dtype=torch.float32)

        return data, mre_data, tabular_features, label

    
