import csv
import glob
import os

import pandas as pd

if __name__ == "__main__":
    reports_csv = "/data/input/preprocessed/meld_files/all_augmented_reports.csv"
    comb_root   = "/data/input/data4sharing/meld_combats"
    out_path    = "/data/input/preprocessed/MELD_BONN_dataset_augmented_final.csv"

    reports_df = pd.read_csv(reports_csv, dtype=str)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['DATA_PATH', 'ROI_PATH', 'harvard_oxford', 'aal'])

        for _, row in reports_df.iterrows():
            sid  = row['subject_id']
            harv = row['report_harvard_oxford']
            aal  = row['report_aal']

            pattern_h5 = os.path.join(comb_root, f"{sid}_*featurematrix*.hdf5")
            h5_list = glob.glob(pattern_h5)
            if not h5_list:
                print(f"HDF5 not found for {sid}")
                continue

            data_path = h5_list[0]
            roi_path  = '' if "control" in sid else data_path

            writer.writerow([data_path, roi_path, harv, aal])
