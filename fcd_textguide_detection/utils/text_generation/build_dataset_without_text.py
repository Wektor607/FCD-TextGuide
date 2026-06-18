import csv
import glob
import os

import pandas as pd

if __name__ == "__main__":
    comb_root = "/data/input/data4sharing/meld_combats"
    out_path  = "/data/input/preprocessed/Dataset_without_text.csv"

    for name in ["MELD_splits.csv", "BONN_splits.csv"]:
        reports_csv = f"/data/input/preprocessed/{name}"
        reports_df  = pd.read_csv(reports_csv, dtype=str)

        with open(out_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['DATA_PATH', 'ROI_PATH'])

            for _, row in reports_df.iterrows():
                sid = row['subject_id']

                pattern_h5 = os.path.join(comb_root, f"{sid}_*featurematrix*.hdf5")
                h5_list = glob.glob(pattern_h5)
                if not h5_list:
                    print(f"HDF5 not found for {sid}")
                    continue

                data_path = h5_list[0]
                roi_path  = '' if "control" in sid else data_path

                writer.writerow([data_path, roi_path])
