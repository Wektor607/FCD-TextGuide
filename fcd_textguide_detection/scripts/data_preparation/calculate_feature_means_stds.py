## Script to calculate the mean and standard deviation of the MELD cohort surface-based features
## Parameters are saved and used for normalisation
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import glob
import json

import numpy as np

from meld_graph.data_preprocessing import Preprocess as Prep
from meld_graph.meld_cohort import MeldCohort, MeldSubject
from meld_graph.paths import BASE_PATH


def load_config(config_file):
    """load config.py file and return config object"""
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config


class StatsRecorder:
    def __init__(self, ndimensions=None, data=None):
        """
        If data != None, keep the original code unchanged.
        If data == None but ndimensions is given, initialise mean and std to zeros.
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions = data.shape[1]
        else:
            # If ndimensions is set, initialise mean/std to zeros of the required length
            if ndimensions is None:
                raise ValueError("If data=None, ndimensions must be provided.")
            self.ndimensions = ndimensions
            self.nobservations = 0
            # immediately create zero arrays of the required length
            self.mean = np.zeros(self.ndimensions, dtype=np.float32)
            self.std  = np.zeros(self.ndimensions, dtype=np.float32)

    def update(self, data):
        """
        When update is called for the first time, data has shape (n_obs, ndimensions).
        If self.nobservations == 0, calling __init__(data) will overwrite mean/std.
        """
        if self.nobservations == 0:
            # Since __init__ accepts data, it will overwrite mean and std itself.
            self.__init__(data=data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")
            newmean = data.mean(axis=0)
            newstd  = data.std(axis=0)
            m = self.nobservations * 1.0
            n = data.shape[0]
            tmp = self.mean.copy()
            # update mean correctly
            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            # update squared std (formula for combining standard deviations)
            var_combined = (
                m/(m+n)* (self.std**2)
                + n/(m+n)* (newstd**2)
                + m*n/(m+n)**2 * (tmp - newmean)**2
            )
            self.std = np.sqrt(var_combined)
            self.nobservations += n


if __name__ == "__main__":

    config = load_config("/meld_graph/scripts/config_files/example_experiment_config.py")
    cohort = MeldCohort(
        hdf5_file_root="{site_code}_featurematrix_combat.hdf5",
        dataset=None,
        data_dir=BASE_PATH
    )
    prep = Prep(cohort=cohort, params=config.data_parameters)
    # subject_ids = cohort.get_subject_ids(group="both")
    subject_ids = sorted(set(
        os.path.basename(p).split('_')[0]
        for p in glob.glob(os.path.join(BASE_PATH, "*.hdf5"))
    ))[1:]
    # two batch-wise stats recorders, one with flair, one without
    flair_mask = np.zeros(len(config.data_parameters["features"]), dtype=bool)
    for fi, feature in enumerate(config.data_parameters["features"]):
        if "FLAIR" in feature:
            flair_mask[fi] = 1

    n_nonflair = np.sum(~flair_mask)
    # Number of “FLAIR” features:
    n_flair    = np.sum(flair_mask)

    mean_std       = StatsRecorder(ndimensions=n_nonflair)
    mean_std_flair = StatsRecorder(ndimensions=n_flair)
    print(subject_ids)
    for si, subj_id in enumerate(subject_ids):
        if si % 30 == 0:
            print(f"{100*si/len(subject_ids)}% complete")

        subject_data_list = prep.get_data_preprocessed(
            subject=subj_id,
            features=config.data_parameters["features"],
            lobes=config.data_parameters["lobes"],
            lesion_bias=False,
        )
        
        for hemisphere_data in subject_data_list:
            feat_hem = hemisphere_data["features"].T
            feat_hem = feat_hem[:, cohort.cortex_mask]
            feat_hem_nf = feat_hem[~flair_mask]
            mean_std.update(feat_hem_nf.T)
            
            if np.sum(feat_hem[6]) != 0:
                feat_hem_f = feat_hem[flair_mask]
                mean_std_flair.update(feat_hem_f.T)

    means_stds = np.zeros((2, len(config.data_parameters["features"])))
    print(dir(mean_std_flair))   
    means_stds[0, flair_mask] = mean_std_flair.mean
    means_stds[1, flair_mask] = mean_std_flair.std
    means_stds[0, ~flair_mask] = mean_std.mean
    means_stds[1, ~flair_mask] = mean_std.std

    mean_stds_dict = {}
    for fi, feature in enumerate(config.data_parameters["features"]):
        mean_stds_dict[feature] = {}
        mean_stds_dict[feature]["mean"] = means_stds[0, fi]
        mean_stds_dict[feature]["std"] = means_stds[1, fi]

    means_stds = np.zeros((2,len(config.data_parameters['features'])))
    means_stds[0,flair_mask] = mean_std_flair.mean
    means_stds[1,flair_mask] = mean_std_flair.std
    means_stds[0,~flair_mask] = mean_std.mean
    means_stds[1,~flair_mask] = mean_std.std

    mean_stds_dict={
                   }
    for fi,feature in enumerate(config.data_parameters['features']):
        mean_stds_dict[feature]={}
        mean_stds_dict[feature]['mean'] = means_stds[0,fi]
        mean_stds_dict[feature]['std'] = means_stds[1,fi]

    with open('/data/feature_means_no_combat.json', 'w') as fp:
        json.dump(mean_stds_dict, fp)

    print('HERE')