import numpy as np
from joblib import dump, load
import warnings
import matplotlib.pyplot as plt
import mne
from time import time
from mne import Epochs
from mne.datasets import eegbci
from mne.io.edf import read_raw_edf
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

mne.set_log_level('WARNING')


def removing_artifacts(raw, n_components, method='fastica'):
    if method == 'infomax' or method == 'picard':
        fit_params = {"extended": True}  # chose parameters for different methods
    raw_tmp = raw.copy()
    ica = ICA(n_components=n_components, method='fastica', random_state=21, fit_params=None)
    ica.fit(raw_tmp, picks=picks)
    ica.plot_components(picks=range(20), inst=raw_tmp)

    eog_indices, scores = ica.find_bads_eog(raw, ch_name='Fpz', threshold=1.5)
    ica.plot_scores(scores, exclude=eog_indices)
    ica.exclude.extend(eog_indices)
    raw_tmp = ica.apply(raw_tmp, n_pca_components=n_components, exclude=ica.exclude)
    print('Bad components to remove:', ica.exclude)
    plt.show()
    return raw_tmp


def get_clear(data):
    # raw.notch_filter(60, picks=picks) # Removing power-line noise can be done with a Notch filter, directly on the Raw object, specifying an array of frequency to be cut off
    # data = removing_artifacts(data, 20, 'fastica')
    data = data.filter(7., 40., fir_design='firwin', skip_by_annotation='edge')  # Apply band-pass filter

    return data


def get_raw(testee, task):
    raw_fnames = eegbci.load_data(testee, task)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw.rename_channels(lambda x: x.strip('.'))
    return raw


def main_preproc():
    tmin, tmax = -1., 4.  # avoid classification of evoked responses by using epochs that start 1s after cue onset.
    execution = [5, 9, 13]
    imagery = [6, 10, 14]
    raw_execution = []
    raw_imagery = []
    raw_raws = []
    event_id = dict(hands=2, feet=3)
    raw = get_raw(1, imagery)
    clear = get_clear(raw)
    events, _ = mne.events_from_annotations(data, event_id=dict(T1=2, T2=3))
    picks = mne.pick_types(clear.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
    epochs = Epochs(clear, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2
    epochs_train


if __name__ == '__main__':
    main_preproc()
