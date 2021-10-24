import argparse
import matplotlib.pyplot as plt
import mne
import os
from time import time
from mne import Epochs
from mne.datasets import eegbci, sample
from mne.io.edf import read_raw_edf
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

mne.set_log_level('WARNING')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='eegbci')
    parser.add_argument('--testee', default=1)
    parser.add_argument('--task', default='imagery')
    parser.add_argument('--bodypart', default='feet')
    parser.add_argument('--viz', default=0)
    args = parser.parse_args()
    return args.__dict__


# def removing_artifacts(raw, n_components, method='fastica'):
#     if method == 'infomax' or method == 'picard':
#         fit_params = {"extended": True}  # chose parameters for different methods
#     raw_tmp = raw.copy()
#     picks = mne.pick_types(raw_tmp.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
#     ica = ICA(n_components=n_components, method='fastica', random_state=21, fit_params=None)
#     ica.fit(raw_tmp, picks=picks)
#     ica.plot_components(picks=range(20), inst=raw_tmp)
#
#     eog_indices, scores = ica.find_bads_eog(raw, ch_name='Fpz', threshold=1.5)
#     ica.plot_scores(scores, exclude=eog_indices)
#     ica.exclude.extend(eog_indices)
#     raw_tmp = ica.apply(raw_tmp, n_pca_components=n_components, exclude=ica.exclude)
#     print('Bad components to remove:', ica.exclude)
#     plt.show()
#     return raw_tmp


def get_clear(raw):
    raw_tmp = raw.copy()
    #raw_tmp = removing_artifacts(raw_tmp, 20, 'fastica')
    data = raw_tmp.filter(7., 40., fir_design='firwin', skip_by_annotation='edge')  # Apply band-pass filter
    return data


def get_raw(testee, task, taskname, data_name):
    print('... Parsing of', taskname, 'runs of', data_name)
    raw_fnames = eegbci.load_data(testee, task)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw.rename_channels(lambda x: x.strip('.'))
    return raw

def plot_object(object, name, add=1):
    if add == 1:
        object.plot()
    else:
        object.plot_psd(average=False)
    plt.savefig(name)
    plt.close()


def main_preproc(data_name, testee, task, viz):
    if data_name == 'eegbci':
        tmin, tmax = -1., 4.  # avoid classification of evoked responses by using epochs that start 1s after cue onset.
        execution_fh = [5, 9, 13]
        imagery_fh = [6, 10, 14]
        execution_lr = [3, 7, 11]
        imagery_lr = [4, 8, 12]

        event_id = dict(hands=2, feet=3)
        biosemi_montage = mne.channels.make_standard_montage('biosemi64')
        if task == 'imagery feet-hand':
            raw = get_raw(testee, imagery_fh, task, data_name)
        elif task == 'execution feet-hand':
            raw = get_raw(testee, execution_fh, task, data_name)
        elif task == 'imagery left-right':
            raw = get_raw(testee, execution_lr, task, data_name)
        elif task == 'execution left-right':
            raw = get_raw(testee, imagery_lr, task, data_name)
        else:
            exit()
        clear = get_clear(raw)
        if viz == True:
            print('Vizualization')
            plot_object(biosemi_montage, 'plots/1.biosemi_montage.png')
            plot_object(raw, 'plots/2.raw_time.png')
            plot_object(clear, 'plots/3.clear_time.png')
            plot_object(raw, 'plots/4.raw_frequency.png', 2)
            plot_object(clear, 'plots/5.clear_frequency.png', 2)
        events, _ = mne.events_from_annotations(clear, event_id=dict(T1=2, T2=3))
        picks = mne.pick_types(clear.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
        epochs = Epochs(clear, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    elif data_name == 'sample':
        print('... Parsing of sample dataset')
        data_path = sample.data_path()
        raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
        event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
        tmin, tmax = -0.1, 0.3
        event_id = dict(aud_l=1, vis_l=3)
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        raw.filter(1, 20, fir_design='firwin')
        events = mne.read_events(event_fname)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                               exclude='bads')
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                            picks=picks, baseline=None, preload=True,
                            verbose=False)
    else:
        raise NotImplementedError('No code for such dataset')
    return epochs


if __name__ == '__main__':
    args = parse_args()
    main_preproc(data_name=args['data'], testee=args['testee'], task=args['task'], viz=args['viz'])
