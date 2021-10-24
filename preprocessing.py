import argparse
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci, sample
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne import Epochs
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf


mne.set_log_level('WARNING')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='eegbci')
    parser.add_argument('--testee', default=1)
    parser.add_argument('--task', default='imagery')
    parser.add_argument('--bodypart', default='feet')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--clear', action="store_true")
    args = parser.parse_args()
    return args.__dict__


def removing_artifacts(raw, n_components):
    raw_tmp = raw.copy().filter(l_freq=1., h_freq=None)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    ica = ICA(n_components=n_components, max_iter='auto', random_state=21)
    ica.fit(raw_tmp, picks=picks)
    ica.plot_components(picks=range(20), inst=raw_tmp)

    eog_indices, scores = ica.find_bads_eog(raw_tmp, ch_name='Fpz', threshold=1.5)
    ica.plot_scores(scores, exclude=eog_indices)
    ica.exclude.extend(eog_indices)
    raw_tmp = ica.apply(raw_tmp, n_pca_components=n_components, exclude=ica.exclude)
    print('Bad components to remove:', ica.exclude)
    plt.show()
    return raw_tmp

def removing_artifacts2(raw, n_components):
    filt_raw = raw.copy().load_data().filter(l_freq=1., h_freq=None)
    ica = ICA(n_components=15, max_iter='auto', random_state=97)
    ica.fit(filt_raw)
    ica.exclude = [0, 1]
    print('Bad components to remove:', ica.exclude)
    ica.exclude = []
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ica.exclude = eog_indices
    # barplot of ICA component "EOG match" scores
    ica.plot_scores(eog_scores)
    # plot diagnostics
    ica.plot_properties(raw, picks=eog_indices)
    # plot ICs applied to raw data, with EOG matches highlighted
    ica.plot_sources(raw, show_scrollbars=False)
    plt.show()

def get_clear(raw, clear):
    raw_tmp = raw.copy()
    data = raw_tmp.filter(7., 40., fir_design='firwin', skip_by_annotation='edge')  # Apply band-pass filter
    if clear == True:
        raw_tmp = removing_artifacts(raw_tmp, 20)
    return data


def get_raw(testee, task, taskname, data_name):
    print('... Parsing of', taskname, 'runs of', data_name)
    print(testee, task)
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
        clear = get_clear(raw, args['clear'])
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
        if args['clear'] == True:
            removing_artifacts2(raw, 20)
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
    main_preproc(data_name=args['data'], testee=int(args['testee']), task=args['task'], viz=args['viz'])
