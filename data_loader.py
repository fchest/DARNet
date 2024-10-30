import math

import numpy as np
import pandas as pd
import torch
from dotmap import DotMap
from mne.decoding import CSP
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader



def get_DTU_data(name="S1", timelen=1, data_document_path="E:/EEG_data/DTU/DATA_preproc"):
    class CustomDatasets(Dataset):
        # initialization: data and label
        def __init__(self, data, label):
            self.data = torch.Tensor(data)
            self.label = torch.tensor(label, dtype=torch.uint8)

        # get the size of data
        def __len__(self):
            return len(self.label)

        # get the data and label
        def __getitem__(self, index):
            return self.data[index], self.label[index]

    def get_data_from_mat(mat_path):
        '''
        discription:load data from mat path and reshape
        param{type}:mat_path: Str
        return{type}: onesub_data
        '''
        mat_eeg_data = []
        mat_wavA_data = []
        mat_wavB_data = []
        mat_event_data = []
        matstruct_contents = loadmat(mat_path)
        matstruct_contents = matstruct_contents['data']
        mat_event = matstruct_contents[0, 0]['event']['eeg'].item()
        mat_event_value = mat_event[0]['value']  # 1*60 1=male, 2=female
        mat_eeg = matstruct_contents[0, 0]['eeg']  # 60 trials 3200*66
        mat_wavA = matstruct_contents[0, 0]['wavA']
        mat_wavB = matstruct_contents[0, 0]['wavB']
        for i in range(mat_eeg.shape[1]):
            mat_eeg_data.append(mat_eeg[0, i])
            mat_wavA_data.append(mat_wavA[0, i])
            mat_wavB_data.append(mat_wavB[0, i])
            mat_event_data.append(mat_event_value[i][0][0])

        # return mat_eeg_data, mat_wavA_data, mat_wavB_data, mat_event_data
        return mat_eeg_data, mat_event_data

    def sliding_window(eeg_datas, labels, args, eeg_channel):
        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        train_eeg = []
        test_eeg = []
        train_label = []
        test_label = []

        for m in range(len(labels)):
            eeg = eeg_datas[m]
            label = labels[m]
            windows = []
            new_label = []
            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                windows.append(window)
                new_label.append(label)
            train_eeg.append(np.array(windows)[:int(len(windows) * 0.9)])
            test_eeg.append(np.array(windows)[int(len(windows) * 0.9):])
            train_label.append(np.array(new_label)[:int(len(windows) * 0.9)])
            test_label.append(np.array(new_label)[int(len(windows) * 0.9):])
        train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        train_label = np.stack(train_label, axis=0).reshape(-1, 1)
        test_label = np.stack(test_label, axis=0).reshape(-1, 1)

        return train_eeg, test_eeg, train_label, test_label

    print("Num GPUs Available: ", torch.cuda.is_available())
    print(name)
    time_len = timelen
    random_seed = 42
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 64 * time_len
    args.window_length = math.ceil(args.fs)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 200
    args.random_seed = random_seed
    args.people_number = 18
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 60
    args.cell_number = 3200
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.log_interval = 20
    args.csp_comp = 64
    args.label_col = 0
    args.log_path = "ConvTran-main-DTU/Results/1s"
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    subpath = args.data_document_path + '/' + str(args.name) + '_data_preproc.mat'
    eeg_data, event_data = get_data_from_mat(subpath)
    eeg_data = np.array(eeg_data)
    eeg_data = eeg_data[:, :, 0:64]

    event_data = np.array(event_data)
    print(eeg_data.shape)
    eeg_data = np.vstack(eeg_data)
    eeg_data = eeg_data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)
    eeg_data = np.array(eeg_data)
    print(eeg_data.shape)

    eeg_data = eeg_data.transpose(0, 2, 1)
    event_data = np.squeeze(event_data - 1)
    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
              norm_trace=True)
    eeg_data = csp.fit_transform(eeg_data, event_data)
    eeg_data = eeg_data.transpose(0, 2, 1)

    train_data, test_data, train_label, test_label = sliding_window(eeg_data, event_data, args, args.csp_comp)
    del eeg_data
    del event_data

    # set the number of training, testing and validating data
    args.n_test = len(test_label)
    args.n_valid = args.n_test
    args.n_train = len(train_label) - args.n_test


    train_data = train_data.transpose(0, 2, 1)
    test_data = test_data.transpose(0, 2, 1)

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data, train_label = train_data[indices], train_label[indices]

    valid_data, valid_label = train_data[args.n_train:], train_label[args.n_train:]
    train_data, train_label = train_data[:args.n_train], train_label[:args.n_train]

    train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=CustomDatasets(valid_data, valid_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    test_loader = DataLoader(dataset=CustomDatasets(test_data, test_label),
                             batch_size=args.batch_size, drop_last=True, pin_memory=True)
    return train_loader, valid_loader, test_loader


def get_KUL_data(name="S1", time_len=1, data_document_path="E:/EEG_data/KUL_single_siongle3"):
    class CustomDatasets(Dataset):
        # initialization: data and label
        def __init__(self, data, label):
            self.data = torch.Tensor(data)
            self.label = torch.tensor(label, dtype=torch.uint8)

        # get the size of data
        def __len__(self):
            return len(self.label)

        # get the data and label
        def __getitem__(self, index):
            return self.data[index], self.label[index]

    def read_prepared_data(args):
        data = []
        target = []
        for l in range(len(args.ConType)):
            label = pd.read_csv(args.data_document_path + "/csv/" + args.name + args.ConType[l] + ".csv")

            for k in range(args.trail_number):
                filename = args.data_document_path + "/" + args.ConType[l] + "/" + args.name + "Tra" + str(
                    k + 1) + ".csv"
                # KUL_single_single3,contype=no,name=s1,len(arg.ConType)=1
                data_pf = pd.read_csv(filename, header=None)
                eeg_data = data_pf.iloc[:, 2:]  # KUL,DTU


                data.append(eeg_data)
                target.append(label.iloc[k, args.label_col])

        return data, target

    def sliding_window(eeg_datas, labels, args, eeg_channel):
        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        train_eeg = []
        test_eeg = []
        train_label = []
        test_label = []

        for m in range(len(labels)):
            eeg = eeg_datas[m]
            label = labels[m]
            windows = []
            new_label = []
            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                windows.append(window)
                new_label.append(label)
            train_eeg.append(np.array(windows)[:int(len(windows) * 0.9)])
            test_eeg.append(np.array(windows)[int(len(windows) * 0.9):])
            train_label.append(np.array(new_label)[:int(len(windows) * 0.9)])
            test_label.append(np.array(new_label)[int(len(windows) * 0.9):])
        train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        train_label = np.stack(train_label, axis=0).reshape(-1, 1)
        test_label = np.stack(test_label, axis=0).reshape(-1, 1)

        return train_eeg, test_eeg, train_label, test_label

    print("Num GPUs Available: ", torch.cuda.is_available())
    print(name)
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 200
    args.random_seed = 1234
    args.image_size = 32
    args.people_number = 16
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 8
    args.cell_number = 46080
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.log_interval = 20
    args.label_col = 0
    args.alpha_low = 8
    args.alpha_high = 13
    args.log_path = "result"
    args.frequency_resolution = args.fs / args.window_length
    args.point_low = math.ceil(args.alpha_low / args.frequency_resolution)
    args.point_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    args.csp_comp = 64

    # load data 和 label
    eeg_data, event_data = read_prepared_data(args)
    data = np.vstack(eeg_data)
    eeg_data = data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)

    eeg_data = eeg_data.transpose(0, 2, 1)
    event_data = np.squeeze(event_data - 1)
    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
              norm_trace=True)
    eeg_data = csp.fit_transform(eeg_data, event_data)
    eeg_data = eeg_data.transpose(0, 2, 1)

    train_data, test_data, train_label, test_label = sliding_window(eeg_data, event_data, args, args.csp_comp)

    # set the number of training, testing and validating data
    args.n_test = len(test_label)
    args.n_valid = args.n_test
    args.n_train = len(train_label) - args.n_test

    print(1, data.shape)
    print("len of test_label", len(test_label), len(train_label))
    del data

    print(train_data.shape, 5)
    train_data = train_data.transpose(0, 2, 1)
    test_data = test_data.transpose(0, 2, 1)

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data, train_label = train_data[indices], train_label[indices]

    print(args.n_train, args.n_valid)
    valid_data, valid_label = train_data[args.n_train:], train_label[args.n_train:]
    train_data, train_label = train_data[:args.n_train], train_label[:args.n_train]

    train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=CustomDatasets(valid_data, valid_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    test_loader = DataLoader(dataset=CustomDatasets(test_data, test_label),
                             batch_size=args.batch_size, drop_last=True, pin_memory=True)
    return train_loader, valid_loader, test_loader

def get_KUL_data_correct(name="S1", time_len=1, data_document_path="E:/EEG_data/KUL_single_siongle3"):
    # CSP after the dataset split:
    class CustomDatasets(Dataset):
        # initialization: data and label
        def __init__(self, data, label):
            self.data = torch.Tensor(data)
            self.label = torch.tensor(label, dtype=torch.uint8)

        # get the size of data
        def __len__(self):
            return len(self.label)

        # get the data and label
        def __getitem__(self, index):
            return self.data[index], self.label[index]

    def read_prepared_data(args):
        data = []
        target = []
        for l in range(len(args.ConType)):
            label = pd.read_csv(args.data_document_path + "/csv/" + args.name + args.ConType[l] + ".csv")

            for k in range(args.trail_number):
                filename = args.data_document_path + "/" + args.ConType[l] + "/" + args.name + "Tra" + str(
                    k + 1) + ".csv"
                # KUL_single_single3,contype=no,name=s1,len(arg.ConType)=1
                data_pf = pd.read_csv(filename, header=None)
                eeg_data = data_pf.iloc[:, 2:]  # KUL,DTU


                data.append(eeg_data)
                target.append(label.iloc[k, args.label_col])

        return data, target

    def sliding_window(eeg_datas, labels, args, out_channels):
        window_size = args.window_length

        stride = int(window_size * (1 - args.overlap))

        eeg_set = []
        label_set = []

        for m in range(len(labels)):  # labels 0-19
            eeg = eeg_datas[m]
            label = labels[m]
            windows = []
            new_label = []
            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                windows.append(window)
                new_label.append(label)

            eeg_set.append(np.array(windows))
            label_set.append(np.array(new_label))

        eeg_set = np.stack(eeg_set, axis=0).reshape(-1, window_size, out_channels)
        label_set = np.stack(label_set, axis=0).reshape(-1, 1)

        return eeg_set, label_set

    def within_data(eeg_datas, labels):
        train_datas = []
        test_datas = []
        train_labels = []
        test_labels = []

        for m in range(len(labels)):  # labels 0-19
            eeg = eeg_datas[m]
            label = labels[m]

            train_datas.append(np.array(eeg)[:, :int(eeg.shape[1] * 0.9)])
            test_datas.append(np.array(eeg)[:, int(eeg.shape[1] * 0.9):])
            train_labels.append(np.array(label))
            test_labels.append(np.array(label))

        train_datas = np.stack(train_datas, axis=0)
        test_datas = np.stack(test_datas, axis=0)
        train_labels = np.stack(train_labels, axis=0)
        test_labels = np.stack(test_labels, axis=0)

        return train_datas, test_datas, train_labels, test_labels

    print("Num GPUs Available: ", torch.cuda.is_available())
    print(name)
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 200
    args.random_seed = 1234
    args.image_size = 32
    args.people_number = 16
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 8
    args.cell_number = 46080
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.log_interval = 20
    args.label_col = 0
    args.alpha_low = 8
    args.alpha_high = 13
    args.log_path = "result"
    args.frequency_resolution = args.fs / args.window_length
    args.point_low = math.ceil(args.alpha_low / args.frequency_resolution)
    args.point_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    args.csp_comp = 64

    # load data 和 label
    eeg_data, event_data = read_prepared_data(args)
    data = np.vstack(eeg_data)
    eeg_data = data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)
    label = np.squeeze(event_data - 1)
    eeg_data = eeg_data.transpose(0, 2, 1)

    # CSP after the dataset split:
    train_data, test_data, train_label, test_label = within_data(eeg_data, label)

    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
              norm_trace=True)
    train_data = csp.fit_transform(train_data, train_label)

    test_data = csp.transform(test_data)

    train_data = train_data.transpose(0, 2, 1)
    test_data = test_data.transpose(0, 2, 1)

    train_data, train_label = sliding_window(train_data, train_label, args, args.csp_comp)
    test_data, test_label = sliding_window(test_data, test_label, args, args.csp_comp)

    train_data = train_data.transpose(0, 2, 1)
    test_data = test_data.transpose(0, 2, 1)
    print(train_data.shape)
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data, train_label = train_data[indices], train_label[indices]

    train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label, test_size=0.1, shuffle=True, random_state=random.randint(0,10000))

    print(train_data.shape)
    train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=CustomDatasets(valid_data, valid_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    test_loader = DataLoader(dataset=CustomDatasets(test_data, test_label),
                             batch_size=args.batch_size, drop_last=True, pin_memory=True)
    return train_loader, valid_loader, test_loader
