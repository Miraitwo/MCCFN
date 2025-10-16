import pickle
import torch
import numpy as np
import torch.utils.data as Data
import h5py



def Load_Dataset(dataset,
                 logger
                 ):
    if dataset == '2016.10a':
        classes = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
                   b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}
    elif dataset == '2016.10b':
        classes = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
                   b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9}
    elif dataset == '2016.04c':
        classes = {b'8PSK': 0, b'AM-DSB': 1, b'AM-SSB': 2, b'BPSK': 3, b'CPFSK': 4,
                        b'GFSK': 5, b'PAM4': 6, b'QAM16': 7, b'QAM64': 8, b'QPSK': 9, b'WBFM': 10}
    elif dataset == 'rml22':
        classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4, 'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7,
              'PAM4': 8, 'QPSK': 9, 'AM-SSB': 10}
        raise NotImplementedError(f'Not Implemented dataset:{dataset}')

    dataset_file = {'2016.10a': 'RML2016.10a_dict.pkl',
                    '2016.10b': 'RML2016.10b.dat',
                    '2016.04c': '2016.04C.multisnr.pkl',
                    'rml22': 'RML22.01A'}

    # file_pointer = './dataset/%s' % dataset_file.get(dataset)
    file_pointer = r'H:\desktop\my_ws\work_station\dataset\%s' % dataset_file.get(dataset)

    Signals = []
    Labels = []
    SNRs = []

    if dataset == '2016.10a' or dataset == '2016.10b' or dataset == '2016.04c' or dataset == 'rml22':
        Set = pickle.load(open(file_pointer, 'rb'), encoding='bytes')
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Set.keys())))), [1, 0])

        for mod in mods:
            for snr in snrs:
                Signals.append(Set[(mod, snr)])
                for i in range(Set[(mod, snr)].shape[0]):
                    Labels.append(mod)
                    SNRs.append(snr)

        Signals = np.vstack(Signals)
        Signals = torch.from_numpy(Signals.astype(np.float32))
        Labels = [classes[i] for i in Labels]
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)
    logger.info('*' * 20)
    logger.info(f'Signals.shape: {list(Signals.shape)}')
    logger.info(f'Labels.shape: {list(Labels.shape)}')
    logger.info('*' * 20)

    return Signals, Labels, SNRs, snrs, mods


def Dataset_Split(Signals,
                  Labels,
                  SNRs,
                  snrs,
                  mods,
                  logger,
                  val_size=0.2,
                  test_size=0.1
                  ):
    global test_idx

    n_examples = Signals.shape[0]
    n_train = int(n_examples * (1 - val_size - test_size))

    train_idx = []
    test_idx = []
    val_idx = []

    Slices_list = np.linspace(0, n_examples, num=len(mods) * len(snrs) + 1)

    for k in range(0, Slices_list.shape[0] - 1):
        train_idx_subset = np.random.choice(
            range(int(Slices_list[k]), int(Slices_list[k + 1])), size=int(n_train / (len(mods) * len(snrs))),
            replace=False)
        Test_Val_idx_subset = list(set(range(int(Slices_list[k]), int(Slices_list[k + 1]))) - set(train_idx_subset))
        test_idx_subset = np.random.choice(Test_Val_idx_subset,
                                           size=int(
                                               (n_examples - n_train) * test_size / (
                                                       (len(mods) * len(snrs)) * (test_size + val_size))),
                                           replace=False)
        val_idx_subset = list(set(Test_Val_idx_subset) - set(test_idx_subset))

        train_idx = np.hstack([train_idx, train_idx_subset])
        val_idx = np.hstack([val_idx, val_idx_subset])
        test_idx = np.hstack([test_idx, test_idx_subset])

    train_idx = np.array(train_idx, dtype='int64')

    Signals_train = Signals[train_idx]
    Labels_train = Labels[train_idx]

    val_idx = np.array(val_idx, dtype='int64')
    test_idx = np.array(test_idx, dtype='int64')

    Signals_test = Signals[test_idx]
    Labels_test = Labels[test_idx]

    Signals_val = Signals[val_idx]
    Labels_val = Labels[val_idx]


    logger.info(f"Signal_train.shape: {list(Signals_train.shape)}", )
    logger.info(f"Signal_val.shape: {list(Signals_val.shape)}")
    logger.info(f"Signal_test.shape: {list(Signals_test.shape)}")
    logger.info('*' * 20)

    return (Signals_train, Labels_train), \
           (Signals_test, Labels_test), \
           (Signals_val, Labels_val), \
            test_idx

def Create_Data_Loader(train_set, val_set, cfg, logger):

    train_data = Data.TensorDataset(*train_set)
    val_data = Data.TensorDataset(*val_set)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_loader = Data.DataLoader(
        dataset=val_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    logger.info(f"train_loader batch: {len(train_loader)}")
    logger.info(f"val_loader batch: {len(val_loader)}")

    return train_loader, val_loader




