import os
import pickle

import h5py
import numpy as np
import torch
# from tqdm import tqdm

# ========================================================
#   Usefull paths
# _datasetFeaturesFiles = {"miniImagenet": "./checkpoints/miniImagenet/WideResNet28_10_S2M2_R/last/novel_features.plk",
#                          "CUB": "./checkpoints/CUB/WideResNet28_10_S2M2_R/last/novel_features.plk",}
_datasetFeaturesFiles = {"miniImagenet": "./miniImagenet/novel_features.plk",
                         "miniImagenet_val": "./miniImagenet_free_lunch/val_features.plk",
                         "miniImagenet_xu": "./miniImagenet/novel_features.plk",
                         "miniImagenet_FEAT": "./miniImagenet_FEAT/test.hdf5",
                         "miniImagenet_FEAT_res12": "./miniImagenet_FEAT_res12/test.hdf5",
                         "miniImagenet_emd": "./miniImagenet_emd/test.hdf5",
                         "meta-baseline": "./meta-baseline/test.hdf5",
                         "BML-Global": "./BML-Global/test.hdf5",
                         "mini_BML": "./mini_BML/test.hdf5",
                         "FRN-order": "./FRN-order/test.hdf5",
                         "CUB_BML_global": "./CUB_BML_global/test.hdf5",
                         "CUB": "./CUB/novel_features.plk",
                         "tiered": "./tiered/novel.hdf5",
                         "tiered_meta_baseline": "./tiered_meta_baseline/test.hdf5",
                         "tiered_FRN": "./tiered_FRN/test.hdf5",
                         "tiered_BML_global": "./tiered_BML_global/test.hdf5",
                         "tiered_feat_wrn": "./tiered_feat_wrn/test.hdf5",
                         "LabHal": "./LabHal/test.hdf5",
                         "LabHal_gen": "./LabHal_gen/test.hdf5",
                         "VITS": "./VITS/test.hdf5",
                         "VITS_v1": "./VITS_v1/test.hdf5",
                         "VITS_5shot": "./VITS_5shot/test.hdf5",
                         "VIT_tiered": "./VIT_tiered/test.hdf5",
                         "VIT_cifar": "./VIT_cifar/test.hdf5",
                         "cifar_BML-Global": "./cifar_BML-Global/test.hdf5",
                         "cifar_DeepEMD": "./cifar_DeepEMD/test.hdf5",
                         "S2M2_cifar": "./S2M2_cifar/test.hdf5",
                         "S2M2_tiered": "./S2M2_tiered/test.hdf5",
                         "mini_imagenet": "./meta-dataset/mini_imagenet/novel_features.plk",
                         "mini2CUB": r"E:\Project\Few_Shot_Distribution_Calibration-master\meta-dataset\mini2CUB\last\novel_features.plk"}


_cacheDir = r"E:\Project\Few_Shot_Distribution_Calibration-master\cache"
# _maxRuns = 10000
_maxRuns = 1
_min_examples = -1

# ========================================================
#   Module internal functions and variables

_randStates = None
_rsCfg = None


def _load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        labels = [np.full(shape=len(data[key]), fill_value=key)
                  for key in data]
        data = [features for key in data for features in data[key]]
        # labels = data[1]
        # data = data[0]

        dataset = dict()
        dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
        dataset['labels'] = torch.LongTensor(np.concatenate(labels))
        return dataset

def _load_h5file(file):
    data = h5py.File(file)
    dataset = dict()
    data_fea = data['all_feats'][...]
    data_lab = data['all_labels'][...]
    # idx = [data_lab != 0][0]
    # dataset['data'] = torch.FloatTensor((data_fea[idx]))
    # dataset['labels'] = torch.LongTensor((data_lab[idx]))
    dataset['data'] = torch.FloatTensor((data_fea))
    dataset['labels'] = torch.LongTensor((data_lab))
    return dataset

def _load_pickle_meta(file):
    dataset = dict()
    all_labels = list()
    all_feats = list()
    with open(file, 'rb') as f:
        data = pickle.load(f)
        for key in data:
            feats = data[key]
            all_feats.extend(feats)
            all_labels.extend([key] * len(feats))
    all_feats_dset = np.array(all_feats)
    dataset['data'] = torch.FloatTensor(all_feats_dset)
    dataset['labels'] = torch.LongTensor(all_labels)
        # labels = [np.full(shape=len(data[key]), fill_value=key)
        #           for key in data]
        # data = [features for key in data for features in data[key]]
        # # labels = data[1]
        # # data = data[0]
        #
        # dataset = dict()
        # dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
        # dataset['labels'] = torch.LongTensor(np.concatenate(labels))
    return dataset

# =========================================================
#    Callable variables and functions from outside the module

data = None
labels = None
dsName = None


def loadDataSet(dsname):
    if dsname not in _datasetFeaturesFiles:
        raise NameError('Unknwown dataset: {}'.format(dsname))

    global dsName, data, labels, _randStates, _rsCfg, _min_examples
    dsName = dsname
    _randStates = None
    _rsCfg = None

    # Loading data from files on computer
    # home = expanduser("~")
    # if dsName == 'miniImagenet_val' or dsName == 'miniImagenet' or dsName == 'miniImagenet_xu' or dsName == 'CUB':
    #     dataset = _load_pickle(_datasetFeaturesFiles[dsname])
    # else:
    #     dataset = _load_h5file(_datasetFeaturesFiles[dsname])
    dataset = _load_pickle_meta(_datasetFeaturesFiles[dsname])

    # Computing the number of items per class in the dataset
    _min_examples = dataset["labels"].shape[0]
    for i in range(dataset["labels"].shape[0]):
        if torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0] > 0:
            _min_examples = min(_min_examples, torch.where(
                dataset["labels"] == dataset["labels"][i])[0].shape[0])
    print("Guaranteed number of items per class: {:d}\n".format(_min_examples))

    # Generating data tensors
    data = torch.zeros((0, _min_examples, dataset["data"].shape[1]))
    labels = dataset["labels"].clone()
    while labels.shape[0] > 0:
        indices = torch.where(dataset["labels"] == labels[0])[0]
        data = torch.cat([data, dataset["data"][indices, :]
                          [:_min_examples].view(1, _min_examples, -1)], dim=0)
        indices = torch.where(labels != labels[0])[0]
        labels = labels[indices]
    print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(
        data.shape[0], data.shape[1], data.shape[2]))


def GenerateRun(iRun, cfg, regenRState=False, generate=True):
    global _randStates, data, _min_examples
    if not regenRState:
        np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    # print(classes)
    shuffle_indices = np.arange(_min_examples)
    dataset = None
    if generate:
        dataset = torch.zeros(
            (cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))
    for i in range(cfg['ways']):
        shuffle_indices = np.random.permutation(shuffle_indices)
        if generate:
            dataset[i] = data[classes[i], shuffle_indices,
                              :][:cfg['shot']+cfg['queries']]

    return dataset


def ClassesInRun(iRun, cfg):
    global _randStates, data
    np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    return classes


def setRandomStates(cfg):
    global _randStates, _maxRuns, _rsCfg
    if _rsCfg == cfg:
        return

    rsFile = os.path.join(_cacheDir, "RandStates_{}_s{}_q{}_w{}".format(
        dsName, cfg['shot'], cfg['queries'], cfg['ways']))
    if not os.path.exists(rsFile):
        print("{} does not exist, regenerating it...".format(rsFile))
        np.random.seed(0)
        _randStates = []
        for iRun in range(_maxRuns):
            _randStates.append(np.random.get_state())
            GenerateRun(iRun, cfg, regenRState=True, generate=False)
        torch.save(_randStates, rsFile)
    else:
        print("reloading random states from file....")
        _randStates = torch.load(rsFile)
    _rsCfg = cfg


def GenerateRunSet(start=None, end=None, cfg=None):
    global dataset, _maxRuns
    if start is None:
        start = 0
    if end is None:
        end = _maxRuns
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "queries": 15}

    setRandomStates(cfg)
    print("generating task from {} to {}".format(start, end))

    dataset = torch.zeros(
        (end-start, cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))
    for iRun in range(end-start):
        dataset[iRun] = GenerateRun(start+iRun, cfg)

    return dataset


# define a main code to test this module
if __name__ == "__main__":

    print("Testing Task loader for Few Shot Learning")
    loadDataSet('miniimagenet')

    cfg = {"shot": 1, "ways": 5, "queries": 15}
    setRandomStates(cfg)

    run10 = GenerateRun(10, cfg)
    print("First call:", run10[:2, :2, :2])

    run10 = GenerateRun(10, cfg)
    print("Second call:", run10[:2, :2, :2])

    ds = GenerateRunSet(start=2, end=12, cfg=cfg)
    print("Third call:", ds[8, :2, :2, :2])
    print(ds.size())
