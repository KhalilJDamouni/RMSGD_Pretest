from contextlib import contextmanager
from pathlib import Path

import warnings
import tempfile
import shutil
import os
from PIL import Image

import pandas as pd
import numpy as np
import torch
from torchvision.datasets.utils import check_integrity,\
    extract_archive, verify_str_arg, download_and_extract_archive
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import Dataset


class MHIST(ImageFolder):
    """`TinyImageNet
    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or
            ``val``.
        transform (callable, optional): A function/transform that  takes in an
            PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its
            path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, split='train', **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "test"))
        self.root = root
        root = Path(root)
        self.classes = ['SSA', 'HP']
        self.class_to_idx = {cls: idx
                             for idx, cls in enumerate(self.classes)}

        data = pd.read_csv(str(root / 'annotations.csv'))
        _len = len(data['Image Name'])
        self.images = list()
        self.targets = list()
        # self.transform = transform
        if not (root / split).exists():
            Path(root / split).mkdir(parents=True)
            Path(root / split / 'SSA').mkdir(parents=True)
            Path(root / split / 'HP').mkdir(parents=True)
            for i in range(_len):
                if data['Partition'][i] == split:
                    (root / 'images' / data['Image Name'][i]).rename(
                        root / split / data['Majority Vote Label'][i] / data['Image Name'][i])
        super(MHIST, self).__init__(self.split_folder, **kwargs)

    @ property
    def split_folder(self):
        return os.path.join(self.root, self.split)


class TinyImageNet(ImageFolder):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    meta_file = 'wnids.txt'
    """`TinyImageNet
    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or
            ``val``.
        transform (callable, optional): A function/transform that  takes in an
            PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its
            path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, split='train', download=False, **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.root = root
        if download:
            # self.download()
            raise ValueError(
                "Downloading of TinyImageNet is not supported. " +
                "You must manually download the 'tiny-imagenet-200.zip' from" +
                f" {self.url} and extract the 'tiny-imagenet-200' folder " +
                "into the folder specified by 'root'. That is, once the" +
                "'tiny-imagenet-200' folder is extracted, specify the data " +
                "directory for this program as the path for to that folder")
        self.parse_archives()
        self.classes = self.load_meta_file()
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}
        super(TinyImageNet, self).__init__(self.split_folder, **kwargs)

    def _check_integrity(self):
        dirs = [d.name for d in Path(self.root).iterdir()]
        if 'train' not in dirs or 'test' not in dirs or 'val' not in dirs:
            return False
        if not (Path(self.root) / 'wnids.txt').exists():
            return False
        return True

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
        download_and_extract_archive(
            self.url, self.root,
            filename=self.filename, md5=None)

    def load_meta_file(self):
        if self._check_integrity():
            with (Path(self.root) / self.meta_file).open('r') as f:
                lines = [line.strip() for line in f.readlines()]
        return lines

    def parse_archives(self):
        if self._check_integrity():
            name = (Path(self.root) / 'train')
            if (name / 'images').exists():
                for c in name.iterdir():
                    os.remove(str(c / f'{c.name}_boxes.txt'))
                    for f in (c / 'images').iterdir():
                        shutil.move(str(f), c)
                    shutil.rmtree(str(c / 'images'))
            name = (Path(self.root) / 'val')
            if (name / 'images').exists():
                with (name / 'val_annotations.txt').open('r') as f:
                    for line in f.readlines():
                        line = line.replace('\t', ' ').strip().split(' ')
                        (name / line[1]).mkdir(exist_ok=True)
                        shutil.move(str(name / 'images' / line[0]),
                                    str(name / line[1]))
                shutil.rmtree(str(name / 'images'))
                os.remove(name / 'val_annotations.txt')

    @ property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


class ImageNet(ImageFolder):
    archive_meta = {
        'train': ('ILSVRC2012_img_train.tar',
                  '1d675b47d978889d74fa0da5fadfb00e'),
        'val': ('ILSVRC2012_img_val.tar',
                '29b22e2961454d5413ddabcf34fc5622'),
        'devkit': ('ILSVRC2012_devkit_t12.tar.gz',
                   'fa75699e90414af021442c21a62c3abf')
    }
    meta_file = "meta.bin"
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or
            ``val``.
        transform (callable, optional): A function/transform that  takes in an
            PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its
            path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, split='train', download=None, **kwargs):
        if download is True:
            msg = ("The dataset is no longer publicly accessible. You need to "
                   "download the archives externally and place them in the "
                   "root directory.")
            raise RuntimeError(msg)
        elif download is False:
            msg = ("The use of the download flag is deprecated, since the "
                   "dataset is no longer publicly accessible.")
            warnings.warn(msg, RuntimeWarning)

        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    def parse_archives(self):
        if not check_integrity(os.path.join(self.root, self.meta_file)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @ property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


def load_meta_file(root, file=None):

    if file is None:
        file = ImageNet.meta_file
    file = os.path.join(root, file)

    if check_integrity(file):
        return torch.load(file)
    else:
        msg = ("The meta file {} is not present in the root directory or is "
               "corrupted. This file is automatically created by the"
               " ImageNet dataset.")
        raise RuntimeError(msg.format(file, root))


def _verify_archive(root, file, md5):
    if not check_integrity(os.path.join(root, file), md5):
        msg = ("The archive {} is not present in the root directory or is"
               "corrupted. You need to download it externally and place it"
               " in {}.")
        raise RuntimeError(msg.format(file, root))


def parse_devkit_archive(root, file=None):
    """Parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.

    Args:
        root (str): Root directory containing the devkit archive
        file (str, optional): Name of devkit archive. Defaults to
            'ILSVRC2012_devkit_t12.tar.gz'
    """
    import scipy.io as sio

    def parse_meta_mat(devkit_root):
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children)
                if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(', ')) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root):
        file = os.path.join(devkit_root, "data",
                            "ILSVRC2012_validation_ground_truth.txt")
        with open(file, 'r') as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    @ contextmanager
    def get_tmp_dir():
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir)

    archive_meta = ImageNet.archive_meta["devkit"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    _verify_archive(root, file, md5)

    with get_tmp_dir() as tmp_dir:
        extract_archive(os.path.join(root, file), tmp_dir)

        devkit_root = os.path.join(tmp_dir, "ILSVRC2012_devkit_t12")
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        torch.save((wnid_to_classes, val_wnids),
                   os.path.join(root, ImageNet.meta_file))


def parse_train_archive(root, file=None, folder="train"):
    """Parse the train images archive of the ImageNet2012 classification
        dataset and prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the train images archive
        file (str, optional): Name of train images archive. Defaults to
            'ILSVRC2012_img_train.tar'
        folder (str, optional): Optional name for train images folder.
            Defaults to 'train'
    """
    archive_meta = ImageNet.archive_meta["train"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    _verify_archive(root, file, md5)

    train_root = os.path.join(root, folder)
    extract_archive(os.path.join(root, file), train_root)

    archives = [os.path.join(train_root, archive)
                for archive in os.listdir(train_root)]
    for archive in archives:
        extract_archive(archive, os.path.splitext(
            archive)[0], remove_finished=False)


def parse_val_archive(root, file=None, wnids=None, folder="val"):
    """Parse the validation images archive of the ImageNet2012 classification
        dataset
    and prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the validation images archive
        file (str, optional): Name of validation images archive. Defaults to
            'ILSVRC2012_img_val.tar'
        wnids (list, optional): List of WordNet IDs of the validation images.
            If None is given, the IDs are loaded from the meta file in the root
            directory
        folder (str, optional): Optional name for validation images folder.
            Defaults to 'val'
    """
    archive_meta = ImageNet.archive_meta["val"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]
    if wnids is None:
        wnids = load_meta_file(root)[1]

    _verify_archive(root, file, md5)

    val_root = os.path.join(root, folder)
    extract_archive(os.path.join(root, file), val_root)

    images = sorted([os.path.join(val_root, image)
                     for image in os.listdir(val_root)])

    for wnid in set(wnids):
        os.mkdir(os.path.join(val_root, wnid))

    for wnid, img_file in zip(wnids, images):
        shutil.move(img_file, os.path.join(
            val_root, wnid, os.path.basename(img_file)))
