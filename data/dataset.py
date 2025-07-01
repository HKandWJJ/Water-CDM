from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import numpy as np

class Dataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=256, split='train', data_len=-1):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.split = split

        if datatype == 'img':
            self.input_path = Util.get_paths_from_images(
                '{}/input'.format(dataroot, l_resolution, r_resolution))
            self.target_path = Util.get_paths_from_images(
                '{}/target'.format(dataroot, r_resolution))

            self.dataset_len = len(self.target_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        target = Image.open(self.target_path[index]).convert("RGB")
        input = Image.open(self.input_path[index]).convert("RGB")

        [input, target] = Util.transform_augment(
            [input, target], split=self.split, min_max=(-1, 1))
        return {'HR': target, 'SR': input, 'Index': index}
