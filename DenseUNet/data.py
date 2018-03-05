from __future__ import print_function
from __future__ import division

from glob import glob
import numpy as np
from tqdm import tqdm


class ImageLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg
        data_dir = cfg.data_dir
        self.train_files = glob(data_dir + '/train/*.npz')
        self.val_files = glob(data_dir + '/val/*.npz')
        self.img_mean = self.cfg.image_mean
        self.img_stddev = self.cfg.image_stddev
        print("Size of training set : ", len(self.train_files))
        print("Size of validation set : ", len(self.val_files))

    def preprocess_image(self, img):
        return (img - self.img_mean) / self.img_stddev

    def load_batch(self, files):
        batch_imgs, batch_bboxes = [], []
        batch_instance_masks, batch_direction_masks = [], []
        for f in files:
            np_data = np.load(f)
            img = np_data['image']
            bboxes = np_data['bboxes']
            instance_mask = np_data['instances']
            direction_mask = np_data['directions']
            img = self.preprocess_image(img)
            batch_imgs.append(img)
            batch_bboxes.append(bboxes)
            batch_instance_masks.append(instance_mask)
            batch_direction_masks.append(direction_mask)
        return (np.array(batch_imgs), np.array(batch_bboxes),
                np.array(batch_instance_masks), np.array(batch_direction_masks))

    def batch_generator(self):
        n_train_files = len(self.train_files)
        train_idx = np.arange(n_train_files)
        batch_size = self.cfg.batch_size
        for _ in range(n_train_files):
            indices = np.random.randint(len(train_idx), size=batch_size)
            batch_files = np.array(self.train_files)[indices]
            batch = self.load_batch(batch_files)
            yield batch

    def test_batch_generator(self):
        val_idx = np.arange(len(self.val_files))
        for index in tqdm(val_idx):
            files = [self.val_files[index]]
            yield self.load_batch(files)
