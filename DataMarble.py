import numpy as np
import os
import json
from skimage.io import imread, imsave
from skimage.util import crop


class DataMarble:

    '''

    '''


    # ------------------------------
    # Initialization BLOCK
    # ------------------------------

    def __init__(self, folder_name, split_ratio=0.7, window_size=85, pool_update_freq=50):

        self.number_images(folder_name)
        self.preprocess_images()
        self.split_data_test_train(split_ratio)

        self.window_size = window_size
        self.pool_update_freq = pool_update_freq
        self.pool_updates = 0

    def number_images(self, folder_name):
        self.folder_name = folder_name
        self.img_names = os.listdir(folder_name)
        self.n_imgs = len(self.img_names)
        if self.n_imgs % 3 != 0:
            raise AssertionError('n_imgs % 3 != 0, something is missing')
        else:
            self.n_imgs //= 3
        return self.n_imgs  # return for outer usage (comfort)

    def split_data_test_train(self, split_ratio):
        seq = [item for item in range(self.n_imgs)]
        np.random.shuffle(seq)  # inplace
        split_idx = int(self.n_imgs * split_ratio)
        self.train_set = [seq[idx] for idx in range(split_idx)]
        self.test_set = [seq[idx] for idx in range(split_idx, self.n_imgs)]
        return self.train_set, self.test_set  # return for outer usage (comfort)

    def preprocess_images(self):
        path = self.folder_name + '_processed_copy'
        half_wnd_sz = (self.window_size - 1) // 2

        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            with open("meta_data.json", "w") as wf:
                self.meta_data = json.load(wf)
            return

        meta_data = dict()

        for item in range(self.n_imgs):

            temp_img_o = imread(self.folder_name + '/' + self.img_names[3 * item], as_gray=True)
            temp_img_s = imread(self.folder_name + '/' + self.img_names[3 * item + 1], as_gray=True)
            temp_img_t = imread(self.folder_name + '/' + self.img_names[3 * item + 2], as_gray=True)

            points_none = []
            points_s = []
            points_t = []
            for r in range(half_wnd_sz, temp_img_o.shape[0] - half_wnd_sz):
                for c in range(half_wnd_sz, temp_img_o.shape[1] - half_wnd_sz):
                    if temp_img_s[r, c] > 250:  # separating the segmentation/twin/none points
                        points_s.append((r, c))
                    elif temp_img_t[r, c] > 250:
                        points_t.append((r, c))
                    else:
                        points_none.append((r, c))

            meta_data[item] = {'none': points_none, 'segm': points_s, 'twin': points_t}

            imsave(path + '/' + self.img_names[3 * item], temp_img_o)
            imsave(path + '/' + self.img_names[3 * item + 1], temp_img_s)
            imsave(path + '/' + self.img_names[3 * item + 2], temp_img_t)

        with open("meta_data.json", "w") as wf:
            json.dump(json.dumps(meta_data, indent=4), wf)

        self.meta_data = meta_data
        self.processed_folder = path


    # ------------------------------
    # Image reader BLOCK
    # ------------------------------

    def read_image(self, id):
        return imread(self.img_names[id * 3], as_gray=True)

    def random_images_train(self, n):
        self.training_img_pool = []
        self.training_id_pool = []
        ids = np.random.choice(self.train_set, n, replace=False)
        for id in ids:
            self.training_img_pool.append(self.read_image(id))
            self.training_id_pool.append(id)

    def random_images_test(self, n):
        self.test_img_pool = []
        self.test_id_pool = []
        ids = np.random.choice(self.test_set, n, replace=False)
        for id in ids:
            self.test_img_pool.append(self.read_image(id))
            self.test_img_pool.append(id)

    def __crop_general(self, img, rc):
        width_r = (rc[0] - (self.window_size - 1) // 2, img.shape[0] - (rc[0] + (self.window_size - 1) // 2))
        width_c = (rc[1] - (self.window_size - 1) // 2, img.shape[1] - (rc[1] + (self.window_size - 1) // 2))
        return crop(img, crop_width=(width_r, width_c), copy=True)

    def crop_segm(self, id, img):
        rc = np.random.choice(self.meta_data[id]['segm'])
        return self.__crop_general(img, rc)

    def crop_twin(self, id, img):
        rc = np.random.choice(self.meta_data[id]['twin'])
        return self.__crop_general(img, rc)

    def crop_none(self, id, img):
        rc = np.random.choice(self.meta_data[id]['none'])
        return self.__crop_general(img, rc)


    # ------------------------------
    # Function for training BLOCK
    # ------------------------------

    def train_batch(self, batch_size, n):
        if self.pool_updates % self.pool_update_freq == 0:
            self.random_images_train(n)

        data_X = np.zeros((batch_size, self.window_size, self.window_size, 1), dtype=np.float32)
        data_Y = np.zeros((batch_size, 3), dtype=np.float32)

        idx = 0
        while idx < batch_size:
            for crop_fv in [self.crop_segm, self.crop_twin, self.crop_none]:
                data_X[idx] = crop_fv(self.training_id_pool[idx], self.training_img_pool[idx])
                data_Y[idx] = np.array([0, 1, 0], dtype=np.float32)
                idx += 1
                if idx == batch_size:
                    return data_X, data_Y

    def test_batch(self, batch_size, n):
        if self.pool_updates % self.pool_update_freq == 0:
            self.random_images_test(n)

        data_X = np.zeros((batch_size, self.window_size, self.window_size, 1), dtype=np.float32)
        data_Y = np.zeros((batch_size, 3), dtype=np.float32)

        idx = 0
        while idx < batch_size:
            for crop_fv, target in zip([self.crop_segm, self.crop_twin, self.crop_none], [[0, 1, 0], [0, 0, 1], [1, 0, 0]]):
                data_X[idx] = crop_fv(self.test_id_pool[idx], self.test_img_pool[idx])
                data_Y[idx] = np.array(target, dtype=np.float32)
                idx += 1
                if idx == batch_size:
                    return data_X, data_Y


    # ------------------------------
    # Image statistics BLOCK
    # ------------------------------

    def number_segmpoints_image(self, image_name):
        id = self.img_names.index(image_name)
        return len(self.meta_data[id]['segm'])

    def number_twinpoints_image(self, image_name):
        id = self.img_names.index(image_name)
        return len(self.meta_data[id]['twin'])

    def number_nonepoints_image(self, image_name):
        id = self.img_names.index(image_name)
        return len(self.meta_data[id]['none'])