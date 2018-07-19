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

    def __init__(self, folder_name, split_ratio=0.7, window_size=85):

        self.number_images(folder_name)
        self.preprocess_images()
        self.split_data_test_train(split_ratio)
        self.window_size = window_size

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

    def preprocess_images(self): # TODO: leave points which are too close to the border, save points for NONE as well
        path = self.folder_name + '_processed_copy'
        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            with open("meta_data.json", "w") as wf:
                self.meta_data = json.load(wf)
            return

        meta_data = dict()

        for item in range(self.n_imgs * 3):

            temp_img = imread(self.folder_name + '/' + self.img_names[item], as_gray=True)
            for ending in ['_s.', '_t.']:  # checking for segmenting and twin crystal points
                if self.img_names[item].find(ending) != -1:
                    points = []
                    for r in temp_img.shape[0]:
                        for c in temp_img.shape[1]:
                            if temp_img[r, c] > 250:  # choosing the segmentation/twin points
                                  points.append((r, c))

                    meta_data[item] = {'name': self.img_names[item], 'points': points, 'num_points': len(points)}

            imsave(path + '/' + self.img_names[item], temp_img)

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

    def __crop_general(self, id, img):
        rc = np.random.choice(self.meta_data[id]['points'])
        width_r = (rc[0] - (self.window_size - 1) // 2, img.shape[0] - (rc[0] + (self.window_size - 1) // 2))
        width_c = (rc[1] - (self.window_size - 1) // 2, img.shape[1] - (rc[1] + (self.window_size - 1) // 2))
        return crop(img, crop_width=(width_r, width_c), copy=True)

    def crop_segm(self, id, img):
        return self.__crop_general(id + 1, img)

    def crop_twin(self, id, img):
        return self.__crop_general(id + 2, img)

    def crop_none(self, id, img):
        return self.__crop_general(id, img)


    # ------------------------------
    # Function for training BLOCK
    # ------------------------------

    def train_batch(self, batch_size):
        pass

    def test_batch(self, batch_size):
        pass


    # ------------------------------
    # Image statistics BLOCK
    # ------------------------------

    def number_segmpoints_image(self, image_name):
        pass

    def number_allpoints_image(self, image_name):
        pass