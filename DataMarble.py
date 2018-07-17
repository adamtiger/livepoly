

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
        pass

    def split_data_test_train(self, split_ratio):
        pass

    def preprocess_images(self):
        pass


    # ------------------------------
    # Image reader BLOCK
    # ------------------------------

    def read_image(self, id):
        pass

    def random_image_train(self):
        pass

    def random_image_test(self):
        pass

    def crop_segm(self):
        pass

    def crop_twin(self):
        pass

    def crop_none(self):
        pass


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

    def number_segmpoints_image(self, image_path):
        pass

    def number_allpoints_image(self, image_path):
        pass