'''
This module is for managing the whole training process for live-polyline.

'''

import os
import random
import pool
import numpy as np
import nn

# discover the folder structure and find images
# file name with ending '_orig' is the original version

folder_name = "imgs"

img_names = os.listdir(folder_name)
img_paths = ["imgs/" + x for x in img_names]


def choose_random_images(number):

    images = []
    num_images = int(len(img_paths) / 2)
    r_list = [x for x in range(num_images)]

    if number > num_images:
        raise ValueError("There are fewer images.")

    for i in range(number):
        r = random.randint(0, num_images-1)
        img_idx = r_list[r]
        num_images -= 1
        r_list.remove(r_list[r])

        img_orig = pool.read_image(img_paths[img_idx])
        img_orig = pool.convert_grey(img_orig)
        img_segm = pool.read_image(img_paths[img_idx+1])
        img_segm = pool.convert_grey(img_segm)

        images.append(pool.Image(img_orig, img_segm))

    return images


def generate_samples(image, number):

    samples = {'image': np.ndarray((number, 86, 86, 1), dtype=np.float32), 'is_segmenting': np.ndarray((number, 1, 1, 1), dtype=np.float32)}
    side = 86

    for i in range(number):

        position = (random.randint(0, image.orig().shape()[0] - (side + 1)), random.randint(0, image.orig().shape()[1] - (side + 1)))
        x = pool.crop_out(image.orig(), position, (side, side, 1))
        y = pool.check_segmentation(image.segm(), (position[0] + side/2, position[1] + side/2 + 1))
        y = y or pool.check_segmentation(image.segm(), (position[0] + side/2 + 1, position[1] + side/2))
        y = y or pool.check_segmentation(image.segm(), (position[0] + side/2, position[1] + side/2 + 1))
        y = y or pool.check_segmentation(image.segm(), (position[0] + side/2 + 1, position[1] + side/2 + 1))

        samples['image'][i,:,:,:] = x[:,:,:]
        samples['is_segmenting'][i, 0, 0, 0] = y

    return samples


# test running neural network
# generate random batch
def test_nn():

    model = nn.create_model()
    batch = {'image': np.ndarray((10, 86, 86, 1), dtype=np.float32), 'is_segmenting': np.ndarray((10, 1, 1, 1), dtype=np.float32)}
    for i in range(10):
        batch['image'][i,:,:,:]= pool.generate_random_image((86, 86, 1))[:,:,:]
        batch['is_segmenting'][i,0,0,0] = random.randint(0, 1)

    nn.train_batch(model, batch, 10, 10)
    print(model.metrics_names)
    print(nn.evaluate(model, batch, 10))


def train_nn():

    # parameters
    iteration = 10
    images_num = 1
    training_sample_num = 16
    test_sample_num = 16
    batch_size = 16
    epochs = 10
    model_file_name = "model.cntk"
    eval_file_name = "eval.txt"

    def gen_data(images, sample_num):
        data_chunk = {}
        data_chunk['image'] = np.ndarray((images_num * sample_num, 86, 86, 1), dtype=np.float32)
        data_chunk['is_segmenting'] = np.ndarray((images_num * sample_num, 1, 1, 1), dtype=np.float32)
        for i in range(len(images)):
            samples = generate_samples(images[i], sample_num)
            data_chunk['image'][i * sample_num:(i + 1) * sample_num, :, :, :] = samples['image'][:, :, :]
            data_chunk['is_segmenting'][i * sample_num:(i + 1) * sample_num, :, :, :] = samples['is_segmenting'][:, :, :]

        return data_chunk

    model = nn.create_model()

    eval_history = []

    for i in range(iteration):
        images = choose_random_images(images_num)
        data_chunk = gen_data(images, training_sample_num)

        nn.train_batch(model, data_chunk, batch_size, epochs)

        if i % 5 == 0:
            test_set = gen_data(images, test_sample_num)
            eval_history.append(nn.evaluate(model, test_set, batch_size))

    nn.save_model(model, model_file_name)

    with open(eval_file_name, "w") as f:
        for item in eval_history:
            line = "Data: " + str(item[0]) + " " + str(item[1] + "\n")
            f.write(line)


train_nn()


