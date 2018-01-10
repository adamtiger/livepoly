'''
This module is for managing the whole training process for live-polyline.

'''

import os
import time
import random
import json
import pool
import numpy as np
import tfnn as nn
import utils as u
import argparse

parser = argparse.ArgumentParser(description="Live-poyline training algorithm")

parser.add_argument("--mode", type=int, default=0, metavar='N',
                    help="0: training, 1: test, 2: weight calculation, 3: transfer learning")
parser.add_argument("--memory", type=float, default=0.5, metavar='N',
                    help="the fraction of the memory")
parser.add_argument("--iteration", type=int, default=5, metavar='N',
                    help='the number of overall iterations (including sample generation)')
parser.add_argument("--lr", type=float, default=0.0001, metavar='N',
                    help="learning rate")
parser.add_argument("--images-num", type=int, default=1, metavar='N',
                    help="the number of images to use at once for generation")
parser.add_argument("--training-sample-num", type=int, default=16, metavar='N',
                    help="the number of samples to use for a training episode")
parser.add_argument("--test-sample-num", type=int, default=8, metavar='N',
                    help="the number of samples to use for testing")
parser.add_argument("--epochs", type=int, default=1, metavar='N',
                    help="traditional epoch")
parser.add_argument("--model-name", default="model", metavar='S',
                    help="the name of the model which will be loaded")
parser.add_argument("--tfr-img-id", type=int, default=0, metavar='N',
                    help="the image on which transfer learning is applied")


args = parser.parse_args()


# -----------------------------------------------
# discover the folder structure and find images
# the necessary images:
#   original: the images which will be the input normally
#   test: images what is not used for training but for testing generalization

test_folder = "test_imgs"
train_folder = "train_imgs"
test_cache = "test_cache/"
train_cache = "train_cache/"

evaluations_folder = "evals/"
models_folder = "models/"


# ------------------------------------------------
# create grey scale image at first if it was not done before

pool.converter(train_folder, train_cache)
pool.converter(test_folder, test_cache)

train_img_names = os.listdir(train_cache)
train_img_paths = [train_cache + t for t in train_img_names]

test_img_names = os.listdir(test_cache)
test_img_paths = [test_cache + t for t in test_img_names]

print(train_img_paths)
print(test_img_paths)


# -----------------------------------------
# Choose images from the given folder.
# Three consecutive images will form an Image object.
# These are: original, segmented and twin.

def choose_random_images(number, paths):

    images = []
    num_images = len(paths)
    r_list = [x for x in range(num_images)]

    if number > num_images:
        raise ValueError("There are fewer images.")

    for i in range(number):
        r = random.randint(0, num_images-1)
        img_idx = r_list[r]
        num_images -= 1
        r_list.remove(r_list[r])
        o_idx = img_idx - (img_idx % 3)

        image = pool.image_from_file(paths[o_idx], paths[o_idx + 1], paths[o_idx + 2])
        images.append(image)

    return images


# -------------------------------------------
# This function creates balanced samples.
# Meaning: the frequency of the bad and good samples are the same.
# Intuition: if it weren't for this way almost all the generated samples would
# be the bad sample.

def generate_samples(image, number):

    h = u.input_size[0]
    w = u.input_size[1]
    samples = u.TrainData(number)

    # Choosing random point in the image.
    def random_point_on_image():
        position = (random.randint(0, image.shape()[0] - (h + 1)), random.randint(0, image.shape()[1] - (w + 1)))
        return pool.sample(image, position)

    # Choosing a point which is on the segmenting curve.
    def random_point_on_segmenting():
        pts = image.get_segm_pts_list()
        rand_idx = random.randint(0, len(pts) - 1)
        position = (pts[rand_idx][0] - int(h / 2), pts[rand_idx][1] - int(w / 2))
        return pool.sample(image, position)

    def random_point_on_twin():
        pts = image.get_twin_pts_list()
        rand_idx = random.randint(0, len(pts) - 1)
        position = (pts[rand_idx][0] - int(h / 2), pts[rand_idx][1] - int(w / 2))
        return pool.sample(image, position)

    for i in range(number):

        r = random.randint(0, 1)

        if r == 1:
            if i % 10 == 0:
                x, ys, yt = random_point_on_twin()
            else:
                x, ys, yt = random_point_on_image()
        else:
            x, ys, yt = random_point_on_segmenting()

        samples.add(x, ys, yt)

    return samples


# -------------------------------------------
# Calculates the weights for each each pixel
# in the given image.
# Saves the result as a png and a json.

def calculate_then_save_weights(model, file_name, weight_file, weight_img):

    img = pool.read_picture(file_name)
    output_shape = (img.shape[0] - u.input_size[0] + 1, img.shape[1] - u.input_size[1] + 1)
    output = np.zeros(img.shape)

    x = u.TrainData(32)
    cntr = 0
    coords = []
    
    max_iter = output_shape[0] * output_shape[1]

    for c in range(output_shape[1]):
        for r in range(output_shape[0]):

            coords.append((r, c))

            chunk_x = pool.crop_out(img, (r, c), u.input_size)
            chunk_ys = np.zeros(u.output_size)
            chunk_yt = np.zeros(u.output_size)
            x.add(chunk_x, chunk_ys, chunk_yt)

            cntr += 1

            if cntr % 2000 == 0:
                print("Iteration at: " + str(cntr) + " / " + str(max_iter))

            if cntr % 32 == 0:
                result = nn.predict(model, x.get_x())

                b = 0
                for cr in coords:
                    output[cr[0] + u.input_size[0]//2 - 1, cr[1] + u.input_size[1]//2 - 1] = result[b, 0, 0, 0]
                    b += 1

                coords.clear()
                x.clear()

    # Use the last saved images.
    result = nn.predict(model, x.get_x())

    b = 0
    for cr in coords:
        output[cr[0] + u.input_size[0] - 1, cr[1] + u.input_size[1] - 1] = result[b, 0, 0, 0]
        b += 1
    
    print("Saving result to json.")
    json_string = json.dumps(output.tolist())

    with open(weight_file, "w") as f:
        f.write(json_string)

    pool.write_picture(weight_img, output*255.0)


# --------------------------------------------
# test running neural network
# generate random batch

def test_nn():

    model = nn.create_model(args.lr, args.memory)
    batch = u.TrainData(10)
    for i in range(10):
        ys = np.zeros(u.output_size, dtype=np.float32)
        yt = np.zeros(u.output_size, dtype=np.float32)
        x = pool.generate_random_image(u.input_size)
        ys += random.randint(0, 1)
        yt += random.randint(0, 1)
        batch.add(x, ys, yt)

    nn.train_batch(model, batch, 10)
    print(nn.metrics_names())
    print(nn.evaluate(model, batch))

    result_original = nn.predict(model, batch.get_x())

    nn.save_model(model, "test")
    model_loaded = nn.load_model("test", 0.8)

    result_loaded = nn.predict(model_loaded, batch.get_x())

    comp = np.isclose(result_original, result_loaded)

    print(comp)


# --------------------------------------------------------
# Training.

def train_nn(model_file_name, eval_file_name, model, train_func):

    eval_history = []

    # define functions for local usage
    def gen_data(images_, sample_num):

        chunk = u.TrainData(args.images_num * sample_num)
        for i in range(len(images_)):
            samples = generate_samples(images_[i], sample_num)
            chunk.append(samples)

        return chunk

    def save_test_results():

        with open(eval_file_name, "a") as f:
            for item in eval_history:
                line = ""
                for e in item:
                    line += str(e) + " "
                line += '\n'
                f.write(line)
        eval_history.clear()

    def full_test():

        norm = 0.0
        com_res = [0.0] * (len(nn.metrics_names()) // 2)

        def math_renorm(lst, term, num):
            for idx in range(len(lst)):
                lst[idx] = lst[idx] + term[idx] * num

        def math_div(lst, term):
            for idx in range(len(lst)):
                lst[idx] = lst[idx] / term

        print("--- Test on each possible pixel on each test picture ---")

        idx = 0
        while idx < len(test_img_paths):
            img = pool.image_from_file(test_img_paths[idx], test_img_paths[idx + 1], test_img_paths[idx + 2])
            idx += 3

            output_shape = (img.shape()[0] - u.input_size[0], img.shape()[1] - u.input_size[1])

            data = u.TrainData(32)
            cntr = 0

            max_iter = output_shape[0] * output_shape[1]

            for c in range(output_shape[1]):
                for r in range(output_shape[0]):
                    x, ys, yt = pool.sample(img, (r, c))
                    data.add(x, ys, yt)

                    cntr += 1

                    if cntr % 2000 == 0:
                        print("Iteration at: " + str(cntr) + " / " + str(max_iter))

                    if cntr % 32 == 0:
                        result = nn.evaluate(model, data)
                        math_renorm(com_res, result, 32.0)
                        norm += 32.0
                        data.clear()

            # Use the last saved images.
            result = nn.evaluate(model, data)
            math_renorm(com_res, result, data.batch)
            norm += float(data.batch)

        math_div(com_res, norm)
        com_res[0] = 0.0
        com_res[1] = 0.0
        com_res += ["full_test"]
        return com_res

    with open(eval_file_name, "w") as f:
        line = "iteration: " + str(args.iteration)
        line += " lr: " + str(args.lr)
        line += " images_num: " + str(args.images_num)
        line += " training_sample_num: " + str(args.training_sample_num)
        line += " test_sample_num: " + str(args.test_sample_num)
        line += " epochs: " + str(args.epochs)
        line += " eval_file_name: " + eval_file_name
        f.write(line + '\n')

        line = ""
        for item in nn.metrics_names():
            line += item + " "
        line += "iteration "
        f.write(line + "\n")

    print("Training started.")

    for i in range(args.iteration):

        if i % 500 == 0:  # 500
            print("Currently at: " + str(i))

        # normal samples
        images = choose_random_images(args.images_num, train_img_paths)
        data_chunk = gen_data(images, args.training_sample_num)

        result = train_func(model, data_chunk, args.epochs)

        # test
        if i % 500 == 0:  # 500
            test_images = choose_random_images(1, test_img_paths)
            test_set = gen_data(test_images, args.test_sample_num)
            result = result + nn.evaluate(model, test_set)
            result.append(i)
            eval_history.append(result)

        if i % 2000 == 0:  # 2000
            save_test_results()

    # testing on all the test images and on all the pixels
    eval_history.append(full_test())

    nn.save_model(model, model_file_name)

    save_test_results()


if args.mode == 0:

    # initialize paramters
    post_id = u.uid()
    model_file_name = models_folder + "model" + post_id + "/model"
    eval_file_name = evaluations_folder + "eval" + post_id + ".csv"

    model = nn.create_model(args.lr, args.memory)

    # Run the training and pre-calculations.
    print("Start pre-training.")
    start_time = time.time()
    train_nn(model_file_name, eval_file_name, model, nn.train_batch)
    elapsed_time = time.time() - start_time

    hh = int(elapsed_time) // 3600
    mm = int(elapsed_time - hh * 3600.0) // 60
    ss = int(elapsed_time - hh * 3600.0 - mm * 60.0)
    print("hh:mm:ss -> " + str(hh) + ":" + str(mm) + ":" + str(ss))

elif args.mode == 1:

    test_nn()

elif args.mode == 2:

    # Calculate the weights
    model = nn.load_model(args.model_name, args.memory)

    for idx in range(0, len(test_img_paths), 3):

        path = test_img_paths[idx]
        weight_file = path.replace(".png", "_w.json")
        weight_img = path.replace(".png", "_w.png")
        calculate_then_save_weights(model, path, weight_file, weight_img)

elif args.mode == 3:

    # Set the chosen image 
    # - the test and train is on the same image
    # - the one on which the user is working 
    temp = test_img_paths[args.tfr_img_id:args.tfr_img_id+3]
    test_img_paths = temp
    train_img_paths = temp

    # initialize parameters
    post_id = u.uid()
    model_file_name = models_folder + "model_trf" + post_id + "/model"
    eval_file_name = evaluations_folder + "eval_trf" + post_id + ".csv"

    model = nn.load_model(args.model_name, args.memory)

    # Run the training.
    print("Start transfer training.")
    start_time = time.time()
    train_nn(model_file_name, eval_file_name, model, nn.transfer_training)
    elapsed_time = time.time() - start_time

    hh = int(elapsed_time) // 3600
    mm = int(elapsed_time - hh * 3600.0) // 60
    ss = int(elapsed_time - hh * 3600.0 - mm * 60.0)
    print("hh:mm:ss -> " + str(hh) + ":" + str(mm) + ":" + str(ss))
