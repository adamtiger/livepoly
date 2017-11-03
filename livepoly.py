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
                    help="0: training, 1: test, 2: weight calculation")
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
parser.add_argument("--batch-size", type=int, default=8, metavar='N',
                    help="traditional batch size")
parser.add_argument("--epochs", type=int, default=1, metavar='N',
                    help="traditional epoch")
parser.add_argument("--model-name", default="model", metavar='S',
                    help="the name of the model which will be loaded")


args = parser.parse_args()

# discover the folder structure and find images
# file name with ending '_orig' is the original version

test_folder = "test_imgs"
orig_folder = "orig_imgs"
new_folder = "imgs/"
evaluations_folder = "evals/"
models_folder = "models/"

# create grey scale image at first if it was not done before
pool.converter(orig_folder, new_folder)

img_names = os.listdir(new_folder)
img_paths = [new_folder + t for t in img_names]

print(img_paths)


def choose_random_images(number):

    images = []
    num_images = len(img_paths)
    r_list = [x for x in range(num_images)]

    if number > num_images:
        raise ValueError("There are fewer images.")

    for i in range(number):
        r = random.randint(0, num_images-1)
        img_idx = r_list[r]
        num_images -= 1
        r_list.remove(r_list[r])
        
        o_idx = 0
        if img_idx % 2 == 0:
            o_idx = img_idx
        else:
            o_idx = img_idx - 1

        img_orig = pool.read_image(img_paths[o_idx])
        img_segm = pool.read_image(img_paths[o_idx+1])

        images.append(pool.Image(img_orig, img_segm))

    return images


'''
 This function creates balanced samples.
 Meaning: the frequency of the bad and good
  samples are the same. 
 Intution: if it weren't for this way
  almost all the generated samples would
  be the bad sample.
'''

def generate_samples(image, number):

    h = u.input_size[0]
    w = u.input_size[1]
    samples = u.TrainData(number)

    # Choosing random point in the image.
    def random_point_on_image():
        position = (random.randint(0, image.shape()[0] - (h + 1)), random.randint(0, image.shape()[1] - (w + 1)))
        x = pool.crop_out(image.orig(), position, u.input_size)
        y = pool.check_segmentation(image.segm(), (position[0] + int(h / 2) - 1, position[1] + int(w / 2) - 1))

        return x, y

    # Choosing a point which is on the segmenting curve.
    def random_point_on_segmenting():
        pts = image.get_segm_pts_list()
        rand_idx = random.randint(0, len(pts) - 1)
        position = (pts[rand_idx][0] - int(h / 2), pts[rand_idx][1] - int(w / 2))
        x = pool.crop_out(image.orig(), position, u.input_size)
        y = pool.check_segmentation(image.segm(), (pts[rand_idx][0], pts[rand_idx][1]))

        return x, y

    for i in range(number):

        r = random.randint(0, 1)
        x = 0
        y = 0
        if r == 1:
            x, y = random_point_on_image()
        else:
            x, y = random_point_on_segmenting()

        samples.add(x, y)

    return samples


def calculate_weights(model, file_name, weight_file, weight_img):

    img = pool.read_image(file_name)
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
            chunk_y = np.zeros(u.output_size)
            x.add(chunk_x, chunk_y)

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

    pool.write_image(weight_img, output*255.0)


# test running neural network
# generate random batch
def test_nn():

    model = nn.create_model(args.lr, args.memory)
    batch = u.TrainData(10)
    for i in range(10):
        y = np.zeros(u.output_size, dtype=np.float32)
        x = pool.generate_random_image(u.input_size)
        y += random.randint(0, 1)
        batch.add(x, y)

    nn.train_batch(model, batch, 10)
    print(nn.metrics_names(model))
    print(nn.evaluate(model, batch))

    result_original = nn.predict(model, batch.get_x())

    nn.save_model(model, "test")
    model_loaded = nn.load_model("test", 0.8)

    result_loaded = nn.predict(model_loaded, batch.get_x())

    comp = np.isclose(result_original, result_loaded)

    print(comp)


def train_nn():

    # parameters
    memory = args.memory
    iteration = args.iteration
    lr = args.lr
    images_num = args.images_num
    training_sample_num = args.training_sample_num
    test_sample_num = args.test_sample_num
    batch_size = args.batch_size
    epochs = args.epochs

    post_id = u.uid()
    model_file_name = models_folder + "model" + post_id + "/model"
    eval_file_name = evaluations_folder + "eval" + post_id + ".csv"

    eval_history = []
    model = nn.create_model(lr, memory)

    with open(eval_file_name, "w") as f:
        line = "iteration: " + str(iteration)
        line += " lr: " + str(lr)
        line += " images_num: " + str(images_num)
        line += " training_sample_num: " + str(training_sample_num)
        line += " test_sample_num: " + str(test_sample_num)
        line += " batch_size: " + str(batch_size)
        line += " epochs: " + str(epochs)
        line += " eval_file_name: " + eval_file_name
        f.write(line + '\n')

        line = ""
        for item in nn.metrics_names(model):
            line += item + " "
        line += "iteration "
        f.write(line + "\n")

    def gen_data(images_, sample_num):

        chunk = u.TrainData(images_num * sample_num)
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

    print("Training started.")

    for i in range(iteration):

        if i % 500 == 0:
            print("Currently at: " + str(i))

        images = choose_random_images(images_num)
        data_chunk = gen_data(images, training_sample_num)

        result = nn.train_batch(model, data_chunk, epochs)

        if i % 100 == 0:
            test_set = gen_data(images, test_sample_num)
            result = result + nn.evaluate(model, test_set)
            result.append(i)
            eval_history.append(result)

        if i % 1000:
            save_test_results()

    nn.save_model(model, model_file_name)

    save_test_results()


if args.mode == 0:

    # Run the training and pre-calculations.
    start_time = time.time()
    train_nn()
    elapsed_time = time.time() - start_time

    hh = int(elapsed_time)//3600
    mm = int(elapsed_time - hh * 3600.0) // 60
    ss = int(elapsed_time - hh * 3600.0 - mm * 60.0)
    print("hh:mm:ss -> " + str(hh) + ":" + str(mm) + ":" + str(ss))

elif args.mode == 1:

    test_nn()

elif args.mode == 2:

    # Calculate the weights
    model = nn.load_model(args.model_name, args.memory)

    for idx in range(0, len(img_paths), 2):

        path = img_paths[idx]
        weight_file = path.replace(".png", "_w.json")
        weight_img = path.replace(".png", "_w.png")
        calculate_weights(model, path, weight_file, weight_img)


