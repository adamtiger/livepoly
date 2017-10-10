'''
This module is for managing the whole training process for live-polyline.

'''

import os
import random
import json
import pool
import numpy as np
import nn

# discover the folder structure and find images
# file name with ending '_orig' is the original version

orig_folder = "orig_imgs"
new_folder = "imgs"
model_file_name = "model.cntk"

# create grey scale image at first if it was not done before
pool.converter(orig_folder, new_folder)

img_names = os.listdir(new_folder)
img_paths = [new_folder + "/" + t for t in img_names]

print (img_paths)


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

    h = nn.input_size[0]
    w = nn.input_size[1]
    samples = nn.TrainData(number)

    # Choosing random point in the image.
    def random_point_on_image():
        position = (random.randint(0, image.shape()[0] - (h + 1)), random.randint(0, image.shape()[1] - (w + 1)))
        x = pool.crop_out(image.orig(), position, nn.input_size)
        y = pool.check_segmentation(image.segm(), (position[0] + int(h / 2) - 1, position[1] + int(w / 2) - 1))

        return x, y

    # Choosing a point which is on the segmenting curve.
    def random_point_on_segmenting():
        pts = image.get_segm_pts_list()
        rand_idx = random.randint(0, len(pts) - 1)
        position = (pts[rand_idx][0] - int(h / 2), pts[rand_idx][1] - int(w / 2))
        x = pool.crop_out(image.orig(), position, nn.input_size)
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


def calculate_weights(model, file_name, output_file_name):

    img = pool.read_image(file_name)
    output_shape = (img.shape[0] - nn.input_size[0] + 1, img.shape[1] - nn.input_size[1] + 1)
    output = np.zeros(img.shape)

    x = nn.TrainData(32)
    cntr = 0
    coords = []
    
    max_iter = output_shape[0] * output_shape[1]

    for c in range(output_shape[1]):
        for r in range(output_shape[0]):

            coords.append((r, c))

            chunk_x = pool.crop_out(img, (r, c), nn.input_size)
            chunk_y = np.zeros(nn.output_size)
            x.add(chunk_x, chunk_y)

            cntr += 1

            if cntr % 2000 == 0:
                print("Iteration at: " + str(cntr) + " / " + str(max_iter))

            if cntr % 32 == 0:
                result = model.predict(x.get_x(), batch_size=32)

                b = 0
                for cr in coords:
                    output[cr[0] + nn.input_size[0] - 1, cr[1] + nn.input_size[1] - 1] = result[b, :, :, 0]
                    b += 1

                coords.clear()
                x.clear()

    result = model.predict(x.get_x(), batch_size=32)

    b = 0
    for cr in coords:
        output[cr[0] + nn.input_size[0] - 1, cr[1] + nn.input_size[1] - 1] = result[b, :, :, 0]
        b += 1
    
    print("Saving result to json.")
    json_string = json.dumps(output.tolist())

    with open(output_file_name, "w") as f:
        f.write(json_string)


# test running neural network
# generate random batch
def test_nn():

    model = nn.create_model()
    batch = nn.TrainData(10)
    y = np.zeros(nn.output_size, dtype=np.float32)
    for i in range(10):
        x = pool.generate_random_image(nn.input_size)
        y += random.randint(0, 1)
        batch.add(x, y)

    nn.train_batch(model, batch, 10, 10)
    print(model.metrics_names)
    print(nn.evaluate(model, batch, 10))


def train_nn():

    # parameters
    iteration = 20000
    images_num = 1
    training_sample_num = 32
    test_sample_num = 16
    batch_size = 32
    epochs = 10
    eval_file_name = "eval.txt"

    def gen_data(images_, sample_num):

        chunk = nn.TrainData(images_num * sample_num)
        for i in range(len(images_)):
            samples = generate_samples(images_[i], sample_num)
            chunk.append(samples)

        return chunk

    model = nn.create_model()

    eval_history = []
    print ("Training started.")

    for i in range(iteration):
        if i % 500 == 0:
            print("Currently at: " + str(i))
        images = choose_random_images(images_num)
        data_chunk = gen_data(images, training_sample_num)

        nn.train_batch(model, data_chunk, batch_size, epochs)

        if i % 100 == 0:
            test_set = gen_data(images, test_sample_num)
            eval_history.append(nn.evaluate(model, test_set, batch_size))

    nn.save_model(model, model_file_name)

    with open(eval_file_name, "w") as f:
        for item in eval_history:
            line = "Data: " + str(item[0]) + " " + str(item[1]) + "\n"
            f.write(line)


#test_nn()

# Run the training and precalculations.

train_nn()

# Calculate the weights
step_in = False
if step_in:

    model = nn.create_model()
    nn.load_model(model, model_file_name)

    for idx in range(0, len(img_paths), 2):

        path = img_paths[idx]
        new_fn = path.replace(".png", "_w.json")
        calculate_weights(model, path, new_fn)


