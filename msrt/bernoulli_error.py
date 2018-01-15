'''
Due to the experimental results that the Monte Carlo approximation for the error
is quite robust and accurate despite the low number of samples, it is possible to
calculate approximated errors for a given image with Bernoulli.
Here multiprocessing is kept in mind as well due to the fairly amount of calculations.
'''

from msrt import curve
from msrt import weights


# ------------------------
# Constants:

img_name = 'FangShan1_1_o.png'


# Data assembler.
def get_data():

    _, img = curve.image_reader(None, img_name, crop=True)
    sgms = curve.find_segmenting_points(img)

    lengths = [20, 30, 50, 70, 80, 90, 100, 150]
    curves = {}
    for l in lengths:
        c = []
        for n in range(100):
            c.append(curve.generate_curve(img, sgms, l))
        curves[l] = c

    inputs = []
    for ps in [0.75, 0.8, 0.85, 0.9, 0.95]:
        for pn in [0.75, 0.8, 0.85, 0.9, 0.95]:
            inputs.append([ps, pn, img, curves])

    return inputs


# Error calculator.
def mp_bernoulli_error(input):

    ps = input[0]
    pn = input[1]
    img = input[2]
    curves = input[3]

    e_rate = {}  # error rate

    # Monte Carlo on Bernoulli weight map
    for n in range(1, 5):

        weight_map = weights.bernoulli(img, ps, pn, 0.01)

        for length in curves.keys():

            temp_err = 0.0
            num = 0.0
            for c in curves[length]:

                num += 1.0
                recommended = curve.get_livepolyline(weight_map, c[0], c[-1])
                if not curve.compare_curves(recommended, c):
                    temp_err += 1.0

            temp_err = temp_err / num

            if length in e_rate:
                e_rate[length] = (e_rate[length] * (n - 1) * num + temp_err * num) / (n * num)
            else:
                e_rate[length] = temp_err

    return e_rate
