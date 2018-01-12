from msrt import curve
from msrt import validation as vldt
import numpy as np
from scipy import misc

_, img = curve.image_reader(None, vldt.img_name)

cv = curve.get_livepolyline(img, vldt.p0, vldt.p1)

out = np.zeros(img.shape)
for p in cv:
    out[p[0], p[1]] = 255

misc.imsave('new_' + vldt.img_name, out)
