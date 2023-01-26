import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import numpy as np


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


WIDTH_REF = 50.0  # pixels
HEIGHT_REF = 50.0  # pixels

img = cv2.imread('Euromex_20.tif')
img = unsharp_mask(img, kernel_size=(19, 19), sigma=1500, threshold=50)
img = cv2.bitwise_not(img)

# alpha = 1.5 # Contrast control (1.0-3.0)
# beta = 50 # Brightness control (0-100)
#
# img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(gray, 80, 255, 0)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

res = np.array(img)

plt.imshow(img)


for c in contours:
    x, y, widthPixel, heightPixel = cv2.boundingRect(c)
    realWidth = widthPixel / WIDTH_REF * 20.0
    realHeight = heightPixel / HEIGHT_REF * 20.0
    cv2.rectangle(res, (x, y), (x + widthPixel, y + heightPixel), (255, 0, 0), 2)
    cv2.putText(res, "{0:.1f}".format(realWidth) + " x " + "{0:.1f}".format(realHeight) + " microns", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    print(realWidth, " x ", realHeight)
    if realWidth < 100:
        rect = patches.Rectangle((x,y), widthPixel, heightPixel, edgecolor='k', fill=False)
        ax = plt.gca()
        ax.add_patch(rect)

plt.show()
