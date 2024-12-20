import tifffile as tiff
import glob
import numpy as np
from utils.data_utils import imagesc
import scipy.ndimage#.morphology.distance_transform_edt
import matplotlib.pyplot as plt
from skimage import measure

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def get_chain_code(binary_image):
    # Get contour using skimage
    #contours = measure.find_contours(binary_image, 0.5)

    #if not contours:
    #    return []

    # Use the first contour
    #contour = contours[0]

    contours = measure.find_contours(xzm[180, :, :])

    chain_code = []

    # Calculate chain code for each point
    for contour in contours:
        for i in range(len(contour) - 1):
            curr = contour[i]
            next_point = contour[i + 1]

            # Calculate direction
            dy = next_point[0] - curr[0]
            dx = next_point[1] - curr[1]

            # Convert to chain code (8-directional)
            # 3 2 1
            # 4 x 0
            # 5 6 7
            angle = np.arctan2(dy, dx)
            code = int(((angle + np.pi) * 4 / np.pi + 0.5) % 8)

            chain_code.append(code)

    plt.plot(chain_code);plt.show()
    analyze_chain_code(chain_code)

    return chain_code


def analyze_chain_code(chain_code):
    # Count direction changes
    changes = sum(1 for i in range(len(chain_code) - 1)
                  if chain_code[i] != chain_code[i + 1])

    # Get direction frequencies
    frequencies = np.bincount(chain_code, minlength=8)

    # Calculate dominant directions
    dominant_dirs = np.argsort(frequencies)[-2:]

    return {
        'direction_changes': changes,
        'direction_frequencies': frequencies,
        'dominant_directions': dominant_dirs
    }


root = '/media/ghc/Ghc_data3/OAI_diffusion_final/isotropic_outs/out/'

list_xz = sorted(glob.glob(root + 'xz/*.tif'))
list_xz2d = sorted(glob.glob(root + 'xz2d/*.tif'))

# Initialize arrays with dictionary comprehension
measurements = ['e_xy', 'e_xy2d', 'a_xy', 'a_xy2d',
                'e_xz', 'e_xz2d', 'a_xz', 'a_xz2d',
                'e_yz', 'e_yz2d', 'a_yz', 'a_yz2d']

arrays = {key: np.array([]) for key in measurements}

for i in tqdm(range(len(list_xz))[:100]):
    xz = tiff.imread(list_xz[i])
    xz2d = tiff.imread(list_xz2d[i])

    xzm = (xz >= 0.4) / 1
    xz2dm = (xz2d >= np.percentile(xz2d, 100 - 100 * xzm.mean())) / 1

    xzm = scipy.ndimage.morphology.distance_transform_edt(xzm)
    xz2dm = scipy.ndimage.morphology.distance_transform_edt(xz2dm)
    xzme = (xzm <= 1) & (xzm > 0)
    xz2dme = (xz2dm <= 1) & (xz2dm > 0)

    # Initialize counters
    counters = {key: 0 for key in measurements}

    for i in range(xz.shape[0]):
        #e = (scipy.ndimage.morphology.distance_transform_edt(xzm[i, :, :]) == 1).sum()
        #e2d = (scipy.ndimage.morphology.distance_transform_edt(xz2dm[i, :, :]) == 1).sum()

        e = xzme[i, :, :].sum()
        e2d = xz2dme[i, :, :].sum()

        counters['e_xz'] += e
        counters['e_xz2d'] += e2d
        counters['a_xz'] += xzm[i, :, :].sum()
        counters['a_xz2d'] += xz2dm[i, :, :].sum()

    for i in range(xz.shape[1]):
        #e = (scipy.ndimage.morphology.distance_transform_edt(xzm[:, i, :]) == 1).sum()
        #e2d = (scipy.ndimage.morphology.distance_transform_edt(xz2dm[:, i, :]) == 1).sum()

        e = xzme[:, i, :].sum()
        e2d = xz2dme[:, i, :].sum()

        counters['e_yz'] += e
        counters['e_yz2d'] += e2d
        counters['a_yz'] += xzm[:, i, :].sum()
        counters['a_yz2d'] += xz2dm[:, i, :].sum()

    for i in range(xz.shape[2]):
        #e = (scipy.ndimage.morphology.distance_transform_edt(xzm[:, :, i]) == 1).sum()
        #e2d = (scipy.ndimage.morphology.distance_transform_edt(xz2dm[:, :, i]) == 1).sum()

        e = xzme[:, :, i].sum()
        e2d = xz2dme[:, :, i].sum()

        counters['e_xy'] += e
        counters['e_xy2d'] += e2d
        counters['a_xy'] += xzm[:, :, i].sum()
        counters['a_xy2d'] += xz2dm[:, :, i].sum()

    # Update arrays
    for key in arrays:
        arrays[key] = np.append(arrays[key], counters[key])  # remove 'a_' prefix to match counter keys

# If you need individual variables at the end:
#a_e_xz, a_e_xz2d, a_a_xz, a_a_xz2d = [arrays[key] for key in ['a_e_xz', 'a_e_xz2d', 'a_a_xz', 'a_a_xz2d']]


rxz = np.array(arrays['e_xz']) / (np.array(arrays['a_xz']) + 1)
rxz2d = np.array(arrays['e_xz2d']) / (np.array(arrays['a_xz2d']) + 1)
ryz = np.array(arrays['e_yz']) / (np.array(arrays['a_yz']) + 1)
ryz2d = np.array(arrays['e_yz2d']) / (np.array(arrays['a_yz2d']) + 1)
rxy = np.array(arrays['e_xy']) / (np.array(arrays['a_xy']) + 1)
rxy2d = np.array(arrays['e_xy2d']) / (np.array(arrays['a_xy2d']) + 1)


# Create boxplot
plt.boxplot([rxz, rxz2d]);plt.show()#, labels=['Data 1', 'Data 2', 'Data 3', 'Data 4'])