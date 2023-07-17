import numpy as np
import numpy.typing as npt
from matplotlib import colors
from matplotlib import cm as cmx
import matplotlib.pyplot as plt

from domain_gap.utils.configs import IMAGE_MEAN, IMAGE_STD

# ------------------------ Color Config ------------------------
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)


def unnormalize_image(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return image * IMAGE_STD + IMAGE_MEAN