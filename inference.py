import cv2
import argparse
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from face_parser import FaceParser

import warnings
warnings.filterwarnings("ignore")


def resize_image(im, max_size=768):
    if np.max(im.shape) > max_size:
        ratio = max_size / np.max(im.shape)
        print(f"Resize image to ({str(int(im.shape[1]*ratio))}, {str(int(im.shape[0]*ratio))}).")
        return cv2.resize(im, (0,0), fx=ratio, fy=ratio)
    return im


parsing_annos = [
    '0, background', '1, skin', '2, left eyebrow', '3, right eyebrow', 
    '4, left eye', '5, right eye', '6, glasses', '7, left ear', '8, right ear', '9, earings',
    '10, nose', '11, mouth', '12, upper lip', '13, lower lip', 
    '14, neck', '15, neck_l', '16, cloth', '17, hair', '18, hat'
]


def show_parsing_with_annos(data):
    fig, ax = plt.subplots(figsize=(8,8))
    #get discrete colormap
    cmap = plt.get_cmap('gist_ncar', len(parsing_annos))
    new_colors = cmap(np.linspace(0, 1, len(parsing_annos)))
    new_colors[0, :] = np.array([0, 0, 0, 1.])
    new_cmap = ListedColormap(new_colors)
    
    # set limits .5 outside true range
    mat = ax.matshow(data, cmap=new_cmap, vmin=-0.5, vmax=18.5)
    
    #tell the colorbar to tick at integers    
    cbar = fig.colorbar(mat, ticks=np.arange(0, len(parsing_annos)))
    cbar.ax.set_yticklabels(parsing_annos)
    plt.axis('off')
    fig.show()

parser = argparse.ArgumentParser()
parser.add_argument("filepath", help="display a square of a given number")


if __name__ == 'main':
    args = parser.parse_args()
    prs = FaceParser()
    im = cv2.imread(args.filepath)[..., ::-1]
    im = resize_image(im) # Resize image to prevent GPU OOM.
    out = prs.parse_face(im)
    h, w, _ = im.shape
    plt.imshow(im)
    show_parsing_with_annos(out[0])
