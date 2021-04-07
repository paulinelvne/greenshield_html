import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import spectral as sp
import os

import pickle
from skimage import feature

def rank_normalizer(X):
    X = pd.DataFrame(X)
    X_n = X.rank(axis=1)/X.shape[1]
    return(X_n.values)

def scale_white_dark(img, dark_spectrum, white_spectrum):
    norm_c_img = (img - dark_spectrum)/(white_spectrum - dark_spectrum)
    norm_c_img = np.clip(norm_c_img,a_max=1,a_min=0)
    return norm_c_img

def scale_max_min(data):
    data = (data-data.min())
    data = data/data.max()
    return(data)

def plot_spectra(img, wave_lengths=(29,112,226), grid_step=20):
    sp.imshow(img, wave_lengths,aspect="auto")
    ax = plt.gca()
    ax.set_xticks(np.arange(0, img.shape[1], int(img.shape[1]/grid_step)))
    ax.set_yticks(np.arange(0, img.shape[0], int(img.shape[0]/grid_step)))
    plt.xticks(rotation=75)
    ax.grid(color='k', linestyle=':', linewidth=1)
    plt.xlabel("y",fontsize=15)
    plt.ylabel("x",fontsize=15)

def zoom(blob,img):
    x = int(blob[0])
    y = int(blob[1])
    s = int(np.ceil(blob[2])*10)
    x_zoom = (max(x-s,0),min(x+s,img.shape[0]))
    y_zoom = (max(y-s,0),min(y+s,img.shape[1]))
    img_zoom = img[x_zoom[0]:x_zoom[1],y_zoom[0]:y_zoom[1], :]
    return(img_zoom)

def save_spectrum_to_file(spectrum, id_, file):
    if os.path.isfile(file):
        spectra =  pd.read_csv(file, header=0, index_col=0)
    else:
        spectra = pd.DataFrame(columns=spectrum.index)
    spectra.loc[id_] = spectrum
    spectra.to_csv(file,header=True)
    
