import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import spectral as sp
import os
import pickle
from tensorflow import keras
from skimage import feature
from keras.models import model_from_json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from spectrum import scale_max_min
import joblib
from keras.losses import mse

def load_model(model_file_name):
    json_file = open(model_file_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_file_name+".h5")
    return(loaded_model)

def get_scores_ae(points, autoencoder):
    d = distance(points, autoencoder)
    scores = np.exp(-d)
    return(scores)
def load_svm(model_file_name):
    return(pickle.load(open(model_file_name,"rb")))

def distance(points, autoencoder):
    points_pred = autoencoder.predict(points)
    return(np.abs(points_pred - points).mean(axis=1))


def prod_probs(list_probs):
    p = list_probs[0]
    for p_ in list_probs[1:]:
        p = p * p_
    return(p)

def detect_aphid_vae_svm(points, model_files = ["loaded_model_v2_sqrtmse_loss"], svm_vae = "./train_models/SVM_spectral_classifier_MSE.pkl"):
    loaded_models = [load_model(model_file) for model_file in model_files ]
    svm_mse = joblib.load(svm_vae)
    # vae_predict (centroids)
    for loaded_model in loaded_models:
        points_pred = loaded_model.predict(points)
        mse_vae = mse(points, points_pred)
        prediction_svm = (svm_mse.predict(np.array(mse_vae).reshape(-1,1)))
    # mse(centroids pred, centroids)
    # svm(mse)
    return (prediction_svm)
    
    
def detect_aphid(points, model_files, svm_model_pf="SVM_pf_rank.model", svm_model_p="SVM_p_rank.model"):

    loaded_models = [load_model(model_file) for model_file in model_files ]
    svm_pf = load_svm(svm_model_pf)
    svm_p = load_svm(svm_model_p)

    scores_aes = [get_scores_ae(points, loaded_model) for loaded_model in loaded_models]
    scores_aes = [scale_max_min(scores_ae) for scores_ae in scores_aes]

    scores_svm = get_scores_svm(points, svm_pf, svm_p)
    scores_svm = scale_max_min(scores_svm)

    scores_ae = prod_probs(scores_aes)#+[scores_svm])
    return(scores_ae * scores_svm)

def get_scores_svm(points, svm_pf, svm_p):
    scores_pf = svm_pf.score_samples(points)
    scores_p = svm_p.score_samples(points)
    scores_pf = (scores_pf - scores_pf.mean())/scores_pf.std()
    scores_p = (scores_p - scores_p.mean())/scores_p.std()
    scores = scores_pf*(scores_pf>scores_p) + scores_p*(scores_pf<scores_p)
    return(scores)

def blob_detector(img, min_sigma=1, max_sigma=8, threshold=0.4, overlap=0.1):
    blobs_dog = feature.blob_dog(img,
                                 min_sigma=min_sigma,
                                 max_sigma=max_sigma,
                                 threshold=threshold,
                                 overlap=overlap)
    return(blobs_dog)


def extract_aphid_spectrum(img_zoom, pca_pc_chosen=2, nb_clusters = 5):
    pca = PCA(n_components=pca_pc_chosen)
    data_points = img_zoom.reshape(-1,img_zoom.shape[2])
    pca = pca.fit(data_points)
    data_points_pca = pca.transform(data_points)
    z_scored = (data_points_pca - data_points_pca.mean(axis=0)) / data_points_pca.std(axis=0)
    kmeans = KMeans(n_clusters=nb_clusters)
    kmeans.fit(z_scored)
    cluster_membership = kmeans.predict(z_scored)
    cluster_membership = cluster_membership.reshape((img_zoom.shape[0],img_zoom.shape[1]))
    return cluster_membership

def cluster_pixels_kmeans(pixels, k):
    km = KMeans(k)
    km.fit(pixels)
    c = km.predict(pixels)
    return(km)
