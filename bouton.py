import spectral as sp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from spectrum import *
from spectrum_ml import *
from ipywidgets import interact, interactive, widgets
from IPython.display import display
import os, os.path, io
from tkinter import Button
from tkinter import Label
from tkinter import StringVar
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.stats import rankdata
import sklearn
from keras.layers import Input, Dense, Conv1D, Conv2DTranspose,Lambda,Flatten,Reshape
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras.utils import plot_model
from keras.losses import mse, binary_crossentropy
from keras.losses import MAPE, cosine_similarity, MSLE, mae
import tensorflow as tf
from sklearn import metrics,svm
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def chargement_environnement():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    print("Environnement Chargé")

def Upload_CSV():
    wl = widgets.FileUpload(
    accept='', 
    multiple=False,  
    description = 'Fichier CSV'
    )

    return wl

def nom_fichier():
    id_ = widgets.Text()

    options1 = {
        'title': 'Exp_name',
    }

    def name1(x):
        id_.value = str(x)

    return (interactive(name1, x = "nom_fichier", options = options1), options1, id_)

def open_image_spectral():
    FILETYPES = [ ("All files", "*") ]
    root = Tk()
    img = sp.open_image(askopenfilename(filetypes=FILETYPES))
    root.mainloop()
    return img

def bouton_clustering(wl, img, dark_ref, white_ref, id_):

    def on_button_clicked1(b):
        with output1:
            output1.clear_output()
            
            wave_lengths = pd.read_csv(io.BytesIO(wl.value[list(wl.value.keys())[0]]['content']), header=None, index_col=0)[1]
            dark_spectrum = dark_ref.asarray().reshape(-1,dark_ref.shape[2]).mean(axis=0)
            white_spectrum = white_ref.asarray().reshape(-1,white_ref.shape[2]).mean(axis=0)
            
            img_norm = scale_white_dark(img.asarray(),dark_spectrum=dark_spectrum,white_spectrum=white_spectrum)
            img_norm = scale_max_min(img_norm)
            img_norm_reshape = img_norm.reshape(-1,img_norm.shape[-1])
            
            pixels_sample = img_norm_reshape[np.random.choice(img_norm_reshape.shape[0], 5000, replace=False)]
            km = cluster_pixels_kmeans(pixels_sample,100)
            c = km.predict(img_norm_reshape)
            
            sns.heatmap(c.reshape(img_norm.shape[:2]))
            plt.show()
            
            cluster_scores = detect_aphid(km.cluster_centers_,
                    model_files=["ML_models/model_vae_1Dconv"],
                    svm_model_pf='ML_models/SVM_pf_rank.model',
                    svm_model_p='ML_models/SVM_p_rank.model')
            
            best_cluster = np.argmax(cluster_scores)
            print("Meilleur Cluster", best_cluster)
            
            plt.plot(cluster_scores,"o:")
            plt.show()
            
            len(km.cluster_centers_[1])
            
            cluster_scores_vae = detect_aphid_vae_svm(km.cluster_centers_,
                    model_files=["ML_models/loaded_model_v2_sqrtmse_loss"],
                    svm_vae='./train_models/SVM_spectral_classifier_MSE.pkl')
            
            best_cluster_vae = np.argmax(cluster_scores_vae)
            print("Meilleur Cluster", best_cluster_vae)
            
            plt.plot(cluster_scores_vae,"o:")
            plt.show()
            
            sns.heatmap((c==best_cluster).reshape(img_norm.shape[:2]))
            plt.show()
            sns.heatmap((c==best_cluster_vae).reshape(img_norm.shape[:2]))
            plt.show()
            
            chosen_pixels = img_norm_reshape[(c==best_cluster)]
            scores_aphids = detect_aphid(chosen_pixels,
                                model_files=["ML_models/model_vae_1Dconv"],
                                svm_model_pf='ML_models/SVM_pf_rank.model',
                                svm_model_p='ML_models/SVM_p_rank.model')
            scores_aphids_vae = detect_aphid_vae_svm(chosen_pixels,
                                model_files=["ML_models/loaded_model_v2_sqrtmse_loss"],
                                svm_vae='./train_models/SVM_spectral_classifier_MSE.pkl')
            
            plt.plot(scores_aphids)
            plt.show()
            plt.plot(scores_aphids_vae)
            plt.show()

            cc = pd.Series(c.copy())
            cc[c!=best_cluster] = 0
            cc[c==best_cluster] = 1
            cc[c==best_cluster] *= scores_aphids
            #print(cc.value_counts())
            cc2 = cc >np.percentile(cc,99.99) #>= np.max(cc) #>np.percentile(cc,99.991)
            print(cc2.value_counts())
            
            cc_vae = pd.Series(c.copy())
            print(cc_vae.value_counts())
            cc_vae[c!=best_cluster] = -1
            cc_vae[c==best_cluster] = 1
            cc_vae[c==best_cluster] *= scores_aphids_vae
            print(cc_vae.value_counts())
            cc2_vae = cc_vae >= 1 #>= np.max(cc) #>np.percentile(cc,99.991)
            cc2_vae.value_counts()
            
            sns.heatmap(cc.values.reshape(img_norm.shape[:2]))
            plt.show()
            
            lignes_colonnes = [(3,2),(2,5),()]
            
            sns.heatmap(cc2.values.reshape(img_norm.shape[:2]))
            plt.show()
            
            print(cc2.value_counts())
            #chosen_pixels = img_norm_reshape[(cc2==True)]
            chosen_pixels = img_norm_reshape[(cc>0)]
            
            print(chosen_pixels)
            c_img = chosen_pixels
            avg_spectrum = pd.Series(c_img.mean(axis=0), index = wave_lengths)
            std_spectrum = pd.Series(c_img.std(axis=0), index = wave_lengths)
            
            plt.plot(avg_spectrum, linewidth=2)
            plt.fill_between(avg_spectrum.index, (avg_spectrum-std_spectrum), (avg_spectrum+std_spectrum), color='b', alpha=.1)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.xlabel("Wave Length",fontsize=15)
            plt.show()
            
            avg_spectrum.index = avg_spectrum.index.map(str)
            
            data_set_name_avg_raw = "final_clust_test_avg.csv"
            data_set_name_std_raw = "final_clust_test_std.csv"
            
            save_spectrum_to_file(avg_spectrum, id_.value, data_set_name_avg_raw)
            save_spectrum_to_file(std_spectrum, id_.value, data_set_name_std_raw)

    button1 = widgets.Button(description="Clustering")
    output1 = widgets.Output()
    display(button1, output1)

    button1.on_click(on_button_clicked1)


def bouton_VAE(wl):

    def on_button_clicked2(b):
        with output2:
            output2.clear_output()
            
            loaded_model = tf.keras.models.model_from_json(open('./train_models/loaded_model_v2_sqrtmse_loss.json').read())
            loaded_model.load_weights("./train_models/loaded_model_v2_sqrtmse_loss.h5")
            loaded_model.compile(optimizer='RMSprop')
            SVM_MSE = joblib.load('./train_models/SVM_spectral_classifier_MSE.pkl')
            
            X = pd.read_csv("./train_models/average_profile_no_outlier.csv",index_col=[0,1,2])
            X = X.iloc[np.random.randint(0,X.shape[0],300),:]
            X += np.random.randn(*X.shape)*0.01
            neg = pd.read_csv("./train_models/spectrum_avg_norm.csv",index_col=0).iloc[:3,:]
            neg = pd.concat((neg,pd.read_csv("./train_models/feuille_spectrum_avg_norm.csv",index_col=0)))
            neg = neg.iloc[np.random.randint(0,3,300),:]
            neg += np.random.randn(*neg.shape)*0.01
            #Spectre_test = pd.read_csv("test_spectrum_avg_norm.csv",index_col=0)
            Spectre_test = pd.read_csv(io.BytesIO(wl.value[list(wl.value.keys())[0]]['content']), index_col=0)
            #Spectre_test
            
            X_test = X
            decoded_X_test = loaded_model.predict(X)
            decoded_neg = loaded_model.predict(neg)
            decoded_Spectre_test = loaded_model.predict(Spectre_test)
            
            #plt.plot(decoded_X.T)
            plt.plot(decoded_Spectre_test.T, "b")
            plt.plot(Spectre_test.T, "red")
            #plt.plot(neg[:20].T,"green", alpha = 0.2)
            #plt.plot(decoded_neg[:20].T , "black", alpha = 0.2)
            plt.show()
            
            lim_inf = 0
            lim_sup = 448

            """
            mse_feuille = np.array(mse(decoded_neg,neg.iloc))
            mse_puceron = np.array(mse(decoded_X, X.iloc))
            mse_Spectre = np.array(mse(decoded_Spectre_test, Spectre_test.iloc))
            """

            mse_feuille = np.array(mse(decoded_neg[:,lim_inf:lim_sup],neg.iloc[:,lim_inf:lim_sup]))
            mse_puceron = np.array(mse(decoded_X_test[:,lim_inf:lim_sup], X_test.iloc[:,lim_inf:lim_sup]))
            mse_Spectre = np.array(mse(decoded_Spectre_test[:,lim_inf:lim_sup], Spectre_test.iloc[:,lim_inf:lim_sup]))


            sqrt_mse_p = np.array(np.sqrt(mse(decoded_X_test[:,lim_inf:lim_sup], X_test.iloc[:,lim_inf:lim_sup])))
            sqrt_mse_f = np.array(np.sqrt(mse(decoded_neg[:,lim_inf:lim_sup], neg.iloc[:,lim_inf:lim_sup])))
            sqrt_Spectre = np.array(np.sqrt(mse(decoded_Spectre_test[:,lim_inf:lim_sup], Spectre_test.iloc[:,lim_inf:lim_sup])))

            mape_feuille = np.array(MAPE(decoded_neg[:,lim_inf:lim_sup], neg.iloc[:,lim_inf:lim_sup]))
            mape_puceron = np.array(MAPE(decoded_X_test[:,lim_inf:lim_sup], X_test.iloc[:,lim_inf:lim_sup]))
            mape_Spectre = np.array(MAPE(decoded_Spectre_test[:,lim_inf:lim_sup], Spectre_test.iloc[:,lim_inf:lim_sup]))

            mselog_feuille = np.array(MSLE(decoded_neg[:,lim_inf:lim_sup], neg.iloc[:,lim_inf:lim_sup]))
            mselog_puceron = np.array(MSLE(decoded_X_test[:,lim_inf:lim_sup], X_test.iloc[:,lim_inf:lim_sup]))
            mselog_Spectre = np.array(MSLE(decoded_Spectre_test[:,lim_inf:lim_sup], Spectre_test.iloc[:,lim_inf:lim_sup]))
            
            bins = np.linspace(-0.001, 0.1, 150)

            plt.hist(mse_feuille, bins, alpha=0.5, label='feuille')
            plt.hist(mse_puceron, bins, alpha=0.5, label='puceron')
            plt.hist(mse_Spectre, bins, alpha=1, label = "Spectre")
            plt.ylim(-0.1,6)
            plt.legend(loc='upper left')
            plt.title("MSE")
            plt.show()
            
            bins = np.linspace(-0.0, 0.3, 150)

            plt.hist(sqrt_mse_f, bins, alpha=0.5, label='feuille')
            plt.hist(sqrt_mse_p, bins, alpha=0.5, label='puceron')
            plt.hist(sqrt_Spectre, bins, alpha=1, label = "Spectre")
            plt.ylim(-0.1,6)
            plt.legend(loc='upper left')
            plt.title("SQRT(MSE)")
            plt.show()
            
            bins = np.linspace(-0.001, 0.003, 100)

            plt.hist(mselog_feuille, bins, alpha=0.5, label='feuille')
            plt.hist(mselog_puceron, bins, alpha=0.5, label='puceron')
            plt.hist(mselog_Spectre, bins, alpha=1, label = "Spectre")
            plt.ylim(-1,10)
            plt.legend(loc='upper left')
            plt.title("Log(MSE)")
            plt.show()
            
            bins = np.linspace(-0.01, 100, 150)

            plt.hist(mape_feuille, bins, alpha=0.5, label='feuille')
            plt.hist(mape_puceron, bins, alpha=0.5, label='puceron')
            plt.hist(mape_Spectre, bins, alpha=1, label = "Spectre")
            plt.legend(loc='upper left')
            plt.title("MAPE")
            plt.show()
            
            print(Spectre_test)
            print(mse_Spectre)
            PREDICTION_SPECTRE = SVM_MSE.predict(np.array(mse_Spectre).reshape(-1,1))
            print("\n Prediction du spectre :")
            list_pred = []
            for i,e in enumerate(Spectre_test.index):
                list_pred.append((str(e)+" : "+str(PREDICTION_SPECTRE[i])))
            #print(Spectre_test.index)
            #print(PREDICTION_SPECTRE)
            print(list_pred)
    
    button2 = widgets.Button(description="VAE")
    output2 = widgets.Output()
    display(button2, output2)

    button2.on_click(on_button_clicked2)