import os
import tensorflow as tf
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA 
from sklearn import manifold 
from sklearn.manifold import TSNE
import umap
from tensorflow.keras import datasets,losses,Sequential,optimizers
import orbit
from tensorflow.keras.optimizers import RMSprop
import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from keras.utils.np_utils import to_categorical
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
def visualize_predictions(decoded,gt,samples=10):
    outputs=None
    for i in range(samples):
        original=(gt[i]*255).astype('uint8')
        recon=(decoded[i]*255).astype('uint8')
        output=np.hstack([original,recon])
        if outputs is None:
            outputs=output
        else:
            outputs=np.vstack([outputs,output])
    return outputs
class ResNetBlock(tf.keras.Model):
    def __init__(self,filter_num,stride=1):
        super(ResNetBlock,self).__init__()
        self.conv1=tf.keras.layers.Conv2D(filter_num,kernel_size=[3,3],strides=stride,padding='same')
        self.bn1=tf.keras.layers.BatchNormalization()
        self.relu=tf.keras.layers.Activation('relu')
        
        self.conv2=tf.keras.layers.Conv2D(filter_num,kernel_size=[3,3],strides=1,padding='same')
        self.bn2=tf.keras.layers.BatchNormalization()
        if stride!=1:
            self.downSample=Sequential([
                tf.keras.layers.Conv2D(filter_num,kernel_size=[1,1],strides=stride)
            ])
        else:
            self.downSample=lambda x:x
    def call(self,inputs,training=None):
        out=self.conv1(inputs)
        out=self.bn1(out,training=False)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out,training=False)
        identity=self.downSample(inputs)
        output=tf.keras.layers.add([identity,out])
        output=tf.nn.relu(output)
        return output

class ConvAutoencoder:
    @staticmethod
    def build(width,height,depth=None,filters=(32,64),latentDim=16):
        inputs=tf.keras.layers.Input(shape=(height,width,depth))
        x=inputs
        x=ResNetBlock(16,1)(x)
        x=ResNetBlock(16,1)(x)
        volumeSize=tf.keras.backend.int_shape(x)
        x=tf.keras.layers.Flatten()(x)
        latent=tf.keras.layers.Dense(latentDim)(x)
        encoder=tf.keras.Model(inputs=inputs,outputs=latent,name='encoder')
        latentinputs=tf.keras.layers.Input(shape=(latentDim,))
        x=tf.keras.layers.Dense(np.prod(volumeSize[1:]))(latentinputs)
        x=tf.keras.layers.Reshape((volumeSize[1],volumeSize[2],volumeSize[3]))(x)
        x=ResNetBlock(16,1)(x)
        x=ResNetBlock(16,1)(x)
        x=tf.keras.layers.Conv2DTranspose(depth,(3,3),padding='same')(x)
        outputs=tf.keras.layers.Activation('sigmoid')(x)
        decoder=tf.keras.Model(latentinputs,outputs,name='decoder')
        autoencoder=tf.keras.Model(inputs,decoder(encoder(inputs)),name='autoencoder')
        return (encoder,decoder,autoencoder)
list1=['3601','3602','3603', '3604','fast1', 'fast2','fast3','fast4', 'H3C1','H3C2','HK1','HK2','honor1',
       'honor2','honor3','honor4','mercury1', 'mercury2', 'mercury3','mercury4','tenda1', 'tenda2', 'tenda3', 
       'tenda4','tp1','tp2','tplink1', 'tplink2','tplink3', 'tplink4']
list2=[]
seed = 7
for i in list1:
    tf.keras.backend.clear_session()
    epochs=1
    lr=1e-3
    batch_size=10
    path=r'D:\\paper1\\pic\\'+i
    X=np.load(path+'.npy')
    kfold = KFold(n_splits=10,shuffle = True) #K-fold cross validation
    mse_scores = []
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        (encoder, decoder, autoencoder) = ConvAutoencoder.build(150, 150,1)
        opt=tf.keras.optimizers.Adam(learning_rate=lr,decay=lr/epochs)
        autoencoder.compile(loss='mse',optimizer=opt,metrics=['acc'])
        autoencoder.fit(X_train,X_train,validation_data=(X_test,X_test),epochs=epochs,batch_size=batch_size) 










