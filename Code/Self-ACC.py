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
import seaborn as sns
from tensorflow.keras import models
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
num=30 
class ResNetBlock(tf.keras.Model):
    def __init__(self,filter_num,stride=1):
        super(ResNetBlock,self).__init__()
        self.conv1=tf.keras.layers.Conv2D(filter_num,kernel_size=[3,3],strides=stride,padding='same')
        self.att=tf.keras.layers.Attention()
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
        q=self.conv1(inputs)
        k=self.conv1(inputs)
        v=self.conv1(inputs)
        out=self.att([q,k,v])

        identity=inputs
        out=tf.keras.layers.add([identity,out])
        out=self.bn1(out,training=False)
        out1=self.relu(out)
        out=self.conv2(out1)
        out=self.bn2(out,training=False)
        identity=out
        out=tf.keras.layers.add([identity,out])
        output=tf.nn.relu(out)
        return output
input1=tf.keras.Input((150,150,1))
q=tf.keras.layers.Conv2D(16,kernel_size=[3,3],strides=1,padding='same')(input1)
k=tf.keras.layers.Conv2D(16,kernel_size=[3,3],strides=1,padding='same')(input1)
v=tf.keras.layers.Conv2D(16,kernel_size=[3,3],strides=1,padding='same')(input1)
out=tf.keras.layers.Attention()([q,k,v])
out=tf.keras.layers.BatchNormalization()(out,training=False)
out=tf.keras.layers.add([input1,out])
out1=tf.keras.layers.Activation('relu')(out)
out=tf.keras.layers.Conv2D(16,kernel_size=[3,3],strides=1,padding='same')(out1)
out=tf.keras.layers.BatchNormalization()(out,training=False)
out=tf.keras.layers.add([out1,out])
output=tf.nn.relu(out)
q=tf.keras.layers.Conv2D(16,kernel_size=[3,3],strides=1,padding='same')(output)
k=tf.keras.layers.Conv2D(16,kernel_size=[3,3],strides=1,padding='same')(output)
v=tf.keras.layers.Conv2D(16,kernel_size=[3,3],strides=1,padding='same')(output)
out=tf.keras.layers.Attention()([q,k,v])
out=tf.keras.layers.BatchNormalization()(out,training=False)
out=tf.keras.layers.add([output,out])
out1=tf.keras.layers.Activation('relu')(out)
out=tf.keras.layers.Conv2D(16,kernel_size=[3,3],strides=1,padding='same')(out1)
out=tf.keras.layers.BatchNormalization()(out,training=False)
out=tf.keras.layers.add([out1,out])
output=tf.nn.relu(out)
q=tf.keras.layers.Conv2D(16,kernel_size=[3,3],strides=1,padding='same')(output)
k=tf.keras.layers.Conv2D(16,kernel_size=[3,3],strides=1,padding='same')(output)
v=tf.keras.layers.Conv2D(16,kernel_size=[3,3],strides=1,padding='same')(output)
out=tf.keras.layers.Attention()([q,k,v])
out=tf.keras.layers.BatchNormalization()(out,training=False)
out=tf.keras.layers.add([output,out])
out1=tf.keras.layers.Activation('relu')(out)
out=tf.keras.layers.Conv2D(16,kernel_size=[3,3],strides=1,padding='same')(out1)
out=tf.keras.layers.BatchNormalization()(out,training=False)
out=tf.keras.layers.add([out1,out])
output=tf.nn.relu(out)
end=tf.keras.layers.MaxPooling2D(3,3)(output)
end=tf.keras.layers.Flatten()(end)
end=tf.keras.layers.Dense(512, activation='relu')(end)
end=tf.keras.layers.Dense(256, activation='relu')(end)
output1=tf.keras.layers.Dense(num, activation='softmax')(end)
input11=np.load(r'D:\\paper1\\weight\\all.npy')
train_labels = np.load(r'D:\\paper1\\weight\\label.npy')
checkpoint_path = "D:\\dataset\\data30\\{epoch}.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)  
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)
kfold = KFold(n_splits=10,shuffle = True)  #K-fold cross validation
for train_index, test_index in kfold.split(input11):
    tf.keras.backend.clear_session()
    X_train, X_test = input11[train_index], input11[test_index]
    Y_train, Y_test = train_labels[train_index], train_labels[test_index]
    model = tf.keras.Model(inputs=input1, outputs=output1)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', 
                  metrics = ['categorical_accuracy']
                  )
    H = model.fit(X_train,Y_train,validation_data=(X_test,Y_test),batch_size=50,
                                  epochs=1,
                                  # validation_steps=50,
                                  verbose=1,
                                  # callbacks=[cp_callback],
                                  shuffle=True
                                  )























