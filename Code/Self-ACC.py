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
model = tf.keras.Model(inputs=input1, outputs=output1)
input11=np.load('D:\\dataset\\data30\\train_decoder.npy')
train_labels = np.load('D:\\dataset\\data30\\train_label.npy')
iinput1=np.load('D:\\dataset\\data30\\test_decoder.npy')
ttrain_labels = np.load('D:\\dataset\\data30\\test_label.npy')
checkpoint_path = "D:\\dataset\\data30\\weight\\{epoch}.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)  
# model.load_weights("D:\\dataset\\data0_3\\15.hdf5")
# optimizer = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics = ['categorical_accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives()]
              )
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)
model.load_weights('D:\\dataset\\data30\\weight\\1.hdf5')
# H = model.fit(input11,train_labels,batch_size=50,validation_data=(iinput1,ttrain_labels),
#                               epochs=15,
#                               # validation_steps=50,
#                               verbose=1,
#                               # callbacks=[cp_callback],
#                               shuffle=True
#                               )



# loss,accuracy = model.evaluate(iinput1,ttrain_labels,batch_size=50)
# print('accuracy',accuracy)
num=35

num2=2606
layer_outputs = [layer.output for layer in model.layers[:num]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 
img=iinput1[num2]
plt.figure()
plt.imshow(img)
input_image=tf.expand_dims(img, 0) 
activations = activation_model.predict(input_image) 
print(activations[0].shape)
# print(iinput1[0].shape)
plt.matshow(activations[3][0,:,:,0], cmap='viridis') 
plt.matshow(activations[3][0,:,:,10], cmap='viridis') 
layer_names = []
for layer in model.layers[:num]:
    layer_names.append(layer.name) 

images_per_row=16 

for layer_name, layer_activation in zip (layer_names[:num], activations[:num]):
    n_feature = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_feature//images_per_row
    display_grid = np.zeros((size*n_cols, images_per_row*size))
    for col in range(n_cols): 
        for row in  range (images_per_row): 
            # print(layer_activation.shape)
            # print(col*images_per_row+row)
            channel_image = layer_activation[0,:,:,col*images_per_row+row] 
            channel_image -= channel_image.mean() 
            channel_image /= channel_image.std()
            channel_image *=64
            channel_image +=128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # print(channel_image.shape)
            # print(display_grid[col*size:(col+1)*size, row*size:(row+1)*size].shape)
            display_grid[col*size:(col+1)*size, row*size:(row+1)*size] = channel_image 
    scale = 1./size 
    plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')






























