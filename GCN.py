import os
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Concatenate, Reshape, ReLU
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from lib.gcn_layer import GraphConv
from sklearn.metrics import accuracy_score, f1_score
from lib.Self_Att import Self_Attn_3D

x_train = np.load("/home/yi/LFolder/抓取/抓取稳定性分类/BiGS-dataset/Electrodes/x_train.npy")
y_train = np.load("/home/yi/LFolder/抓取/抓取稳定性分类/BiGS-dataset/label/y_train.npy")
x_test = np.load("/home/yi/LFolder/抓取/抓取稳定性分类/BiGS-dataset/Electrodes/x_test.npy")
y_test = np.load("/home/yi/LFolder/抓取/抓取稳定性分类/BiGS-dataset/label/y_test.npy")


timesteps = x_train.shape[1]
nodes = x_train.shape[2]
features = x_train.shape[3]

dropout_rate = 0
learning_rate = 5e-4

X_in = Input([timesteps, nodes, features], dtype='float32')

x = GraphConv(8, t_kernels=1)(X_in)
# x = Self_Attn_3D(8, 4)(x)    #self attention
x = ReLU()(x)


x = Flatten()(x)
# x = Dense(32)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=X_in, outputs=output)
model.summary()
model.compile(optimizer=Adam(lr=learning_rate), loss=['binary_crossentropy'], metrics=['acc'])
history = model.fit(x_train, y_train, epochs=50, batch_size=32, shuffle=False)

y_pred = model.predict(x_test)
y_pred = np.asarray(y_pred.round())
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
f1 = f1_score(y_test, y_pred, average='binary')
print("F1 score:", f1)

# 绘制损失曲线和精度曲线
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc)+1)
#
# plt.plot(epochs, acc, 'b', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'r', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('GCN')
# plt.legend()

# plt.show()