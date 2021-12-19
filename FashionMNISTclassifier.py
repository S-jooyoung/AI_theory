# 라이브러리 호출
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#데이터 불러오기
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data() 

#데이터 이미지 확인
class_names = ["T-shirt/top" , "Trouser", "Pullover", "Dress", "Coat",
               "Sandal" , "Shirt", "Sneaker", "Bag", "Ankle boot"]

plt.figure(figsize=(10,5))

for c in range(5):
    plt.subplot(1,5,c+1)
    plt.imshow(X_train_full[c], cmap ="gray")
    plt.title(class_names[y_train_full[c]])
    plt.axis("off")
plt.show()

#학습:검증:테스트 분류
X_valid, X_train = X_train_full[:5000]/ 255.,  X_train_full[5000:] /255.
y_valid, y_train = y_train_full[:5000] , y_train_full[5000:] 
X_test = X_test /255.

print(" 학습:검증:테스트 데이터의 개수  = {}:{}:{}".format(len(X_train)
                                              ,len(X_valid),len(X_test)))

#Build a model

# (1) 분류기
np.random.seed(42)
tf.random.set_seed(42)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
model.add(tf.keras.layers.Dense(100, activation = "relu"))
model.add(tf.keras.layers.Dense(50, activation = "relu"))
model.add(tf.keras.layers.Dense(10, activation = "softmax"))

model.summary()
"""
# 강의노트 분류기
np.random.seed(42)
tf.random.set_seed(42)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
model.add(tf.keras.layers.Dense(300, activation = "relu"))
model.add(tf.keras.layers.Dense(100, activation = "relu"))
model.add(tf.keras.layers.Dense(10, activation = "softmax"))

model.summary()
"""

#Set-up training, compile()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "sgd",
              metrics = ["accuracy"])
# sgd = "Stochastic Gradient Descent" 

#Training model, fit()
#모델의 weights와 bias값을 경사하강법(GDA)를 이용해서 학습을 통해 결정
# verbose -> (1 =자세하게), (2 = 간략하게) 
history = model.fit(X_train, y_train, epochs=30,batch_size = 32, verbose =2,
                    validation_data = (X_valid, y_valid))