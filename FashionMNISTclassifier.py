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


