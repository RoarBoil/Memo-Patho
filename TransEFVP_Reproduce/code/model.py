'''import os
import numpy as np
import tensorflow as tf
from Encoder import Encoder
from tensorflow.keras import layers, models, optimizers



def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)
    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
        p_t = y_true*y_pred + (tf.ones_like(y_true)-y_true)*(tf.ones_like(y_true)-y_pred) + tf.keras.backend.epsilon()
        focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true)-p_t),gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return binary_focal_loss_fixed

def get_model():
    inputESM = tf.keras.layers.Input(shape=(6, 1280))
    inputProtT5 = tf.keras.layers.Input(shape=(4, 1024))
    
    # ESM 分支
    featureESM = tf.keras.layers.Dense(512)(inputESM)
    featureESM = Encoder(4, 512, 8, 1024, rate=0.3)(featureESM)
    featureESM = tf.keras.layers.Flatten()(featureESM)
    
    # ProtT5 分支
    featureProtT5 = tf.keras.layers.Dense(512)(inputProtT5)
    featureProtT5 = Encoder(4, 512, 8, 1024, rate=0.3)(featureProtT5)
    featureProtT5 = tf.keras.layers.Flatten()(featureProtT5)
    
    # 特征拼接
    featureConcat = tf.keras.layers.Concatenate()([featureESM, featureProtT5])
    
    # 全连接层
    feature = tf.keras.layers.Dense(512, activation='relu')(featureConcat)
    feature = tf.keras.layers.Dense(256, activation='relu')(feature)
    feature = tf.keras.layers.Dense(128, activation='relu')(feature)
    feature = tf.keras.layers.Dropout(0.1)(feature)
    
    # 输出层
    y = tf.keras.layers.Dense(1, activation='sigmoid')(feature)

    # 创建模型
    model = tf.keras.models.Model(inputs=[inputESM, inputProtT5], outputs=y)
    
    # 优化器
    adam = tf.keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0, clipvalue=0.5)

    # 编译模型
    model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], optimizer=adam, metrics=['accuracy'])
    
    # 自定义回调函数，输出验证集结果
    class ValidationCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_data):
            self.val_data = val_data  # 传入验证集数据

        def on_epoch_end(self, epoch, logs=None):
            val_loss = logs.get('val_loss')
            val_acc = logs.get('val_accuracy')
            print(f"Epoch {epoch+1}: val_loss = {val_loss}, val_accuracy = {val_acc}")
            
            # 获取验证集的真实标签和预测结果
            X_val, y_val = self.val_data
            y_pred = self.model.predict(X_val)
            y_pred_binary = (y_pred > 0.5).astype(int)  # 将概率值转为0/1分类
            
            # 输出部分预测结果和真实标签
            print("\n部分验证集预测 vs 真实标签：")
            for i in range(10):  # 输出前10个样本
                print(f"预测: {y_pred_binary[i][0]}, 真实: {y_val[i]}")
                print(y_pred)
            # 统计预测类别分布
            unique, counts = np.unique(y_pred_binary, return_counts=True)
            print(f"预测类别分布: {dict(zip(unique, counts))}\n")

            

    # 返回模型和回调
    return model, ValidationCallback
'''
    
import os
import numpy as np
import tensorflow as tf
from Encoder import Encoder


def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)
    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
        p_t = y_true*y_pred + (tf.ones_like(y_true)-y_true)*(tf.ones_like(y_true)-y_pred) + tf.keras.backend.epsilon()
        focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true)-p_t),gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return binary_focal_loss_fixed


def get_model():
    inputESM = tf.keras.layers.Input(shape=(6, 1280))
    inputProtT5 = tf.keras.layers.Input(shape=(4, 1024))
    featureESM = tf.keras.layers.Dense(512)(inputESM)
    featureESM = Encoder(4, 512, 8, 1024, rate=0.3)(featureESM)
    featureESM = tf.keras.layers.Flatten()(featureESM)
    featureProtT5 = tf.keras.layers.Dense(512)(inputProtT5)
    featureProtT5 = Encoder(4, 512, 8, 1024, rate=0.3)(featureProtT5)
    featureProtT5 = tf.keras.layers.Flatten()(featureProtT5)
    featureConcat = tf.keras.layers.Concatenate()([featureESM, featureProtT5])
    feature = tf.keras.layers.Dense(512, activation='relu')(featureConcat)
    feature = tf.keras.layers.Dense(256, activation='relu')(feature)
    feature = tf.keras.layers.Dense(128, activation='relu')(feature)
    feature = tf.keras.layers.Dropout(0.1)(feature)
    y = tf.keras.layers.Dense(1, activation='sigmoid')(feature)

    model = tf.keras.models.Model(inputs=[inputESM, inputProtT5], outputs=y)
    adam = tf.keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0, clipvalue=0.5)
    model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

