import os
import numpy as np
import joblib
import tensorflow as tf
from Encoder import Encoder
from model import get_model
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, auc, roc_curve


def train():
    X_train_esm = np.load('/data2/guest1/TransEFVP-master/Last_train/test_on_NewHumansavar/combined_esm.npy')
    X_train_protT5 = np.load('/data2/guest1/TransEFVP-master/Last_train/test_on_NewHumansavar/combined_T5.npy')
    y_train = np.load('/data2/guest1/TransEFVP-master/Last_train/test_on_NewHumansavar/Sign.npy')
    
    batch_size = 1024
    earlyStopPatience = 20
    monitor = 'val_loss'
    input_train = [X_train_esm, X_train_protT5]

    encoder_model = get_model()
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=earlyStopPatience, verbose=1, mode='auto')
    log_dir = "./model/logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
    history_callback = encoder_model.fit(x=input_train, y=y_train, batch_size=batch_size, epochs=1000, verbose=2,
                                         callbacks=[earlystopping], validation_split=0.1, shuffle=True, )
    encoder_model.save_weights("/data2/guest1/TransEFVP-master/trainTrans_onmydata/model/model_model.h5")

    feature_layer = tf.keras.models.Model(inputs=encoder_model.input,
                                          outputs=encoder_model.get_layer('concatenate').output)
    train_feature_output = feature_layer.predict(input_train)
    pca_model = PCA(n_components=0.9)
    train_feature_output_pca = pca_model.fit_transform(train_feature_output)
    svc_model = SVC()
    svc_model.fit(train_feature_output_pca, y_train)
    joblib.dump(pca_model, "/data2/guest1/TransEFVP-master/trainTrans_onmydata/model/pca_model.pkl")
    joblib.dump(svc_model, "/data2/guest1/TransEFVP-master/trainTrans_onmydata/model/svc_model.pkl")


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    train()

'''
import os
import numpy as np
import joblib
import tensorflow as tf
from Encoder import Encoder
from model import get_model
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, auc, roc_curve


def train():
    X_train_esm = np.load('/data2/guest1/trainfiles/fortrain/combined_esm.npy')
    X_train_protT5 = np.load('/data2/guest1/trainfiles/fortrain/combined_T5.npy')
    y_train = np.load('/data2/guest1/trainfiles/fortrain/trainlog_shuffled.npy')
    
    batch_size = 1024
    earlyStopPatience = 15
    monitor = 'val_loss'
    input_train = [X_train_esm, X_train_protT5]

    encoder_model = get_model()
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=earlyStopPatience, verbose=1, mode='auto')
    log_dir = "./model/logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
    history_callback = encoder_model.fit(x=input_train, y=y_train, batch_size=batch_size, epochs=1000, verbose=2,
                                         callbacks=[earlystopping], validation_split=0.2, shuffle=True, )
    encoder_model.save_weights("/data2/guest1/trainfiles/fortrain/model/encoder_model.h5")

    feature_layer = tf.keras.models.Model(inputs=encoder_model.input,
                                          outputs=encoder_model.get_layer('concatenate').output)
    train_feature_output = feature_layer.predict(input_train)
    pca_model = PCA(n_components=0.9)
    train_feature_output_pca = pca_model.fit_transform(train_feature_output)
    svc_model = SVC()
    svc_model.fit(train_feature_output_pca, y_train)
    joblib.dump(pca_model, "/data2/guest1/trainfiles/fortrain/model/pca_model.pkl")
    joblib.dump(svc_model, "/data2/guest1/trainfiles/fortrain/model/svc_model.pkl")


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    train()
'''