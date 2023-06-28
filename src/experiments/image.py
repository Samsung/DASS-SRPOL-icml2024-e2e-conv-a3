from pathlib import Path
from argparse import ArgumentParser
import os
import pickle
import gzip

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_file", type=Path, default="result.txt", 
                        help="Path to a result file.")
    parser.add_argument("-d", "--dataset", type=str, 
                        help="Name of a dataset. One of: cifar10 | mnist | fmnist")
    parser.add_argument('-c', '--normal_cls', type=int, default=0, 
                        help="Normal class.")
    parser.add_argument("-n", "--run_n_times", type=int, default=1,
                        help="Number of algorithm trials.")
    parser.add_argument("--gamma", type=float, default=1.0, 
                        help="E2EConvA3 gamma value.")

    return parser.parse_args()


class E2EConvA3(tf.keras.Model):
    def __init__(self, input_dim, intermediate_dims, 
                 decision_layers_intermediate_dims, n_channels=3, gamma=1.0, activation_mode='pool_to_minsize'):
        super().__init__()
        self.input_dim = input_dim
        self.intermediate_dims = intermediate_dims
        self.decision_layers_intermediate_dims = decision_layers_intermediate_dims
        self.n_channels = n_channels
        self.gamma = gamma
        self.activation_mode = activation_mode
        
        encoder_layers = []
        for dim in intermediate_dims:
            encoder_layers.append(tf.keras.layers.Conv2D(dim, (3, 3), strides=1, activation="relu", padding="same"))
            encoder_layers.append(tf.keras.layers.MaxPooling2D())
            encoder_layers.append(tf.keras.layers.BatchNormalization())
                               
        self.encoder = tf.keras.Sequential(encoder_layers, name='encoder')
        
        decoder_layers = []
        
        for dim in intermediate_dims[::-1]:
            decoder_layers.append(tf.keras.layers.Conv2DTranspose(dim, (3, 3), activation="relu", strides=2, padding="same"))
            decoder_layers.append(tf.keras.layers.BatchNormalization())
        
        decoder_layers.append(tf.keras.layers.Conv2DTranspose(self.n_channels, (3, 3), activation="sigmoid", padding="same"))
            
        self.decoder = tf.keras.Sequential(decoder_layers, name='decoder')
        
        decision_layers = []
        for dim in self.decision_layers_intermediate_dims:
            decision_layers.append(tf.keras.layers.Conv2D(dim, (3, 3), strides=1, activation="relu", padding="same"))
            decision_layers.append(tf.keras.layers.MaxPooling2D())
        decision_layers.append(tf.keras.layers.Flatten())
        decision_layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        self.decision_module = tf.keras.Sequential(decision_layers, name='decision_module')
        
        self.clf_loss_fn = tf.keras.losses.BinaryCrossentropy(name='clf_loss')
        
        self.recon_loss_tracker = tf.keras.metrics.Mean(name='recon_loss')
        self.clf_loss_tracker = tf.keras.metrics.Mean(name='clf_loss')
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        
    def call_ae(self, inputs, training=False):
        z = self.encoder(inputs)
        recon_x = self.decoder(z)
        return recon_x
        
    def call_decision(self, inputs, training=False):
        pred = self.decision_module(inputs)
        return pred
        
    def call(self, inputs, training=False):
        x = inputs
        recon_x = self.call_ae(x, training=training)
        activations = self.activations(x)
        pred = self.call_decision(activations, training=training)
        return recon_x, pred
    
    def activations(self, x):
        ret = []
        layer_idx = 0
        last_pad_dim = None
        for l in self.encoder.layers:
            x = l(x)
            if 'conv' in l.name:
                if self.activation_mode == 'flatten':
                    ret.append(tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])]).numpy())
                elif self.activation_mode == 'average_kernels':
                    ret.append(tf.reduce_mean(x, axis=1))
                elif self.activation_mode == 'pool_to_minsize':
                    ret.append(tf.nn.max_pool(x,
                                              ksize=2**(len(self.intermediate_dims)-layer_idx),
                                              strides=2**(len(self.intermediate_dims)-layer_idx),
                                              padding="SAME"))
                elif self.activation_mode == 'pad_to_maxsize':
                    if last_pad_dim is None:
                        last_pad_dim = x.shape[1]
                    current_pad_dim = x.shape[1]
                    ret.append(tf.pad(x, tf.constant([[0, 0], [int((last_pad_dim - current_pad_dim) / 2)] * 2, 
                                                      [int((last_pad_dim - current_pad_dim) / 2)] * 2, [0, 0]]),
                                      'CONSTANT'))
                else:
                    ret.append(x)
                layer_idx += 1
        if self.activation_mode in ('flatten', 'average_kernels'):
            ret = tf.concat(ret, axis=1)
            return np.vstack(ret)
        elif self.activation_mode in ('pool_to_minsize', 'pad_to_maxsize'):
            ret = tf.concat(ret, axis=-1)
            return ret
        ret = [tf.expand_dims(tensor, axis=-1) for tensor in ret]
        ret = tf.concat(ret, axis=-1)
        return ret
    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            recon_x, pred = self(x, training=True)
            is_typical = tf.cast((y == 0), dtype=tf.float32)
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - recon_x)) * is_typical)
            clf_loss = self.clf_loss_fn(y, pred)
            loss = recon_loss + self.gamma * clf_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
            
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.recon_loss_tracker.update_state(recon_loss)
        self.clf_loss_tracker.update_state(clf_loss)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "clf_loss": self.clf_loss_tracker.result()}
    
    def test_step(self, data):
        x, y = data
        recon_x, pred = self(x, training=False)
        is_typical = tf.cast((y == 0), dtype=tf.float32)
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - recon_x)) * is_typical)
        clf_loss = self.clf_loss_fn(y, pred)
        loss = recon_loss + self.gamma * clf_loss
        
        self.recon_loss_tracker.update_state(recon_loss)
        self.clf_loss_tracker.update_state(clf_loss)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "clf_loss": self.clf_loss_tracker.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_loss_tracker, self.clf_loss_tracker]
    

def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
        return d


def read_cifar_data(directory):
    data = []
    labels = []
    for filename in directory.iterdir():
        if filename.stem.startswith('data_batch'):
            data_batch = unpickle(filename)
            data.append(data_batch[b'data'])
            labels.append(data_batch[b'labels'])
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    return data, labels
    

def read_data(dataset_name):
    if dataset_name == 'cifar10': 
        data = []
        labels = []   
        for filename in Path('../data/cifar-10-batches-py').iterdir():
            if filename.stem.startswith('data_batch'):
                data_batch = unpickle(filename)
                data.append(data_batch[b'data'])
                labels.append(data_batch[b'labels'])
        x = np.concatenate(data)
        y = np.concatenate(labels)
        x = x.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1) / 255.0

    elif dataset_name == 'fmnist':
        labels_path = os.path.join('../data/fmnist',
                                   'train-labels-idx1-ubyte.gz')
        images_path = os.path.join('../data/fmnist',
                                   'train-images-idx3-ubyte.gz')

        with gzip.open(labels_path, 'rb') as lbpath:
            y = np.frombuffer(lbpath.read(), dtype=np.uint8,
                              offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            x = np.frombuffer(imgpath.read(), dtype=np.uint8,
                              offset=16).reshape(len(y), 784)

        x = x.reshape((-1, 28, 28, 1)) / 255.0

    elif dataset_name == 'mnist':
        (x, y), (_, _) = tf.keras.datasets.mnist.load_data()
        x = x.reshape((-1, 28, 28, 1)) / 255.0

    else:
        raise ValueError(f"Wrong value -> {dataset_name=}!")
    
    return x, y

    
def run_experiment(data, labels, normal_cls_id, activation_mode, gamma):
    data_anomalous = data[labels != normal_cls_id]
    data_normal = data[labels == normal_cls_id]
    
    data = np.vstack([data_normal, data_anomalous])
    labels = np.concatenate([np.zeros(data_normal.shape[0]), np.ones(data_anomalous.shape[0])])
    
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, 
                                                      stratify=labels)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5,
                                                    stratify=y_val)
    
    model = E2EConvA3(input_dim=data.shape[1], intermediate_dims=[64, 128],
                      decision_layers_intermediate_dims=[64, 128],
                      n_channels=data.shape[-1],
                      gamma=gamma, activation_mode=activation_mode)
    model.build(input_shape=(None, data.shape[1], data.shape[2], data.shape[-1]))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    
    model.fit(x_train, y_train, epochs=200, batch_size=256,
            validation_data=(x_val, y_val), 
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=False)
    
    _, preds_train = model.predict(x_train, verbose=False)
    _, preds_val = model.predict(x_val, verbose=False)
    _, preds_test = model.predict(x_test, verbose=False)
    auc_train = roc_auc_score(y_train, preds_train.ravel())
    auc_val = roc_auc_score(y_val, preds_val.ravel())
    auc_test = roc_auc_score(y_test, preds_test.ravel())

    return {'auc_train': auc_train, 'auc_val': auc_val, 'auc_test': auc_test}


def main(args):
    x, y = read_data(args.dataset)

    aucs = []
    for _ in range(args.run_n_times):
        aucs.append(run_experiment(x, y, args.normal_cls, 'pool_to_minsize', args.gamma))

    Path(args.output_file.parent).mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(data=aucs)
    df.to_csv(args.output_file)


if __name__ == '__main__':
    main(parse_args())
