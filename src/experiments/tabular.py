from pathlib import Path
from argparse import ArgumentParser

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_file", type=Path, default="result.txt", 
                        help="Path to a result file.")
    parser.add_argument("-d", "--dataset", type=str, 
                        help="Name of a dataset. One of: ccfd | kddps | kddh | celeba | census | campaign | thyroid.")
    parser.add_argument("-m", "--model", type=str, default="e2econva3",
                        help="Name of the model. One of: e2econva3 | a3.")
    parser.add_argument("-n", "--run_n_times", type=int, default=1,
                        help="Number of algorithm trials.")
    parser.add_argument("--gamma", type=float, default=1.0, 
                        help="E2EConvA3 gamma value.")

    return parser.parse_args()


class E2ETabConvA3(tf.keras.Model):
    def __init__(self, input_dim, in_channel_dim, in_channel_num, intermediate_dims, 
                 decision_layers_intermediate_dims, gamma=1.0, activation_mode='pool_to_minsize'):
        super().__init__()
        self.input_dim = input_dim
        self.in_channel_dim = in_channel_dim
        self.in_channel_num = in_channel_num
        self.intermediate_dims = intermediate_dims
        self.decision_layers_intermediate_dims = decision_layers_intermediate_dims
        self.gamma = gamma
        self.activation_mode = activation_mode
        
        encoder_layers = [tf.keras.layers.Dense(self.in_channel_dim * self.in_channel_num, activation='elu', use_bias=False),
                          tf.keras.layers.Reshape((self.in_channel_dim, self.in_channel_num)),
                          tf.keras.layers.BatchNormalization(),
                          ]
        for dim in intermediate_dims:
            encoder_layers.append(tf.keras.layers.Conv1D(dim, 3, strides=1, padding='same', activation='relu'))
            encoder_layers.append(tf.keras.layers.MaxPooling1D())
            encoder_layers.append(tf.keras.layers.BatchNormalization())
                               
        self.encoder = tf.keras.Sequential(encoder_layers, name='encoder')
        
        decoder_layers = []
        
        for dim in intermediate_dims[::-1]:
            decoder_layers.append(tf.keras.layers.Conv1DTranspose(dim, 3, activation="relu", strides=2, padding="same"))
            decoder_layers.append(tf.keras.layers.BatchNormalization())
        
        decoder_layers.append(tf.keras.layers.Conv1DTranspose(self.in_channel_num, 3, activation='relu', padding="same"))
        decoder_layers.append(tf.keras.layers.Flatten())
        decoder_layers.append(tf.keras.layers.Dense(self.input_dim, activation=None))
            
        self.decoder = tf.keras.Sequential(decoder_layers, name='decoder')
        
        decision_layers = []
        for dim in self.decision_layers_intermediate_dims:
            decision_layers.append(tf.keras.layers.Conv1D(dim, 3, strides=1, activation="relu", padding="same"))
            decision_layers.append(tf.keras.layers.MaxPooling1D())
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
                    ret.append(tf.pad(x, 
                                      tf.constant([[0, 0], [int((last_pad_dim - current_pad_dim) / 2),
                                                            int((last_pad_dim - current_pad_dim) / 2)], [0, 0]]),
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
    

class FCBlock(tf.keras.Model):
    def __init__(self, dims, name=""):
        super().__init__()
        self.dims = dims
        self.name_ = name
        self.layers_ = []
        for dim in self.dims:
            self.layers_.append(tf.keras.layers.Dense(dim, activation='relu'))
        self.layers_ = tf.keras.Sequential(self.layers_, name=self.name_)
        
    def call(self, inputs, training=False):
        return self.layers_(inputs)


class AE(tf.keras.Model):
    def __init__(self, input_dim, intermediate_dims, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.intermediate_dims = intermediate_dims
        self.latent_dim = latent_dim
        
        self.encoder = tf.keras.Sequential([FCBlock(self.intermediate_dims),
                                            tf.keras.layers.Dense(self.latent_dim, activation=None)], name='encoder')
        self.decoder = tf.keras.Sequential([FCBlock(self.intermediate_dims[::-1]),
                                            tf.keras.layers.Dense(self.input_dim, activation=None)], name='decoder')
        
    def call(self, inputs, training=False):
        z = self.encoder(inputs)
        recon_x = self.decoder(z)
        return recon_x
    
    def activations(self, x, batch_size=256):
        ret = []
        for sub_x in np.array_split(x, x.shape[0] // batch_size):
            batch_ret = []
            for l in self.encoder.layers[0].layers[0].layers:
                sub_x = l(sub_x)
                batch_ret.append(sub_x.numpy())
            batch_ret.append(self.encoder.layers[1](sub_x).numpy())
            batch_ret = np.concatenate(batch_ret, axis=1)
            ret.append(batch_ret)
        ret = np.concatenate(ret, axis=0)
        return ret


def read_data(dataset_name):
    if dataset_name == 'ccfd':
        df = pd.read_csv('../data/creditcard.csv')
        df = df.drop_duplicates()
        x = df.iloc[:, 1:-1]
        y = df.iloc[:, -1]

    elif dataset_name in ('kddps', 'kddh'):
        with open('../data/kddcup99/kddcup.names') as f:
            column_labels = f.read()
        column_labels = column_labels.split('\n')[:-1]
        column_types = dict()
        for line in column_labels[1:]:
            col, tp = line.split(': ')
            column_types[col] = tp
        df = pd.read_csv('../data/kddcup99/kddcup.data.gz', header=None)
        df.columns = [*column_types.keys(), 'target']
        
        if dataset_name == 'kddps':
            df = df[df.target.isin(['normal.', 'portsweep.'])]
            df = df.drop_duplicates()
            df.target = (df.target == 'portsweep.').astype(int)
            df = pd.get_dummies(df, columns=[col_name for col_name, col_type in column_types.items() if col_type == 'symbolic.'],
                                drop_first=True)

        else:
            df = df[df.service == 'http']
            df = df.drop_duplicates()
            df.target = (df.target != 'normal.').astype(int)           
            df = pd.get_dummies(df, columns=[col_name for col_name, col_type in column_types.items() if col_type == 'symbolic.'],
                                drop_first=True)
            
        x = df.loc[:, df.columns != 'target']
        y = df.target.to_numpy()

    elif dataset_name == 'campaign':
        df = pd.read_csv('../data/campaign/bank-additional-full_normalised.txt')
        df = df.drop_duplicates()
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()

    elif dataset_name == 'thyroid':
        df = pd.read_csv('../data/anthyroid.txt')
        df = df.drop_duplicates()
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()

    elif dataset_name == 'celeba':
        df = pd.read_csv('../data/celeba_bald.txt')
        df = df.drop_duplicates()
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()

    elif dataset_name == 'census':
        df = pd.read_csv('../data/census-income-full-mixed-binarized.csv')
        df = df.drop_duplicates()
        x = df.iloc[:, :-1].to_numpy().astype(float)
        y = df.iloc[:, -1].to_numpy()

    else:
        raise ValueError(f"Wrong value -> {dataset_name=}!")

    return x, y

    
def run_experiment(x, y, m='e2econva3', gamma=1.0):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, 
                                                      stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5,
                                                    stratify=y_val)
    
    scaler = QuantileTransformer(output_distribution='normal')
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    if m == 'e2econva3':
        model_args = dict(input_dim=x_train.shape[1], in_channel_dim=16, in_channel_num=16,
                    intermediate_dims=[32, 64], decision_layers_intermediate_dims=[96, 192],
                    gamma=gamma, activation_mode='pool_to_minsize')
        model = E2ETabConvA3(**model_args)
        model.build(input_shape=(None, x_train.shape[1]))
        model.compile(optimizer='adam')
        
        model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=256,
                  epochs=100, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                  verbose=False)
        
        _, preds_train = model.predict(x_train, verbose=False)
        _, preds_val = model.predict(x_val, verbose=False)
        _, preds_test = model.predict(x_test, verbose=False)

    elif m == 'a3':
        x_train_typical = x_train[y_train == 0]
        x_val_typical = x_val[y_val == 0]

        ae = AE(input_dim=x_train.shape[1], intermediate_dims=[128, 128, 256], latent_dim=64)
        ae.build(input_shape=(None, x_train.shape[1]))
        ae.compile(optimizer='adam', loss='mse')
        ae.fit(x_train_typical, x_train_typical, validation_data=(x_val_typical, x_val_typical), batch_size=256, epochs=100,
               callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
               verbose=False)
        
        activations_train = np.array(ae.activations(x_train))
        activations_val = np.array(ae.activations(x_val))
        activations_test = np.array(ae.activations(x_test))

        inp = tf.keras.Input(shape=(activations_train.shape[1]))
        o = tf.keras.layers.Dense(128, activation="relu")(inp)
        o = tf.keras.layers.Dense(128, activation="relu")(o)
        o = tf.keras.layers.Dense(1, activation='sigmoid')(o)
        model = tf.keras.Model(inputs=inp, outputs=o, name='a3')
        model.compile(optimizer='adam', loss='bce', metrics=['AUC'])
        model.fit(activations_train, y_train, validation_data=(activations_val, y_val),
                  epochs=100, batch_size=256, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                  verbose=False)
   
        preds_train = model.predict(activations_train, verbose=False)
        preds_val = model.predict(activations_val, verbose=False)
        preds_test = model.predict(activations_test, verbose=False)

    else:
        raise ValueError(f"Wrong value -> {m=}!")
    
    auc_train = roc_auc_score(y_train, preds_train.ravel())
    auc_val = roc_auc_score(y_val, preds_val.ravel())
    auc_test = roc_auc_score(y_test, preds_test.ravel())

    return {'auc_train': auc_train, 'auc_val': auc_val, 'auc_test': auc_test}


def main(args):
    x, y = read_data(args.dataset)
        
    aucs = []
    for _ in range(args.run_n_times):
        aucs.append(run_experiment(x, y, args.model, args.gamma))

    Path(args.output_file.parent).mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(data=aucs)
    df.to_csv(args.output_file)


if __name__ == '__main__':
    main(parse_args())
