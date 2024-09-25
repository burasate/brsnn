# -*- coding: utf-8 -*-
"""
Burased Neaural Network v2
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os, json

class util:
    @staticmethod
    def normalize(data):
        norm = (data - np.unique(data).min()) / (np.unique(data).max() - np.unique(data).min())
        return norm

    @staticmethod
    def denormalize(norm_data, data_min, data_max):
        denorm = (norm_data * (data_max - data_min)) + data_min
        return denorm

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_deriv(z):
        return z * (1 - z)


class brs_neuro_net:
    def __init__(self, csv_path, sample_limit=999, set_hidden=[200,100,20] , set_output=1, is_normalized_data=False):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.wb_path = self.base_path + '/weight_bias'
        os.makedirs(self.wb_path, exist_ok=True)

        print('\n--------------\n brs neuro net initialize \n--------------')
        self.data = np.array([[1, 1]])
        self.weight_dict = {}
        self.bias_dict = {}
        self.layer_dict = {}
        self.delta_dict = {}
        self.hidden = set_hidden
        self.output = set_output
        self.x_train = None
        self.y_train = None
        self.x_train_norm = None
        self.y_train_norm = None

        # Load CSV Data
        self.data = np.loadtxt(csv_path, delimiter=',')
        self.data = self.data[:sample_limit]
        self.m, self.n = self.data.shape
        np.random.shuffle(self.data)
        data_train = np.transpose(self.data)
        self.y_train = np.array([data_train[0]]).T  # Select First Column as Target
        self.x_train = data_train[1:].T  # Select Left Column as Feature
        #print('Data Train Shape {}'.format(data_train.shape))
        #print('X Train Shape {}'.format(self.x_train.shape))
        #print('Y Train Shape {}'.format(self.y_train.shape))

        # Normalized Data
        if not is_normalized_data:
            self.x_train_norm = util.normalize(self.x_train)
            self.y_train_norm = util.normalize(self.y_train)
        else:
            self.x_train_norm = self.x_train
            self.y_train_norm = self.y_train

        # Hidden Leyers
        self.hidden = set_hidden
        self.output = set_output
        self.weight_dict = {}
        self.bias_dict = {}
        prev_nr_count = self.x_train.shape[1]
        for idx, nr_count in enumerate(self.hidden):
            self.weight_dict[idx] = np.random.uniform(-0.5, 0.5, (prev_nr_count, nr_count))
            self.bias_dict[idx] = np.zeros((1, nr_count))
            #print('Init Hidden[{}] Weight {}'.format(idx, self.weight_dict[idx].shape))
            #print('Init Hidden[{}] Bias {}'.format(idx, self.bias_dict[idx].shape))
            prev_nr_count = nr_count

        # Output Leyers
        out_layer_idx = range(len(self.hidden))[-1] + 1
        self.weight_dict[out_layer_idx] = np.random.uniform(-0.5, 0.5, (prev_nr_count, self.output))
        self.bias_dict[out_layer_idx] = np.zeros((1, self.output))
        #print('Init Hidden To Output Weight {}'.format(self.weight_dict[out_layer_idx].shape))
        #print('Init Hidden To Output Bias {}'.format(self.bias_dict[out_layer_idx].shape))

        # Layers
        self.layer_dict = {}
        self.delta_dict = {}
        #print('Total Layers {}'.format(len(self.weight_dict) + 1))
        #print('Total Neurons {}   index {}'.format(len(self.weight_dict), list(self.weight_dict)))

    def forward_prop(self, x):
        for i in sorted(self.weight_dict, reverse=False):
            if i == 0:
                z = np.dot(x, self.weight_dict[i]) + self.bias_dict[i]
            else:
                z = np.dot(self.layer_dict[i - 1], self.weight_dict[i]) + self.bias_dict[i]
            a = util.sigmoid(z)
            self.layer_dict[i] = a
        return self.layer_dict[max(self.layer_dict)]

    def backward_prop(self, target, learn_rate):
        for i in sorted(self.weight_dict, reverse=True):
            if i == max(self.layer_dict):
                error = self.layer_dict[i] - target  # output
                deriv = util.sigmoid_deriv(self.layer_dict[i])
                delta = error * deriv
                self.delta_dict[i] = delta
                self.weight_dict[i] = self.weight_dict[i] - learn_rate * np.dot(delta.T, self.layer_dict[i - 1]).T
                self.bias_dict[i] = self.bias_dict[i] - learn_rate * self.delta_dict[i].sum(axis=0)
            elif i > 0:
                error = np.dot(self.delta_dict[i + 1], self.weight_dict[i + 1].T)
                deriv = util.sigmoid_deriv(self.layer_dict[i])
                delta = error * deriv
                self.delta_dict[i] = delta
                self.weight_dict[i] = self.weight_dict[i] - learn_rate * np.dot(delta.T, self.layer_dict[i - 1]).T
                self.bias_dict[i] = self.bias_dict[i] - learn_rate * self.delta_dict[i].sum(axis=0)

    def save_weight(self, name='BRS_WB', description=''):
        data = {
            'name': name,
            'description': description,
            'index': list(self.weight_dict),
            'hidden': self.hidden,
            'output': self.output,
            'x_train': self.x_train.shape,
            'y_train': self.y_train.shape,
            'x_min_max': (np.min(self.x_train), np.max(self.x_train)),
            'y_min_max': (np.min(self.y_train), np.max(self.y_train)),
            'x_denorm_min': np.min(self.x_train),
            'x_denorm_max': np.max(self.x_train),
            'y_denorm_min': np.min(self.y_train),
            'y_denorm_max': np.max(self.y_train)
        }
        for i in sorted(self.weight_dict, reverse=False):
            np.savetxt(self.wb_path + '/{}_W{}.csv'.format(name, i), self.weight_dict[i], delimiter=',', fmt='%s')
            np.savetxt(self.wb_path + '/{}_B{}.csv'.format(name, i), self.bias_dict[i], delimiter=',', fmt='%s')
        with open(self.wb_path + '/{}.json'.format(name), 'w', encoding='utf-8') as out_file:
            json.dump(data, out_file, sort_keys=False, indent=4)
            out_file.close()

    def train(self, epoch=100000, learn_rate=0.001, weight_name='BRS_WB'):
        output_dict = {}
        #learn_rate = learn_rate
        #epoch = epoch
        print_count = int(round(epoch / 15))
        print('\nTraining')
        sample = range(self.x_train_norm.shape[0])
        print('Sample Count {}'.format(len(sample)))
        for idx in range(epoch):
            out = self.forward_prop(self.x_train_norm)
            output_dict[idx] = out
            target = self.y_train_norm
            error = self.y_train_norm - out
            if idx % print_count == 0:
                mse = np.mean(np.square(error))
                print('{:.2f} %  '.format((idx / float(epoch)) * 100), 'E {:.6f}'.format(mse))
            self.backward_prop(target, learn_rate)

        # Result
        actual = util.denormalize(self.y_train_norm, np.min(self.y_train), np.max(self.y_train))
        result = util.denormalize(output_dict[max(output_dict)], np.min(self.y_train), np.max(self.y_train))
        result = np.round(result)
        accuracy = ((actual == result).sum() / actual.shape[0]) * 100
        print('\nResult')
        print('Target')
        print(actual.T)
        print('Predicted')
        print(result.T)
        print('Accuracy : {}%'.format(accuracy))

        # Save Weight
        self.save_weight(name=weight_name, description='Accuracy : {}%'.format(accuracy))

    def predict(self, x, weight_name='BRS_WB', denormalized=True, convert_round=True):
        #global weight_dict, bias_dict, layer_dict, hidden
        j = json.load(open(self.wb_path + '/{}.json'.format(weight_name)))
        self.weight_dict = {}
        self.bias_dict = {}
        self.layer_dict = {}
        self.hidden = j['hidden']
        for i in j['index']:
            self.weight_dict[i] = np.loadtxt(self.wb_path + '/{}_W{}.csv'.format(weight_name, i), delimiter=',')
            self.bias_dict[i] = np.loadtxt(self.wb_path + '/{}_B{}.csv'.format(weight_name, i), delimiter=',')
        if np.max(x) > 1:
            x = util.normalize(x)
        z = self.forward_prop(x) * j['y_min_max'][1]
        if convert_round:
            z = np.round(z)
        return z

if __name__ == '__main__':
    train_path = r"C:\Users\DEX3D_I7\Downloads\mnist_train_100.csv"
    predict_path = r"C:\Users\DEX3D_I7\Downloads\mnist_train_100.csv"
    weight_name = 'BRS_WB_weights'

    # Train and Save the Model Weights
    brsnn = brs_neuro_net(
        train_path,
        is_normalized_data=False,
        sample_limit=999,
        set_hidden=[255, 128, 50, 2],
        set_output=1
    )
    brsnn.train(epoch=50000, learn_rate=0.001, weight_name=weight_name)
    x_test = brsnn.x_train[:10]
    y_test = brsnn.y_train[:10]
    predict = brsnn.predict(x_test, weight_name='BRS_WB', convert_round=False)
    print('Y Predicted', np.round(predict, 3).flatten())
    print('Y Actual   ', y_test.T.flatten())


    # Load Weights for Prediction without Re-initialization
    brsnn = brs_neuro_net(predict_path, is_normalized_data=False)
    x_data = brsnn.x_train[:10]
    predict = brsnn.predict(x_data, weight_name=weight_name, convert_round=False)
    print('Result', np.round(predict, 3).flatten())