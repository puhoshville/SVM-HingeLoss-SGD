import sys
import numpy as np
from cline import cline
from SGD import SGD
import time

class SVM:
    def __init__(self, loss = 'HingeLoss', optimizer = 'SGD', \
                train_loss_history = True, train_model_history = False):
        #self.features_size = features_size
        # Here is defined model parameters as one array, where
        # 'b' is represented as the last element in Wb
        # self.Wb = np.zeros(features_size + 1)
        # self.Wb[0] += 1.0
        self.istrained  = False
        self.loss       = loss
        self.optimizer  = optimizer

        self.Wb = None

        self.epoch_learned  = 0

        self.train_loss     = 99999.0
        self.test_loss      = 99999.0


        if train_loss_history:
            self.train_loss_history  = []
        else:
            self.train_loss_history  = None

        self.train_model_history  = []
        if not train_model_history:
            self.train_model_history = None

    def fit(self, X_train, y_train, X_test = None, y_test = None, \
                    batch_size = 15, n_epoch = 1, learning_rate = 0.0001, \
                     print_output = False):
        t1 = time.time()

        if not self.optimizer == 'SGD':
            print 'Unknown optimizer type! SVM.fit, line', cline()
            sys.exit()

        if not X_train.ndim == 2:
            print 'X_train has bad shapes in SVM.fit, line', cline()
            sys.exit()

        if not (X_train.shape[0] == y_train.shape[0]):
            print 'Input arrays shapes mismatching in SVM.fit, line', cline()
            sys.exit()

        if not self.istrained:
            self.features_size = X_train.shape[1]
            self.Wb = np.random.randn(self.features_size + 1)
            #self.Wb = np.zeros(self.features_size + 1)
            if not (self.train_loss_history is None):
                self.train_loss_history.append([self.train_loss, self.test_loss])
            if not (self.train_model_history is None):
                self.train_model_history.append(list(self.Wb))
        if self.optimizer == 'SGD':
            if self.epoch_learned == 0:
                solver = SGD(self, X_train, y_train, \
                        loss_type = self.loss, batch_size = batch_size, \
                        learning_rate = learning_rate)

            min_loss = 999999.0
            without_updates = 0
            while self.epoch_learned < n_epoch:
                solver.make_epoch()
                self.epoch_learned += 1

                self.train_loss     = solver.calc_loss(X_train, y_train)

                if not (X_test is None):
                    self.test_loss      = solver.calc_loss(X_test, y_test)

                if print_output:
                    print 'epoch %d, train_loss %1.3lf, test_loss %1.3lf'%\
                            (self.epoch_learned, self.train_loss, self.test_loss)
                self.istrained = True
                if self.test_loss < min_loss:
                    min_loss = self.test_loss
                    without_updates = 0
                else:
                    without_updates += 1
                    if without_updates > 20:
                        print 'Loss on the test set stops decreasing'
                        break
                if not (self.train_loss_history is None):
                    self.train_loss_history.append([self.train_loss, self.test_loss])
                if not (self.train_model_history is None):
                    self.train_model_history.append(list(self.Wb))

        t2 = time.time()
        print 'epoch %d, time %1.2lfs, train_loss %1.3lf, test_loss %1.3lf'%\
                (self.epoch_learned, t2-t1, self.train_loss, self.test_loss)

    def predict(self, X, binar = True):
        # Required type(X) equal to np.ndarray with dim = 2
        if not type(X) == np.ndarray:
            print 'Wrong input type in function "SVM.predict", line:', cline()
            sys.exit()
        if not X.ndim == 2:
            print 'Wrong input ndarray size in function "SVM.predict", line:', cline()
            sys.exit()

        if binar:
            return np.sign(np.dot(X, self.Wb[:-1]) + self.Wb[-1])
        else:
            return np.dot(X, self.Wb[:-1]) + self.Wb[-1]

    def get_accuracy(self, X, y_true):
        if not (X.shape[0] == y_true.shape[0]):
            print 'Input arrays shapes mismatching in SVM.get_accuracy, line', cline()

        y = self.predict(X)

        return np.sum(y == y_true) / float(y.shape[0])

    def export(self):
        pass
