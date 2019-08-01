# encoding=utf-8
"""
    Created on 21:36 2019/07/23
    @author: Chenxi Huang
    It about the network.
"""
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.nn import init as nninit
import numpy as np
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
import scipy.linalg
from torch.autograd import Function

_ARCH_REGISTRY = {}


def kernel(ker, X1, X2, gamma):
    """
    get the Kernel
    :param ker: the type pf kernel
    :param X1: a domain
    :param X2: another domain
    :param gamma: use in the RBF model
    :return: the evaluate of pairwise distances or affinity of sets of samples
    """
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


def architecture(name, sample_shape):
    """
    Decorator to register an architecture;
    Use like so:
    @architecture('my_architecture', (3, 32, 32))
    ... class MyNetwork(nn.Module):
    ...     def __init__(self, n_classes):
    ...         # Build network
    ...         pass
    """

    def decorate(fn):
        _ARCH_REGISTRY[name] = (fn, sample_shape)
        return fn

    return decorate


def get_net_and_shape_for_architecture(arch_name):
    """
    Get network building function and expected sample shape:

    For example:
    net_class, shape = get_net_and_shape_for_architecture('my_architecture')

    if shape != expected_shape:
    ...     raise Exception('Incorrect shape')
    """
    return _ARCH_REGISTRY[arch_name]


def conv2d(m, n, k, act=True):
    # use to construct the network
    layers = [nn.Conv2d(m, n, k, padding=1)]

    if act:
        layers += [nn.ELU()]

    return nn.Sequential(
        *layers
    )


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class TCA:
    """
     This is network of "Domain Adaptation via Transfer Component Analysis"
    """

    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        """
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        """
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        """
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        """
        X = np.hstack((Xs.T, Xt.T))  # np.hstack(): tiling in the horizontal direction -> [Xs^T, Xt^T]
        # print(type(X))
        # print()
        # print(type(np.linalg.norm(X, axis=0)))
        X /= np.linalg.norm(X, axis=0)  # np.linalg.norm(X, axis=0), find X's form according to the columns
        m, n = X.shape  # n = ns + nt
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))  # (ns+nt, 1)

        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')  # frobenius=sqrt(\lambda(A^TA))
        H = np.eye(n) - 1 / n * np.ones((n, n))  # H=I_{ns+nt} - 1/(n_s+n_t)11^T, centering matrix
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)

        n_eye = m if self.kernel_type == 'primal' else n
        # a: KMK^T+\lambda Im   b: KHK^T
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)  # (KLK^T+\lambda I)^{-1}KHK^T, w is the eigenvalue, and v is eigenvectors
        ind = np.argsort(w)  # sort and return the index
        A = V[:, ind[:self.dim]]  # get the first m numbers
        Z = np.dot(A.T, K)  # W^TK
        Z /= np.linalg.norm(Z, axis=0)  # np.linalg.norm(Z, axis=0) = KWW^TK = the original K
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        """
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_red: accuracy and predicted labels on the target domain
        """
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = KNeighborsClassifier(n_neighbors=1)  # use K-Neighbors Classfier
        clf.fit(Xs_new, Ys.ravel())  # ravel: make it to one dimension
        y_pred = clf.predict(Xt_new)  # predict the value of Y
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)  # compare, and compute the accuracy
        return acc, y_pred


class JDA:
    """
    This is network of "Transfer Feature Learning with Joint Distribution Adaptation", refer to WJD.
    """

    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
        """
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        """
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T  # compared to TCA, it add the iteration number

    def fit_predict(self, Xs, Ys, Xt, Yt):
        """
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        """
        list_acc = []
        # set predefined variables
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        # construct MMD matrix
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        M = 0
        Y_tar_pseudo = None

        for t in range(self.T):
            N = 0
            M0 = e * e.T * C  # construct MMD matrix
            # the difference between TCA and JDA
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:  # Repeat
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    tt = Ys == c
                    e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])  # can't write tt is True. It is different.
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)

            M = M0 + N
            M = M / np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)

            n_eye = m if self.kernel_type == 'primal' else n
            # (X \sum_{c=0}^CM_c X^T + \lambda I)A=X H X^T A \Phi
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
            print('JDA iteration [{}/{}]: Acc: {:.4f}'.format(t + 1, self.T, acc))
        return acc, Y_tar_pseudo, list_acc


class JDA_LMS:
    """
     This is network of "Transfer Feature Learning with Joint Distribution Adaptation", refer to LMS
    """

    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
        """
        Init func
        :param kernel_type: kernel values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation, in paper is mu
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        """
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T  # compared to TCA, it add the iteration number

    def fit_predict(self, Xs, Ys, Xt, Yt0):
        """
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt0: nt * 1, target label
        :return: Z, A: the new input data and the first m eigenvalues
        """
        list_acc = []
        # set predefined variables
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape  # 800, 2081
        ns, nt = len(Xs), len(Xt)  # 1123, 958
        C = len(np.unique(Ys))
        # construct MMD matrix
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T * C
        if Yt0 is not None and len(Yt0) == nt:
            for c in np.reshape(np.unique(Ys), C, 1):
                e = np.zeros((n, n))
                e[np.where(Ys == c)] = 1 / len(Ys[np.where(Ys == c)])
                yy = Yt0 == c
                ind = np.where(yy == True)
                inds = [item + ns for item in ind]
                e[tuple(inds)] = -1 / len(Yt0[np.where(Yt0 == c)])
                e[np.isinf(e)] = 0
                M = M + np.dot(e, e.T)
        M = M / np.linalg.norm(M, 'fro')

        # construct centering matrix
        H = np.eye(n) - 1 / n * np.ones((n, n))

        # Joint Distribution Adaptation: JDA
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        # (X \sum_{c=0}^CM_c X^T + \lambda I)A=X H X^T A \Phi
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        return Z, A


class BaselineM2U(nn.Module):
    """
    This is Network of MNIST to USPS
    """

    def __init__(self, n_classes):
        super(BaselineM2U, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, (5, 5))
        self.conv1_1_bn = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3))
        self.conv2_1_bn = nn.BatchNorm2d(64)

        self.conv2_2 = nn.Conv2d(64, 64, (3, 3))
        self.conv2_2_bn = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d((2, 2))
        # Default p=0.5
        self.drop1 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(1024, 256)

        self.fc4 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(x))))
        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x))))
        x = x.view(-1, 1024)
        x = self.drop1(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def get_large_classifier(in_features_size, n_classes):

    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features_size, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, n_classes)
    )

    return classifier


class ResNet50(nn.Module):
    def __init__(self, bottleneck_dim=256, n_classes=1000, pretrained=True, use_dropout=False):
        super(ResNet50, self).__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.use_dropout = use_dropout

        resnet50 = torchvision.models.resnet50(pretrained=pretrained)

        # Extracter
        self.feature_extracter = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4,
            resnet50.avgpool,
        )

        self.bottleneck = nn.Linear(resnet50.fc.in_features, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.features_output_size = bottleneck_dim

        if use_dropout:
            self.dropout= nn.Dropout(0.5)

        # Class Classifier
        self.classifier = get_large_classifier(
            in_features_size=self.features_output_size,
            n_classes=n_classes,
        )
        self.classifier.apply(init_weights)

    def forward(self, x, get_features=False, get_class_outputs=True):
        if get_features == False and get_class_outputs == False:
            return None
        features = self.feature_extracter(x)
        features = features.view(features.size(0), -1)
        features = self.bottleneck(features)

        if self.use_dropout:
            features = self.dropout(features)

        if get_features == True and get_class_outputs == False:
            return features

        class_outputs = self.classifier(features)

        if get_features:
            return features, class_outputs
        else:
            return class_outputs

    def get_parameters(self):
        parameters = [
            {'params': self.feature_extracter.parameters(), 'lr_mult': 1, 'decay_mult': 1},
            {'params': self.bottleneck.parameters(), 'lr_mult': 10, 'decay_mult': 2},
            {'params': self.classifier.parameters(), 'lr_mult': 10, 'decay_mult': 2}
        ]

        return parameters


class DANN(object):
    """
    This is Network of Domain Adversarial Neural Network for classification
    """

    def __init__(self, learning_rate=0.05, hidden_layer_size=25, lambda_adapt=1., maxiter=200,
                 epsilon_init=None, adversarial_representation=True, seed=12342, verbose=False):
        """
        option "learning_rate" is the learning rate of the neural network. In paper is \mu.
        option "hidden_layer_size" is the hidden layer size.
        option "lambda_adapt" weights the domain adaptation regularization term. In paper is \lambda.
                if 0 or None or False, then no domain adaptation regularization is performed
        option "maxiter" number of training iterations.
        option "epsilon_init" is a term used for initialization.
                if None the weight matrices are weighted by 6/(sqrt(r+c))
                (where r and c are the dimensions of the weight matrix)
        option "adversarial_representation": if False, the adversarial classifier is trained
                but has no impact on the hidden layer representation. The label predictor is
                then the same as a standard neural-network one (see experiments_moon.py figures).
        option "seed" is the seed of the random number generator.
        """
        self.hidden_layer_size = hidden_layer_size
        self.maxiter = maxiter
        self.lambda_adapt = lambda_adapt if lambda_adapt not in (None, False) else 0.
        self.epsilon_init = epsilon_init
        self.learning_rate = learning_rate
        self.adversarial_representation = adversarial_representation
        self.seed = seed
        self.verbose = verbose

    def sigmoid(self, z):
        """
        Sigmoid function.
        """
        return 1. / (1. + np.exp(-z))

    def softmax(self, z):
        """
        Softmax function.
        """
        v = np.exp(z)
        return v / np.sum(v, axis=0)

    def random_init(self, l_in, l_out):
        """
        This method is used to initialize the weight matrices of the DA neural network
        """
        if self.epsilon_init is not None:
            epsilon = self.epsilon_init
        else:
            epsilon = sqrt(6.0 / (l_in + l_out))

        return epsilon * (2 * np.random.rand(l_out, l_in) - 1.0)

    def fit(self, X, Y, X_adapt, X_valid=None, Y_valid=None, do_random_init=True):
        """
        Trains the domain adversarial neural network until it reaches a total number of
        iterations of "self.maxiter" since it was initialize.
        inputs:
              X : Source data matrix
              Y : Source labels
              X_adapt : Target data matrix
              (X_valid, Y_valid) : validation set used for early stopping.
              do_random_init : A boolean indicating whether to use random initialization or not.
        """
        # nb_examples: n
        nb_examples, nb_features = np.shape(X)
        nb_labels = len(set(Y))
        nb_examples_adapt, _ = np.shape(X_adapt)

        if self.verbose:
            print('[DANN parameters]', self.__dict__)

        np.random.seed(self.seed)

        if do_random_init:
            # W, V <- random_init(D)
            # b,c,u,d <- 0
            W = self.random_init(nb_features, self.hidden_layer_size)
            V = self.random_init(self.hidden_layer_size, nb_labels)
            b = np.zeros(self.hidden_layer_size)
            c = np.zeros(nb_labels)
            U = np.zeros(self.hidden_layer_size)
            d = 0.
        else:
            W, V, b, c, U, d = self.W, self.V, self.b, self.c, self.U, self.d

        best_valid_risk = 2.0
        continue_until = 30

        for t in range(self.maxiter):
            for i in range(nb_examples):
                x_t, y_t = X[i, :], Y[i]
                # forward propagation
                # hidden_layer: G_f(x_i); output_layer: G_y(G_f(x_i))
                hidden_layer = self.sigmoid(np.dot(W, x_t) + b)
                output_layer = self.softmax(np.dot(V, hidden_layer) + c)

                # e(y_t): one-hot vector
                y_hot = np.zeros(nb_labels)
                y_hot[y_t] = 1.0

                # backward propagation
                # not \Delta_c <- (e(y_i)-G_y(G_f(x_i))) ???
                # delta_c = y_hot - output_layer
                delta_c = output_layer - y_hot
                # \Delta_V <- \Delta_c G_f(x_i)^T
                # -1 means not determined, according to the program.
                delta_V = np.dot(delta_c.reshape(-1, 1), hidden_layer.reshape(1, -1))
                # \Delta_b <- (V^T \Delta_c)\odot G_f(x_i) \odot (1 - G_f(x_i))
                delta_b = np.dot(V.T, delta_c) * hidden_layer * (1. - hidden_layer)
                # \Delta W <- \Delta_b\cdot(x_i)^T
                delta_W = np.dot(delta_b.reshape(-1, 1), x_t.reshape(1, -1))

                if self.lambda_adapt == 0.:
                    delta_U, delta_d = 0., 0.
                else:
                    # add domain adaptation regularizer from current domain
                    # G_d(G_f(x_i)) <- sigm(d + u^T G_f(x_i))
                    gho_x_t = self.sigmoid(np.dot(U.T, hidden_layer) + d)
                    # \Delta_d <- \lambda(1 - G_d(G_f(x_i)))
                    delta_d = self.lambda_adapt * (1. - gho_x_t)
                    # \Delta_u <- \Delta_d G_f(x_i)
                    delta_U = delta_d * hidden_layer

                    if self.adversarial_representation:
                        # tmp <- \Delta_d \times u \odot G_f(x_i) \odot (1 - G_f(x_i))
                        tmp = delta_d * U * hidden_layer * (1. - hidden_layer)
                        # \Delta_b <- \Delta_b+tmp
                        delta_b += tmp
                        # \Delta_W <- \Delta_W + tmp * (x_i)^T
                        delta_W += tmp.reshape(-1, 1) * x_t.reshape(1, -1)

                    # add domain adaptation regularizer from other domain
                    # j <- uniform_integer(1,..., n')
                    i_2 = np.random.randint(nb_examples_adapt)
                    x_t_2 = X_adapt[i_2, :]
                    # G_f(x_j) <- sigm(b + W x_j)
                    hidden_layer_2 = self.sigmoid(np.dot(W, x_t_2) + b)
                    # G_d(G_f(x_j)) <- sigm(d + u^T G_f(x_j))
                    gho_x_t_2 = self.sigmoid(np.dot(U.T, hidden_layer_2) + d)
                    # \Delta_d <- \Delta_d -\lambda G_d(G_f(x_j))
                    delta_d -= self.lambda_adapt * gho_x_t_2
                    # \Delta_u <- \Delta_u-\lambda G_d(G_f(x_j))G_f(x_j)
                    delta_U -= self.lambda_adapt * gho_x_t_2 * hidden_layer_2

                    if self.adversarial_representation:
                        # tmp <- -\lambda G_d(G_f(x_j)) \times u \odot G_f(x_j) \odot (1 - G_f(x_j))
                        tmp = -self.lambda_adapt * gho_x_t_2 * U * hidden_layer_2 * (1. - hidden_layer_2)
                        # \Delta_b <- \Delta_b+tmp
                        delta_b += tmp
                        # \Delta_W <- \Delta_W + tmp * (x_j)^T
                        delta_W += tmp.reshape(-1, 1) * x_t_2.reshape(1, -1)

                # Update neural network internal parameters
                # W <- W - \mu \Delta_W
                W -= delta_W * self.learning_rate
                # b <- b - \mu \Delta_b
                b -= delta_b * self.learning_rate
                # V <- V -\mu\Delta_V
                V -= delta_V * self.learning_rate
                # c <- c - \mu \Delta_c
                c -= delta_c * self.learning_rate

                # Update domain classifier
                # u <- u - \mu \Delta_u
                U += delta_U * self.learning_rate
                # d <- d - \mu \Delta_d
                d += delta_d * self.learning_rate
                # END for i in range(nb_examples)

            self.W, self.V, self.b, self.c, self.U, self.d = W, V, b, c, U, d

            # early stopping
            if X_valid is not None:
                valid_pred = self.predict(X_valid)
                valid_risk = np.mean(valid_pred != Y_valid)
                if valid_risk <= best_valid_risk:
                    if self.verbose:
                        print('[DANN best valid risk so far] %f (iter %d)' % (valid_risk, t))
                    best_valid_risk = valid_risk
                    best_weights = (W.copy(), V.copy(), b.copy(), c.copy())
                    best_t = t
                    continue_until = max(continue_until, int(1.5 * t))
                elif t > continue_until:
                    if self.verbose:
                        print('[DANN early stop] iter %d' % t)
                    break
        # END for t in range(self.maxiter)

        if X_valid is not None:
            self.W, self.V, self.b, self.c = best_weights
            self.nb_iter = best_t
            self.valid_risk = best_valid_risk
        else:
            self.nb_iter = self.maxiter
            self.valid_risk = 2.

    def forward(self, X):
        """
         Compute and return the network outputs for X, i.e., a 2D array of size len(X) by len(set(Y)).
         the ith row of the array contains output probabilities for each class for the ith example.
        """
        # G_f(x;W,b) = sigm(Wx+b)
        hidden_layer = self.sigmoid(np.dot(self.W, X.T) + self.b[:, np.newaxis])
        # G_y(G_f(x);V,c) = softmax(V G_f(x) + c)
        output_layer = self.softmax(np.dot(self.V, hidden_layer) + self.c[:, np.newaxis])
        return output_layer

    def hidden_representation(self, X):
        """
         Compute and return the network hidden layer values for X.
        """
        # G_d(G_f(x);u,z) = sigm(u^T G_f(x) + z)
        hidden_layer = self.sigmoid(np.dot(self.W, X.T) + self.b[:, np.newaxis])
        return hidden_layer.T

    def predict(self, X):
        """
         Compute and return the label predictions for X, i.e., a 1D array of size len(X).
         the ith row of the array contains the predicted class for the ith example .
        """
        output_layer = self.forward(X)
        # If the i-th row is the largest, it means that the i-th class is the most likely
        return np.argmax(output_layer, 0)

    def predict_domain(self, X):
        """
         Compute and return the domain predictions for X, i.e., a 1D array of size len(X).
         the ith row of the array contains the predicted domain (0 or 1) for the ith example.
        """
        hidden_layer = self.sigmoid(np.dot(self.W, X.T) + self.b[:, np.newaxis])
        output_layer = self.sigmoid(np.dot(self.U, hidden_layer) + self.d)
        return np.array(output_layer < .5, dtype=int)


# The following are network of "Self-ensembling for visual domain adaptation"
@architecture('mnist-bn-32-64-256', (1, 28, 28))
class MNIST_BN_32_64_256(nn.Module):
    def __init__(self, n_classes):
        super(MNIST_BN_32_64_256, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, (5, 5))
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3))
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3))
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout()
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(x))))

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x))))
        x = x.view(-1, 1024)
        x = self.drop1(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


@architecture('grey-32-64-128-gp', (1, 32, 32))
class Grey_32_64_128_gp(nn.Module):
    def __init__(self, n_classes):
        super(Grey_32_64_128_gp, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(128)
        self.conv3_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout()

        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = self.pool1(F.relu(self.conv1_2_bn(self.conv1_2(x))))

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))

        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        x = self.pool3(F.relu(self.conv3_3_bn(self.conv3_3(x))))

        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 128)
        x = self.drop1(x)

        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


@architecture('grey-32-64-128-gp-wn', (1, 32, 32))
class Grey_32_64_128_gp_wn(nn.Module):
    def __init__(self, n_classes):
        super(Grey_32_64_128_gp_wn, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv2_3 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout()

        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, n_classes)

        nninit.xavier_normal(self.conv1_1.weight)
        nninit.xavier_normal(self.conv1_2.weight)
        nninit.xavier_normal(self.conv2_1.weight)
        nninit.xavier_normal(self.conv2_2.weight)
        nninit.xavier_normal(self.conv2_3.weight)
        nninit.xavier_normal(self.conv3_1.weight)
        nninit.xavier_normal(self.conv3_2.weight)
        nninit.xavier_normal(self.conv3_3.weight)
        nninit.xavier_normal(self.fc4.weight)
        nninit.xavier_normal(self.fc5.weight)

        nn.utils.weight_norm(self.conv1_1, 'weight')
        nn.utils.weight_norm(self.conv1_2, 'weight')
        nn.utils.weight_norm(self.conv2_1, 'weight')
        nn.utils.weight_norm(self.conv2_2, 'weight')
        nn.utils.weight_norm(self.conv2_3, 'weight')
        nn.utils.weight_norm(self.conv3_1, 'weight')
        nn.utils.weight_norm(self.conv3_2, 'weight')
        nn.utils.weight_norm(self.conv3_3, 'weight')
        nn.utils.weight_norm(self.fc4, 'weight')

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(F.relu(self.conv1_2(x)))

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(F.relu(self.conv2_3(x)))

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(F.relu(self.conv3_3(x)))

        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 128)
        x = self.drop1(x)

        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


@architecture('grey-32-64-128-gp-nonorm', (1, 32, 32))
class Grey_32_64_128_gp_nonorm(nn.Module):
    def __init__(self, n_classes):
        super(Grey_32_64_128_gp_nonorm, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv2_3 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout(p=0.5)

        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, n_classes)

        nninit.xavier_normal(self.conv1_1.weight)
        nninit.xavier_normal(self.conv1_2.weight)
        nninit.xavier_normal(self.conv2_1.weight)
        nninit.xavier_normal(self.conv2_2.weight)
        nninit.xavier_normal(self.conv2_3.weight)
        nninit.xavier_normal(self.conv3_1.weight)
        nninit.xavier_normal(self.conv3_2.weight)
        nninit.xavier_normal(self.conv3_3.weight)
        nninit.xavier_normal(self.fc4.weight)
        nninit.xavier_normal(self.fc5.weight)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(F.relu(self.conv1_2(x)))

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(F.relu(self.conv2_3(x)))

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(F.relu(self.conv3_3(x)))

        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 128)
        x = self.drop1(x)

        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


@architecture('rgb-48-96-192-gp', (3, 32, 32))
class RGB_48_96_192_gp(nn.Module):
    def __init__(self, n_classes):
        super(RGB_48_96_192_gp, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 48, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(48)
        self.conv1_2 = nn.Conv2d(48, 48, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(48, 96, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(96)
        self.conv2_2 = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(96)
        self.conv2_3 = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(96)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(96, 192, (3, 3), padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(192)
        self.conv3_2 = nn.Conv2d(192, 192, (3, 3), padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(192)
        self.conv3_3 = nn.Conv2d(192, 192, (3, 3), padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(192)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout(p=0.5)

        self.fc4 = nn.Linear(192, 192)
        self.fc5 = nn.Linear(192, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = self.pool1(F.relu(self.conv1_2_bn(self.conv1_2(x))))

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))

        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        x = self.pool3(F.relu(self.conv3_3_bn(self.conv3_3(x))))

        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 192)
        x = self.drop1(x)

        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


@architecture('rgb-128-256-down-gp', (3, 32, 32))
class RGB_128_256_down_gp(nn.Module):
    def __init__(self, n_classes):
        super(RGB_128_256_down_gp, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(128)
        self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(128)
        self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_3_bn = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout()

        self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(256)
        self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout()

        self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        self.conv3_1_bn = nn.BatchNorm2d(512)
        self.nin3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
        self.nin3_2_bn = nn.BatchNorm2d(256)
        self.nin3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
        self.nin3_3_bn = nn.BatchNorm2d(128)

        self.fc4 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x))))
        x = self.drop1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))
        x = self.drop2(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.nin3_2_bn(self.nin3_2(x)))
        x = F.relu(self.nin3_3_bn(self.nin3_3(x)))

        x = F.avg_pool2d(x, 6)
        x = x.view(-1, 128)

        x = self.fc4(x)
        return x


@architecture('rgb40-48-96-192-384-gp', (3, 40, 40))
class RGB40_48_96_192_384_gp(nn.Module):
    def __init__(self, n_classes):
        super(RGB40_48_96_192_384_gp, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 48, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(48)
        self.conv1_2 = nn.Conv2d(48, 48, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(48, 96, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(96)
        self.conv2_2 = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(96)
        self.conv2_3 = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(96)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(96, 192, (3, 3), padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(192)
        self.conv3_2 = nn.Conv2d(192, 192, (3, 3), padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(192)
        self.conv3_3 = nn.Conv2d(192, 192, (3, 3), padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(192)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.conv4_1 = nn.Conv2d(192, 384, (3, 3), padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(384)
        self.conv4_2 = nn.Conv2d(384, 384, (3, 3))
        self.conv4_2_bn = nn.BatchNorm2d(384)

        self.drop1 = nn.Dropout()

        self.fc5 = nn.Linear(384, 384)
        self.fc6 = nn.Linear(384, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = self.pool1(F.relu(self.conv1_2_bn(self.conv1_2(x))))

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))

        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        x = self.pool3(F.relu(self.conv3_3_bn(self.conv3_3(x))))

        x = F.relu(self.conv4_1_bn(self.conv4_1(x)))
        x = F.relu(self.conv4_2_bn(self.conv4_2(x)))

        x = F.avg_pool2d(x, 3)
        x = x.view(-1, 384)
        x = self.drop1(x)

        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


@architecture('rgb40-96-192-384-gp', (3, 40, 40))
class RGB40_96_192_384_gp(nn.Module):
    def __init__(self, n_classes):
        super(RGB40_96_192_384_gp, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 96, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(96)
        self.conv1_2 = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(96)
        self.conv1_3 = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.conv1_3_bn = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(96, 192, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(192)
        self.conv2_2 = nn.Conv2d(192, 192, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(192)
        self.conv2_3 = nn.Conv2d(192, 192, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(192, 384, (3, 3), padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(384)
        self.conv3_2 = nn.Conv2d(384, 384, (3, 3), padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(384)
        self.conv3_3 = nn.Conv2d(384, 384, (3, 3), padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(384)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout()

        self.fc4 = nn.Linear(384, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x))))

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))

        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        x = self.pool3(F.relu(self.conv3_3_bn(self.conv3_3(x))))

        x = self.drop1(x)
        x = F.avg_pool2d(x, 5)
        x = x.view(-1, 384)

        x = self.fc4(x)
        return x


def robust_binary_crossentropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))


def bugged_cls_bal_bce(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))


def log_cls_bal(pred, tgt):
    return -torch.log(pred + 1.0e-6)


def get_cls_bal_function(name):
    if name == 'bce':
        return robust_binary_crossentropy
    elif name == 'log':
        return log_cls_bal
    elif name == 'bug':
        return bugged_cls_bal_bce


# The end of "Self-ensembling for visual domain adaptation"

# ADA Network
class SVHNmodel(nn.Module):
    """
    Model for application on SVHN data (32x32x3)
    Architecture identical to https://github.com/haeusser/learning_by_association
    """

    def __init__(self):
        super(SVHNmodel, self).__init__()

        self.features = nn.Sequential(
            nn.InstanceNorm2d(3),
            conv2d(3, 32, 3),
            conv2d(32, 32, 3),
            conv2d(32, 32, 3),
            nn.MaxPool2d(2, 2, padding=0),
            conv2d(32, 64, 3),
            conv2d(64, 64, 3),
            conv2d(64, 64, 3),
            nn.MaxPool2d(2, 2, padding=0),
            conv2d(64, 128, 3),
            conv2d(128, 128, 3),
            conv2d(128, 128, 3),
            nn.MaxPool2d(2, 2, padding=0)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 10)
        )

    def forward(self, x):
        phi = self.features(x)
        phi_mean = phi.view(-1, 128, 16).mean(dim=-1)
        phi = phi.view(-1, 128 * 4 * 4)
        y = self.classifier(phi)

        return phi_mean, y


class FrenchModel(nn.Module):
    """
    Model used in "Self-Ensembling for Visual Domain Adaptation"
    It is same with "RGB_128_256_down_gp(nn.Module):"
    by French et al.
    """

    def __init__(self):
        super(FrenchModel, self).__init__()

        def conv2d_3x3(inp, outp, pad=1):
            return nn.Sequential(
                nn.Conv2d(inp, outp, kernel_size=3, padding=pad),
                nn.BatchNorm2d(outp),
                nn.ReLU()
            )

        def conv2d_1x1(inp, outp):
            return nn.Sequential(
                nn.Conv2d(inp, outp, kernel_size=1, padding=0),
                nn.BatchNorm2d(outp),
                nn.ReLU()
            )

        def block(inp, outp):
            return nn.Sequential(
                conv2d_3x3(inp, outp),
                conv2d_3x3(outp, outp),
                conv2d_3x3(outp, outp)
            )

        self.features = nn.Sequential(
            block(3, 128),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Dropout2d(p=0.5),
            block(128, 256),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Dropout2d(p=0.5),
            conv2d_3x3(256, 512, pad=0),
            conv2d_1x1(512, 256),
            conv2d_1x1(256, 128),
            nn.AvgPool2d(6, 6, padding=0)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        phi = self.features(x)
        phi = phi.view(-1, 128)
        # print(x.size(), phi.size())
        y = self.classifier(phi)

        return phi, y


# MCD_UDA network
class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * -self.lambd


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


# In MCD_UDA network
# svhn to mnist
class s2mFeature(nn.Module):
    def __init__(self):
        super(s2mFeature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class s2mPredictor(nn.Module):
    def __init__(self, prob=0.5):
        super(s2mPredictor, self).__init__()
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x


# syn to gtrsb
class s2gFeature(nn.Module):
    def __init__(self):
        super(s2gFeature, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 144, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(144)
        self.conv3 = nn.Conv2d(144, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, padding=0)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, padding=0)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), stride=2, kernel_size=2, padding=0)
        x = x.view(x.size(0), 6400)
        return x


class s2gPredictor(nn.Module):
    def __init__(self):
        super(s2gPredictor, self).__init__()
        self.fc2 = nn.Linear(6400, 512)
        self.bn2_fc = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 43)
        self.bn_fc3 = nn.BatchNorm1d(43)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x


# usps
class uFeature(nn.Module):
    def __init__(self):
        super(uFeature, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(48)

    def forward(self, x):
        x = torch.mean(x, 1).view(x.size()[0], 1, x.size()[2], x.size()[3])
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, dilation=(1, 1))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, dilation=(1, 1))
        # print(x.size())
        x = x.view(x.size(0), 48 * 4 * 4)
        return x


class uPredictor(nn.Module):
    def __init__(self, prob=0.5):
        super(uPredictor, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.bn1_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2_fc = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        return x


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, lr_mult=10, decay_mult=2, output_num=1, sigmoid=True):
        super(AdversarialNetwork, self).__init__()
        self.in_feature = in_feature
        # self.hidden_size = hidden_size
        self.lr_mult = lr_mult
        self.decay_mult = decay_mult

        self.discriminator = nn.Sequential(
            # nn.Linear(in_feature, hidden_size),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(hidden_size, output_num),
            nn.Linear(self.in_features_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        # if sigmoid:
        #     self.discriminator.add_module(name='sigmoid', module=nn.Sigmoid())
        #
        # self.output_num = output_num
        self.discriminator.apply(init_weights)

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)

        y = self.discriminator(x)

        return y

    def get_parameters(self):
        parameters = [
            {"params": self.discriminator.parameters(), "lr_mult": self.lr_mult, 'decay_mult': self.decay_mult}
        ]
        return parameters


# MADA
class MADA(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(MADA, self).__init__()

        self.n_classes = n_classes
        self.pretrained = pretrained

        self.base_model = ResNet50(n_classes=n_classes, pretrained=pretrained)
        self.lr_mult = 10
        self.decay_mult = 2
        self.use_init = True
        self.use_dropout = True

        self.domain_classifiers = nn.ModuleList()
        for i in range(n_classes):
            self.domain_classifiers.append(
                AdversarialNetwork(
                    in_feature=self.base_model.features_output_size,
                    # hidden_size=1024,
                    lr_mult=self.lr_mult,
                    decay_mult=self.decay_mult,
                    # output_num=1,
                    # sigmoid=True
                )
            )

    def forward(self, x, alpha=1.0, test_mode=False):
        if test_mode:
            class_outputs = self.base_model(x, get_features=False, get_class_outputs=True)
            return class_outputs

        features, class_outputs = self.base_model(x, get_features=True, get_class_outputs=True)

        softmax_class_outputs = nn.Softmax(dim=1)(class_outputs).detach()

        i = -1
        domain_outputs = []
        for ad in self.domain_classifiers:
            i += 1
            weighted_features = softmax_class_outputs[:, i].view(-1, 1) * features
            if i == 0:
                domain_outputs = ad(weighted_features, alpha=alpha)
            else:
                domain_outputs = torch.cat([domain_outputs, ad(weighted_features, alpha=alpha)], dim=1)

        return domain_outputs, class_outputs

    def get_parameters(self):
        parameters = self.base_model.get_parameters()
        for ad in self.domain_classifiers:
            parameters += ad.get_parameters()

        return parameters


class Classifier(nn.Module):
    def __init__(self, n_classes, input_dimension):
        super().__init__()

        self._n_classes = n_classes
        self._clf = nn.Linear(input_dimension, n_classes)

    def forward(self, x):
        return self._clf(x)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self._convnet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self._convnet(x)


class GRL(torch.autograd.Function):
    def __init__(self, factor=-1):
        super().__init__()
        self._factor = factor

    def forward(self, x):
        return x

    def backward(self, grad):
        return self._factor * grad


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
