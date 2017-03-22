import mnist
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy
import scipy.io as sio

SIGMA = 0.1
PIE = np.pi

KAGGLE = True
GD = True
SGD = False
DEBUGGING = False

STANDARIZE = 'standardize'
TRANSFORM = 'log_transform'
BINARIZE = 'binarize'
NONE = 'none'

def loss(reg, beta, X, labels, preds):
    if (DEBUGGING):
        return metrics.accuracy_score(labels, preds)
    else:
        return int(reg * np.linalg.norm(beta) - labels.T.dot(np.log(1/(1+np.exp(-X.dot(beta))))) - (1-labels).T.dot(np.log(np.exp(-X.dot(beta))/(1+np.exp(-X.dot(beta))))))

def load_dataset():

    spam_data = sio.loadmat('spam.mat')
    X_train = spam_data['Xtrain']
    y_train = spam_data['ytrain']
    X_test = spam_data['Xtest']

    if (PREPROCESSING == STANDARIZE):

        col_T = [[] for i in range(X_train.shape[1])]

        for i in range(X_train.shape[1]):
            col_T[i] = X_train[:,i]
            mean = np.mean(col_T[i])
            sd = np.std(col_T[i])
            col_T[i] = (col_T[i] - mean) / sd
        X_train = np.column_stack(col_T)

        col_T = [[] for i in range(X_test.shape[1])]

        for i in range(X_test.shape[1]):
            col_T[i] = X_test[:,i]
            mean = np.mean(col_T[i])
            sd = np.std(col_T[i])
            col_T[i] = (col_T[i] - mean) / sd
        X_test = np.column_stack(col_T)

    if (PREPROCESSING == TRANSFORM):

        col_T = [[] for i in range(X_train.shape[1])]

        for i in range(X_train.shape[1]):
            col_T[i] = X_train[:,i]
            col_T[i] = np.log(col_T[i] + 0.1)
        X_train = np.column_stack(col_T)

        col_T = [[] for i in range(X_test.shape[1])]
        for i in range(X_test.shape[1]):
            col_T[i] = X_test[:,i]
            col_T[i] = np.log(col_T[i] + 0.1)
        X_test = np.column_stack(col_T)

    if (PREPROCESSING == BINARIZE):

        for i in range(X_train.shape[0]):
            for j in range(X_train.shape[1]):
                if (X_train[i][j] > 0):
                    X_train[i][j] = 1
                else:
                    X_train[i][j] = 0

        for i in range(X_test.shape[0]):
            for j in range(X_test.shape[1]):
                if (X_test[i][j] > 0):
                    X_test[i][j] = 1
                else:
                    X_test[i][j] = 0

    return (X_train, y_train), X_test

def train_gd(X_train, y_train, alpha, reg, num_iter):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    # Initialize W
    start_train_accuracy = []
    start_test_accuracy = []
    train_accuracy = []
    test_accuracy = []
    beta = np.zeros((X_train.shape[1], 1))
    original_alpha = alpha
    div = 1
    div_cnt = 1
    for i in range(0, num_iter):
        beta = beta - alpha * (2 * reg * beta - X_train.T.dot(y_train - 1 / (1 + np.exp(-X_train.dot(beta))))) / X_train.shape[0]
        # alpha = original_alpha / div
        # div_cnt -= 1
        # if (div_cnt == 0):
        #     div += 1
        #     div_cnt = div * 100
        # Record the accuracy of the whole process
        if (i % (num_iter // 100) == 0):
            pred_labels_train = predict(beta, X_train)
            # pred_labels_test = predict(beta, X_test)
            l = loss(reg, beta, X_train, y_train, pred_labels_train)
            train_accuracy.append(l)
            print("{0:.2f}%".format(i/num_iter*100) + "Train accuracy: {0}".format(l))

    iterations = list(range(0, num_iter, num_iter // 100))
    plt.plot(iterations, train_accuracy)
    if (DEBUGGING):
        plt.axis([0, num_iter, 0.6, 1])
    plt.savefig('gd_' + PREPROCESSING + '_.png')
    plt.clf()

    return beta

def train_sgd(X_train, y_train, alpha, reg, num_iter):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    start_train_accuracy = []
    train_accuracy = []
    beta = np.zeros((X_train.shape[1], 1))
    original_alpha = alpha
    div = 1
    div_cnt = 1
    for i in range(0, num_iter):
        sample_index = np.floor(np.random.rand()*X_train.shape[0])
        Xi = np.matrix(X_train[sample_index])
        Yi = np.matrix(y_train[sample_index])
        beta = beta - alpha * (2 * reg * beta - Xi.T.dot(Yi - 1 / (1 + np.exp(-Xi.dot(beta))))) / Xi.shape[0]

        alpha = original_alpha / div
        div_cnt -= 1
        if (div_cnt == 0):
            div += 1
            div_cnt = div * 500
        # Record the accuracy
        if (i % (num_iter // 100) == 0):
            pred_labels_train = predict(beta, X_train)
            l = loss(reg, beta, X_train, y_train, pred_labels_train)
            train_accuracy.append(l)
            print("{0:.2f}%".format(i / num_iter * 100) + "Train accuracy: {0}".format(l))

    iterations = list(range(0, num_iter, num_iter // 100))
    plt.plot(iterations, train_accuracy)
    if (DEBUGGING):
        plt.axis([0, num_iter, 0.6, 1])
    plt.savefig('sgd_' + PREPROCESSING + '_.png')
    plt.clf()
    return beta

def predict(beta, X):
    ''' From model and data points, output prediction vectors '''
    results = np.zeros((X.shape[0], 1))
    prob = [[] for i in range(X.shape[0])]
    prob = 1 / (1 + np.exp(-X.dot(beta)))
    for i in range(X.shape[0]):
        if (prob[i][0] > 0.5):
            results[i][0] = 1
        else:
            results[i][0] = 0
    return results

def phi(Gt, b, X):
    ''' Featurize the inputs using random Fourier features '''

    # Transpose X to be number of dimensions by number of datas
    X = X.T

    # Calculate phi
    phi = np.cos(np.dot(Gt, X) + b)

    # Transpose phi back to number of datas by number of dimensions
    phi = phi.T
    #phi = X.T
    return phi

def generate_Gt(X, sigma = SIGMA):
    # Construct Gt, 5000 by number of of dimensions
    Gt = np.random.randn(1000, X.shape[1]) * sigma
    return Gt

def generate_b(X):
    # Construct b
    b = np.random.rand(1000,1) * 2 * PIE
    return b



if __name__ == "__main__":

    for PREPROCESSING in [STANDARIZE, TRANSFORM, BINARIZE]:
        (X_train, y_train), X_test = load_dataset()
        Gt = generate_Gt(X_train)
        b = generate_b(X_train)
        X_train = phi(Gt, b, X_train)

        if (GD):
            beta = train_gd(X_train, y_train, alpha=0.04, reg=0.1, num_iter=20000)
            pred_labels_train = predict(beta, X_train)
            if (KAGGLE):
                X_test = phi(Gt, b, X_test)
                pred_labels_test = predict(beta, X_test)
                pred_labels_test = [pred_labels_test[i][0] for i in range(len(pred_labels_test))]
                output = [["Id"] + list(range(0,len(pred_labels_test))), ["Category"] + pred_labels_test]
                output = np.matrix(output)
                output = output.T
                output = np.asarray(output)
                with open('predictions_closed_form.csv', 'w') as mycsvfile:
                    writer = csv.writer(mycsvfile)
                    writer.writerows(output)


        if (SGD):

            beta = train_sgd(X_train, y_train, alpha=0.01, reg=0.05, num_iter=60000)
            pred_labels_train = predict(beta, X_train)
