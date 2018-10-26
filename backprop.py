import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, b1,  W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z

# determine clssification rate
# num correct / num total
def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total +=1
        if Y[i] == P[i]:
            n_correct +=1
    return float(n_correct) / n_total

def derivative_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1]

    # slow method
    # ret1 = np.zeros((M, K))
    # for n in range(N):
        # for m in range(M):
            # for k in range(K):
                # ret1 += (T[n,k] - Y[n,k] * Z[n,m])
    
    ret1 = Z.T.dot(T - Y)
    return ret1

def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape

    # slow mthod
    # ret1 = np.zeros((D, M))
    # for n in range(N):
        # for k in range(K):
            # for m in range(M):
                # for d in range(D):
                    # ret1[d,m] += (T[n,k] - Y[n,k]) * W2[m,k] * Z[n,m] * (1 - Z[n,m]) * X[n,d]
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    return X.T.dot(dZ)

def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)

def cost(T, Y):
    # define the cost function
    tot = T * np.log(Y)
    return tot.sum()

def main():
    #create the data
    Nclass = 500
    D = 2 # dimensionality of input
    M = 3 # hidden layer size
    K = 3 # number of classes

    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)
    N = len(Y)

    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1
    
    # see what the data looks like
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()

    # randomly initialize the weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 10e-7
    costs = []
    for epoch in range(100000):
        output, hidden = forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print("cost:", c, "classification_rate:", r)
            costs.append(c)
        
        # gradient ascent
        W2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate * derivative_b1(T, output, W2, hidden)

    # plot the costs
    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()