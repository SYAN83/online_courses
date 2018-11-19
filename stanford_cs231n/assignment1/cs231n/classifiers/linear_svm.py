import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    n = X.shape[0]
    k = W.shape[-1]

    for i in range(n):
        scores = X[i].dot(W)
        for j in range(k):
            if j == y[i]:
                continue
            else:
                margin = scores[j] - scores[y[i]] + 1
                if margin > 0:
                    loss += margin
                    dW[:, j] += X[i]
                    dW[:, y[i]] -= X[i]

    # Add regularization to the loss
    loss = loss / n + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    dW = dW / n + 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """

    n = X.shape[0]
    k = W.shape[-1]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores = X.dot(W)
    bias = np.ones((n, k))
    bias[list(zip(*list(enumerate(y))))] = 0
    loss_matrix = scores - scores[list(zip(*list(enumerate(y))))].reshape(n, -1) + bias
    loss = np.sum(np.maximum(loss_matrix, 0))
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    grad_cond = (loss_matrix > 0).astype(np.int)
    grad_cond[list(zip(*list(enumerate(y))))] -= np.sum(grad_cond, axis=1)
    dW = X.T.dot(grad_cond)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    loss = loss / n + reg * np.sum(W * W)
    dW = dW / n + 2 * reg * W
    return loss, dW