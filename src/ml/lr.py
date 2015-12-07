import math
import random

class LRModel(object):
    """Logistic Regression Model
    """
    def __init__(self, d):
        """
        d: dimensionality of the model
        """
        self.d = d
        self.w = [1.0] * d
        self.b = 1.0

    def predict(self, x):
        """
        predict the prob that x belongs to postive class
        x: d dim vector
        """
        z = self.b
        for xi, wi in zip(x, self.w):
            z += xi * wi
        p = 1.0 / (1.0 + math.exp(-z))
        return p
    
    def derivative(self, x, c):
        """
        compute the derivative of the parameters at the given data
        parameter
        ---------
        x: d dim vector
        c: the class x belongs to
        return
        ------
        dw, db: delta w and delta b
        """
        p = self.predict(x)
        dw = [(c - p) * xi for xi in x]
        db = c - p 
        return dw, db
        
    def update(self, dw, db, alpha):
        """
        update the parameter
        parameter
        ---------
        dw: delta w, d dim vector
        db: delta b, float
        alpha: learning rate, float
        """
        self.w = [wi + alpha * dwi for wi, dwi in zip(self.w, dw)]
        self.b = self.b + alpha * db

    def __str__(self):
        """
        print the model
        """
        return 'w:{}, b:{}'.format(self.w, self.b)
        

def online_train(model, X, T, n):
    """
    online update algorithm
    X: data set
    T: the class of each x
    n: iteration number
    """
    for i in range(n):
        for x, t in zip(X, T):
            dw, db = model.derivative(x, t)
            model.update(dw, db, 0.0001)
            
def batch_train(model, X, T, n):
    """
    batch update algorithm
    """
    for i in range(n):
        dw = model.d * [0.0]
        db = 0.0
        for x, t in zip(X, T):
            idw, idb = model.derivative(x, t)
            dw = [dwi + idwi for dwi, idwi in zip(dw, idw)]
            db += idb
        model.update(dw, db, 0.0001)

def generate_2d_data():
    """
    generate toy 2d data
    """
    X = [[2, 2], [3, 3], [1, 1],
         [-1, -1], [0, 0], [0.4, 0.4]]
    T = [1, 1, 1, 0, 0, 0]
    return X, T

def main():
    X, T = generate_2d_data()
    model = LRModel(2)
    batch_train(model, X, T, 20000)
    print model.predict([0.5, 0.5])
    print model
    
if __name__ == '__main__':
    main()
