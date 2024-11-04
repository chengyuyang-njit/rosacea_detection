import numpy as np

# arr1 and arr2 are 2 numpy arrays with the same shape
def cal_e_distance(arr1, arr2):
    return np.sum(np.square(arr1 - arr2))



# U:d x N
# U_ : d x r
# x: d x 1
# y: r x 1
def get_pca_projection(U,x,r = None):
    if r is None:
        r = np.shape(U)[1]
    U_ = U[:,:r]
    y = np.dot(np.transpose(U_), x)
    return y




# Each column of U is a data point
# r is the number of principal components to be selected

def cal_m_distance(U,S,x,X_mean, X_std, r):
    U_ = U[:,:r]
    S__2 = np.square(np.diag(1/S[:r]))
    x_ = (x - X_mean) / X_std
    y = np.dot(np.transpose(U_),x_) # r x 786432 x 786432 x 1 -> r x 1
    inter_result = np.dot(y,S__2)
    return np.dot(inter_result, y) 


