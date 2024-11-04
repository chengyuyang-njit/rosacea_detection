import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.datasets import make_blobs
import os
from PIL import Image
from scipy import stats
import torch
import os
import matplotlib.pyplot as plt
import helper

# read X:
X_normal = np.zeros((512 * 512 * 3, 250))
X_rosacea = np.zeros((512 * 512 * 3, 250))

source_normal = "../Dataset/train/normal"
source_rosacea = "../Dataset/train/rosacea"
i = 0
for filename in os.listdir(source_normal):
    file_path = os.path.join(source_normal, filename)
    img = Image.open(file_path) # 255

    # print(np.shape(np.array(img).flatten().copy()))
    X_normal[:, i] = np.array(img).flatten().copy()
    i += 1
    if i == 250:
        break

i = 0
for filename in os.listdir(source_rosacea):
    file_path = os.path.join(source_rosacea, filename)
    img = Image.open(file_path)
    X_rosacea[:, i] = np.array(img).flatten().copy()
    i += 1



U_normal,S_normal,_ = np.linalg.svd(X_normal, full_matrices = False)
U_rosacea, S_rosacea, _ = np.linalg.svd(X_rosacea, full_matrices = False)


sum = np.zeros((512,512,3))
count = 0
for filename in os.listdir(source_normal):
    file_path = os.path.join(source_normal, filename)
    img = Image.open(file_path)
    sum += np.array(img)
    count += 1
mean_normal = sum / count 


sum = np.zeros((512,512,3))
count = 0
for filename in os.listdir(source_rosacea):
    file_path = os.path.join(source_rosacea, filename)
    img = Image.open(file_path)
    sum += np.array(img)
    count += 1
mean_rosacea = sum / count

# U:d x N
# U_ : d x r
# x: d x 1
def get_pca_projection(U,x,r = None):
    if r is None:
        r = np.shape(U)[1]
    U_ = U[:,:r]
    y = np.dot(np.transpose(U_), x)
    return y

normal_mean_projection = get_pca_projection(U_normal, mean_normal.flatten())
rosacea_mean_projection = get_pca_projection(U_rosacea, mean_rosacea.flatten())

# Test or Validation
TP = 0
FN = 0

i = 0
test_normal_source = "../Dataset/test/normal"
test_rosacea_source = "../Dataset/test/rosacea"

projected_mean_rosacea = get_pca_projection(U_rosacea, mean_rosacea.flatten(),r=180)
projected_mean_normal = get_pca_projection(U_normal, mean_normal.flatten(),r=180)
for filename in os.listdir(test_rosacea_source):
    print(str(i) + "'s case processing... TP:"+ str(TP) + " FN:" + str(FN))
    i += 1
    file_path = os.path.join(test_rosacea_source, filename)
    img = Image.open(file_path)
    img = np.array(img)
    x = img.flatten()
    d_r = helper.cal_e_distance(projected_mean_rosacea,
                          get_pca_projection(U_rosacea, x,r=180))
    d_n = helper.cal_e_distance(projected_mean_normal, 
                         get_pca_projection(U_normal, x,r=180))
    print("distance to rosacea:" + str(d_r))
    print("distance to normal:" + str(d_n))
    if d_r < d_n :
        TP += 1
    else:
        FN += 1 
        # plt.imshow(Image.open(file_path))
print(str(i) + "'s case processing... TP:"+ str(TP) + " FN:" + str(FN))

TN = 0
FP = 0



i = 0
for filename in os.listdir(test_normal_source):
    print(str(i) + "'s case processing... TN:"+ str(TN) + " FP:" + str(FP))
    i += 1
    file_path = os.path.join(test_normal_source, filename)
    img = Image.open(file_path)
    img = np.array(img)
    x = img.flatten()
    d_r = helper.cal_e_distance(projected_mean_rosacea,
                          get_pca_projection(U_rosacea, x,r=180))
    d_n = helper.cal_e_distance(projected_mean_normal, 
                         get_pca_projection(U_normal, x,r=180))

    print("distance to rosacea:" + str(d_r))
    print("distance to normal:" + str(d_n))
    if d_n < d_r:
        TN += 1
    else:
        FP += 1 
print(str(i) + "'s case processing... TN:"+ str(TN) + " FP:" + str(FP))