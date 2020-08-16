# Data:
# http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html
# http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# we will only be using ratings.csv
import pdb
from scipy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import csv
from scipy.sparse.linalg import svds

with open('./ratings.csv', 'r') as f:
    data = list(csv.reader(f, delimiter=',', quotechar='"'))

len(data)

Counter([x[0] for x in data[1:]]).most_common(100)
num_users = len(Counter([x[0] for x in data[1:]]))
Counter([x[1] for x in data]).most_common(100)
num_movies = len(Counter([x[1] for x in data[1:]]))
M = np.zeros((num_users, num_movies))  # users by movies matirx

user_idx = [y[0] for y in Counter([x[0] for x in data[1:]]).most_common()]
movie_idx = [y[0] for y in Counter([x[1] for x in data[1:]]).most_common()]

for row in data[1:]:
    M[user_idx.index(row[0]), movie_idx.index(row[1])] = float(row[2])

M_norm = np.mean(M, axis=1)
M_demeaned = M - M_norm.reshape(-1, 1)

k = 500
U, S, V = svds(M_demeaned, k=k)  # how computationally expensive?
M_recovered = np.dot(np.dot(U, np.diag(S)), V) + M_norm.reshape(-1, 1)
np.linalg.norm(M-M_recovered)

# SVD 02 Choosing K
M_demeaned.shape
U, S, V = svd(M_demeaned)

# S is  Monotonically decreaseing
# So look for elbow curve
plt.plot(S)
plt.show()

# Check shapes of U,S,V
U.shape
S.shape
V.shape

# How do they fit togeather?
# So we can use the top 30 dimensions
M_recovered = np.dot(np.dot(U[:, :30], np.diag(
    S[:30])), V[:30, :]) + M_norm.reshape(-1, 1)

# Do we believe ablity to predictict a zero is useful?!!?
np.linalg.norm(M-M_recovered)

# Confirm svds and svd work the same
tmp = S[:30]
U, S, V = svds(M_demeaned, k=30)
np.isclose(tmp, np.flip(sorted(S)))

# Monotonically decreaseing

def mini_experiment(M, M_recovered, pos):
    mask = np.where(M[pos, :] != 0)
    return scipy.linalg.norm(M[pos, mask]-M_recovered[pos, mask])


pdb.set_trace()

# SVD 03 Experiment in our space
# Catch our breath
# What is in M?
# Bias for action
#  So what is really in M
Counter([x for y in M for x in y])
# real zeros?
Counter([x[2] for x in data])
plt.hist([x for y in M for x in y])
plt.show()
# ugh 0's break graph

# probablity the "right way" is:
plt.hist([x for y in M for x in y], log=True)
plt.show()

# Who cares FTM:
plt.hist(sorted([x[2] for x in data]))
plt.show()

# probably those .5'ers are strange
plt.hist(sorted([x[2] for x in data]), bins=5)
plt.show()

# hate my .5'ers but seems like we need to use them
U, S, V = svd(M_demeaned)
M_recovered = np.dot(np.dot(U[:, :30], np.diag(
    S[:30])), V[:30, :]) + M_norm.reshape(-1, 1)
metric = np.array([.5, 1, 1.5, 2, 2.5, 3.5, 4, 4.5, 5])
pos = 10
mask = np.where(M[pos, :] != 0)
real_scores = M[pos, mask][0]
pred_scores = (M_recovered[pos, mask])[0]
plt.scatter(real_scores, pred_scores)
plt.show()
tmp = []
for idx, x in enumerate(pred_scores):
    loc = np.where(min(abs(metric-x)) == abs(metric-x))[0][0]
    tmp.append((real_scores[idx], metric[loc]))
tmp
Counter(tmp)
M_recovered = np.dot(np.dot(U[:, :500], np.diag(
    S[:500])), V[:500, :]) + M_norm.reshape(-1, 1)
pred_scores = (M_recovered[pos, mask])[0]
plt.scatter(real_scores, pred_scores)
plt.show()
tmp = []
for idx, x in enumerate(pred_scores):
    loc = np.where(min(abs(metric-x)) == abs(metric-x))[0][0]
    tmp.append((real_scores[idx], metric[loc]))
Counter(tmp)
M_recovered = np.dot(np.dot(U[:, :250], np.diag(
    S[:250])), V[:250, :]) + M_norm.reshape(-1, 1)
pred_scores = (M_recovered[pos, mask])[0]
plt.scatter(real_scores, pred_scores)
plt.show()
tmp = []
for idx, x in enumerate(pred_scores):
    loc = np.where(min(abs(metric-x)) == abs(metric-x))[0][0]
    tmp.append((real_scores[idx], metric[loc]))
Counter(tmp)
M_recovered = np.dot(np.dot(U[:, :100], np.diag(
    S[:100])), V[:100, :]) + M_norm.reshape(-1, 1)
pred_scores = (M_recovered[pos, mask])[0]
plt.scatter(real_scores, pred_scores)
plt.show()
tmp = []
for idx, x in enumerate(pred_scores):
    loc = np.where(min(abs(metric-x)) == abs(metric-x))[0][0]
    tmp.append((real_scores[idx], metric[loc]))
Counter(tmp)
