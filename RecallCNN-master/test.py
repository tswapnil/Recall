
import numpy as np
import datetime
import os
import shutil
import PIL
from sklearn.decomposition import PCA

print("Start")
pca = PCA()
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
X.shape = (2,6);
pca.fit(X)
print(pca.transform(X))
