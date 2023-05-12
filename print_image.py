import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

feature = np.load("instance_feature_woLoss.npy",allow_pickle=True)
feature = feature.reshape(-1, 2048)
feature = feature[::5]
target = np.array([[i] * 20 for i in range(1, 81)]).reshape(-1)
X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(feature)
plt.figure(figsize=(10, 5))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target, label="t-SNE")
plt.legend()
plt.savefig("t-SNE_woLoss_instance.png")
plt.show()
# fig, ax = plt.subplots(figsize=(8, 8))
# plt.matshow(arr)
# # im = ax.imshow(arr, cmap='plasma')
# ax.set_title("2D Matrix Visualization", fontsize=16)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# # plt.colorbar(im)
# plt.savefig("confuse_matrix_woLoss.png")
# plt.show()