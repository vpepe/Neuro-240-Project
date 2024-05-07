def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from ae_structure import AE, DATA_DIM
from cleaner import graphDictClean, acceptable_phones

HIDDEN_DIM = 15
model = AE(HIDDEN_DIM=HIDDEN_DIM)
model.load_state_dict(torch.load("phonology_autoencoder_15_sparse.pt"))
model.eval()

testData = [torch.FloatTensor(list(graphDictClean[i].values())) for i in graphDictClean.keys()] #[np.zeros(73) for i in range(73)]

for i in range(DATA_DIM):
   encoded_unrounded = model.encode(torch.FloatTensor(testData[i]))
   testData[i] = encoded_unrounded

fig, ax = plt.subplots(3,1)
graph = np.reshape(torch.column_stack(testData).detach().numpy(), (HIDDEN_DIM, DATA_DIM))
ax[0].imshow(graph,cmap="viridis")
plt.xticks(range(len(acceptable_phones)),acceptable_phones)

ax[1].imshow(np.round(graph),cmap="viridis")
plt.xticks(range(len(acceptable_phones)),acceptable_phones)

print("-"*50)
print(acceptable_phones)
print("-"*50)

for i in range(HIDDEN_DIM):
    act_list = sorted(graph[i].tolist())
    mean, median = sum(act_list)/len(act_list),act_list[len(act_list)//2]
    gm = GaussianMixture(n_components=2, random_state=0).fit(np.reshape(np.array(act_list),(-1,1)))
    max_cluster = max(gm.means_)
    thresh = (max_cluster-(1-max_cluster)+0.02)[0]
    for j, value in enumerate(graph[i]):
        if value >= thresh:
            graph[i][j] = 1
        else:
            graph[i][j] = 0
    feature_idx = torch.nonzero(torch.FloatTensor(graph[i]>=thresh),as_tuple=True)[0].tolist()
    feature_chars = [acceptable_phones[j] for j in feature_idx]
    print(f"Feature {i}, Length {len(feature_chars)}: {feature_chars}")

ax[2].imshow(graph,cmap="viridis")
plt.xticks(range(len(acceptable_phones)),acceptable_phones)
plt.show()