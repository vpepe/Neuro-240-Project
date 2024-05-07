from cleaner import graphDictClean, acceptable_phones
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from ae_structure import *

HIDDEN_DIM = 15
data = [list(graphDictClean[i].values()) for i in graphDictClean.keys()]
dataset = torch.FloatTensor(data)

loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = DATA_DIM,
                                     shuffle = True)

count = 0
seed = random.randint(-2**63,2**63)
torch.manual_seed(seed)
model = AE(HIDDEN_DIM = HIDDEN_DIM)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

epochs = 10000
losses = []

for epoch in tqdm.tqdm(range(epochs)):
  for image in loader:
  
      image = image.reshape(-1, DATA_DIM)

      reconstructed = model(image)
      loss = loss_function(reconstructed, (2*image)+1) + 0.000*torch.linalg.norm(model.encode(image), dim=1, ord=1).sum()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      losses.append(loss.item())

      if epoch % 1000 == 0:
        count = 0
        loader2 = torch.utils.data.DataLoader(dataset = dataset,
                                    batch_size = 1,
                                    shuffle = True)

        for image in loader2:
          image = image.reshape(-1, DATA_DIM)
          reconstructed = torch.round(model(image))
          if torch.equal(image,reconstructed):
            count += 1
        print(f"Epoch {epoch} accuracy: {count}/{DATA_DIM}")
  if count == DATA_DIM:
    break
print(f"Run accuracy: {count}/{DATA_DIM} with seed {seed}")

torch.save(model.state_dict(),"phonology_autoencoder_15_sparse.pt")        

testData = [torch.FloatTensor(list(graphDictClean[i].values())) for i in graphDictClean.keys()] #[np.zeros(73) for i in range(73)]

for i in range(DATA_DIM):
   testData[i][i] = 1
   encoded = torch.round(model.encode(torch.FloatTensor(testData[i])))
   testData[i] = encoded

graph = np.reshape(torch.column_stack(testData).detach().numpy(), (HIDDEN_DIM, DATA_DIM))

plt.subplot()
plt.imshow(graph,cmap="viridis")
plt.show()