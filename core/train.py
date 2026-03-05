#unfinished 

import torch
from torch.utils.data import DataLoader
from model import build_model, get_loss, get_optimizer
from preprocess import dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(device)
criterion = get_loss()
optimizer = get_optimizer(model)

# Assume your Dataset returns:
# spectrogram tensor shape = (3, 224, 224)
# label = 0 or 1

train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

for epoch in range(10):
    model.train()

    for specs, labels in train_loader:
        specs = specs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(specs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} complete")