from rdas.evaluation.VAE.model import VAE, Encoder, Decoder
from rdas.datasets.linear import get_dataset
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


BATCH_SIZE = 300 
EPOCHS = 50   
LR = 0.3*1e-3         
BETA = 1      
PATH_DATASET = "/Users/melnikov/Desktop/study/Skoltech/NLA/proj/rdas/datasets/samples/line_3d.txt" # Just example
TEST_SIZE = 0.2

NH = 10
D_LATENT = 2  # dimension of latent space 
D = -1 # dimension of data space

def train_epoch(model, train_loader, optimizer, use_cuda, loss_key='total'):
    model.train()

    stats = defaultdict(list)
    for x in train_loader:
        x = x[0]
        if use_cuda:
            x = x.cuda()
        losses = model.loss(x)
        optimizer.zero_grad()
        losses[loss_key].backward()
        optimizer.step()

        for k, v in losses.items():
            stats[k].append(v.item())

    return stats


def eval_model(model, data_loader, use_cuda):
    model.eval()
    stats = defaultdict(float)
    with torch.no_grad():
        for x in data_loader:
            x = x[0]
            if use_cuda:
                x = x.cuda()
            losses = model.loss(x)
            for k, v in losses.items():
                stats[k] += v.item() * x.shape[0]

        for k in stats.keys():
            stats[k] /= len(data_loader.dataset)
    return stats

def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    use_tqdm=False,
    use_cuda=False,
    loss_key='total_loss'
):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)
    if use_cuda:
        model = model.cuda()

    for epoch in forrange:
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, use_cuda, loss_key)
        test_loss = eval_model(model, test_loader, use_cuda)

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
        print(f"{test_losses['elbo_loss']=}")
        # print(f"{test_losses['kl_loss']=}")
        # print(f"{test_losses['recon_loss']=}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': test_losses['elbo_loss'],
            }, 
            'ckpt.pth'
            )
    return dict(train_losses), dict(test_losses)

def main():
    dataset = np.loadtxt(PATH_DATASET, delimiter=',') 
    SCALE = np.max(dataset)
    dataset = dataset/(0.5*SCALE) - 1

    D = dataset.shape[1]

    test_dataset = TensorDataset(torch.Tensor(dataset[:int(TEST_SIZE*dataset.shape[0])])) 
    train_dataset = TensorDataset(torch.Tensor(dataset[int(TEST_SIZE*dataset.shape[0]):])) 
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    enc = Encoder(D, D_LATENT, NH)
    dec = Decoder(D, D_LATENT, NH)

    model = VAE(enc, dec, D_LATENT)

    train_losses, test_losses = train_model(
        model, 
        train_loader, 
        test_loader, 
        epochs=EPOCHS, 
        lr=LR, 
        loss_key='elbo_loss', 
        use_tqdm=False, 
        use_cuda=False, 
    )

if __name__ == '__main__':
    main()