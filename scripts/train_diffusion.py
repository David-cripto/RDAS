from rdas.methods.diffusion.model import ScoreNet
from rdas.methods.diffusion.sde import VESDE
from rdas.datasets.linear import get_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange
from rdas.methods.diffusion.loss import loss_fn
import torch
from diffusers.optimization import get_scheduler

DEVICE = "cuda"

LR = 1e-4
N_EPOCHS = 10**3
BATCH_SIZE = 128


def main():
    sde = VESDE()
    marginal_prob_std = lambda t: sde.marginal_prob(None, t)[1]

    model = ScoreNet(marginal_prob_std=marginal_prob_std)
    model.to(DEVICE)
    model.train()

    optimizer = Adam(model.parameters(), lr=LR)

    train_dataset = get_dataset()
    data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    loss_history = []

    lr_scheduler = get_scheduler(
        'constant_with_warmup',
        optimizer=optimizer,
        num_warmup_steps=1*len(data_loader),
        num_training_steps=N_EPOCHS*len(data_loader),
    )

    tqdm_epoch = trange(N_EPOCHS)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = x.to(DEVICE)    
            loss = loss_fn(sde, model, x)
            loss.backward()    
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        loss_history.append(avg_loss / num_items)
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': loss_history,
            }, 
            'ckpt.pth'
            )

if __name__ == '__main__':
    main()