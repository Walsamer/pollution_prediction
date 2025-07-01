import os
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: dict
) -> dict:
    """
    Train the model using parameters from a configuration dictionary.

    Parameters:
        model : torch.nn.Module
            The neural network to train.
        train_loader : DataLoader
            DataLoader for training data.
        val_loader : DataLoader
            DataLoader for validation data.
        device : torch.device
            Device on which to perform training (cpu, cuda, etc.).
        cfg : dict
            Configuration dictionary with the following keys:
            - epochs (int): max number of training epochs
            - optimizer (str): name of optimizer class, e.g., 'Adam'
            - learning_rate (float): initial learning rate
            - weight_decay (float): L2 regularization
            - scheduler (dict, optional): LR scheduler config with 'name' and params
            - early_stopping_patience (int): epochs to wait before early stopping
            - model_dir (str): directory to save model checkpoints and plots
            - tensorboard_log_dir (str|None): directory for TensorBoard logs
            - checkpoint (dict): checkpointing strategy

    Returns:
        dict
            Training history, including 'train_loss', 'val_loss', 'best_epoch', and 'best_val_loss'.
    """
    #preparing dirs.
    model_dir = cfg['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    tb_dir = cfg.get('tensorboard_log_dir')
    writer = SummaryWriter(log_dir=tb_dir) if tb_dir else None


    opt_cls = getattr(torch.optim, cfg['optimizer'])     # Optimizer
    optimizer = opt_cls(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay']
    )


    scheduler = None     # Scheduler
    sch_cfg = cfg.get('scheduler')
    if sch_cfg:
        sched_cls = getattr(torch.optim.lr_scheduler, sch_cfg['name'])
        params = {k: v for k, v in sch_cfg.items() if k != 'name'}
        scheduler = sched_cls(optimizer, **params)

    epochs = cfg['epochs']
    patience = cfg['early_stopping_patience']
    ckpt_cfg = cfg.get('checkpoint', {})

    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.to(device).train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = torch.nn.functional.mse_loss(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).squeeze()
                loss = torch.nn.functional.mse_loss(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        if writer:
            writer.add_scalar('Loss/val', val_loss, epoch)

        logger.info('Epoch %d: train_loss=%.4f, val_loss=%.4f', epoch, train_loss, val_loss)


        if val_loss < best_val:         # Checkpoints
            best_val = val_loss
            epochs_no_improve = 0
            if ckpt_cfg.get('save_best_only', True):
                path = os.path.join(model_dir, 'best_model.pth')
                torch.save(model.state_dict(), path)
                logger.info('Saved best model to %s', path)
        else:
            epochs_no_improve += 1

        if ckpt_cfg.get('save_every_n_epochs') and epoch % ckpt_cfg['save_every_n_epochs'] == 0:
            path = os.path.join(model_dir, f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), path)
            logger.info('Saved checkpoint to %s', path)

        
        if scheduler: #if lr scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        
        if epochs_no_improve >= patience: # Early stopping
            logger.info('Early stopping at epoch %d', epoch)
            break

    if writer:
        writer.close()

    # plot loss
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plot_path = os.path.join(model_dir, 'loss_curve.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    logger.info('Saved loss curve to %s', plot_path)

    history['best_epoch'] = epoch - epochs_no_improve
    history['best_val_loss'] = best_val
    return history
