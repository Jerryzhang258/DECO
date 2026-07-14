import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _apply_tactile_dropout(tactile_imgs: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out the *entire* tactile observation for a fraction `p` of the batch.

    This is a whole-sensor dropout (not per-pixel), so the policy can't rely on tactile
    being present at inference time to solve the task -- it has to keep using vision too.
    """
    if tactile_imgs is None or p <= 0:
        return tactile_imgs
    b = tactile_imgs.shape[0]
    keep = (torch.rand(b, device=tactile_imgs.device) >= p).view(b, 1, 1, 1, 1)
    return tactile_imgs * keep


def _masked_flow_loss(pred_velocity: torch.Tensor, noise: torch.Tensor, action: torch.Tensor, mask: torch.Tensor, act_dim: int) -> torch.Tensor:
    """Flow-matching MSE loss, masked so that padded (post-episode-end) chunk steps don't
    contribute. `mask` is [B, chunk] boolean/float, True/1 for valid (non-padded) steps.
    """
    target = noise - action
    err = (pred_velocity - target).square()  # [B, chunk, act_dim]
    return (err * mask[..., None]).sum() / (mask.sum() * act_dim)


def train(net, net_without_ddp, train_loader, optimizer, criterion, warmup_scheduler, epoch, opt, act_dim, obs_state, scaler, local_rank):
    if local_rank == 0:
        print("Start training")
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{opt.epochs}', postfix=dict, mininterval=0.3)
    net.train()
    total_loss = 0
    tactile_dropout = getattr(opt, 'tactile_dropout', 0.0)

    for batch_idx, (img1, img2, tactile_imgs, obs, action, mask, lang_embed) in enumerate(train_loader):
        img1 = img1.cuda(local_rank)  # (b, 3, h, w)
        img2 = img2.cuda(local_rank)  # (b, 3, h, w)
        obs = obs.cuda(local_rank)    # (b, obs_dim)
        action = action.cuda(local_rank)  # (b, chunksize, act_dim)
        mask = mask.cuda(local_rank).float()  # (b, chunksize)
        lang_embed = lang_embed.cuda(local_rank)  # (b, lang_embed_dim)
        tactile_imgs = tactile_imgs.cuda(local_rank)  # (b, n_sensors*t_hist, 3, h, w)
        tactile_imgs = _apply_tactile_dropout(tactile_imgs, tactile_dropout)

        optimizer.zero_grad()
        if epoch <= opt.warm_up_epoch:
            warmup_scheduler.step()
        if not opt.amp:
            out, noise = net(img1, img2, obs=obs, act=action, lang_embed=lang_embed, tactile_imgs=tactile_imgs, training=True)
            loss = _masked_flow_loss(out, noise, action, mask, act_dim)
            loss.backward()
            optimizer.step()
        else:
            with autocast(device_type='cuda', enabled=True, dtype=torch.float16):
                out, noise = net(img1, img2, obs=obs, act=action, lang_embed=lang_embed, tactile_imgs=tactile_imgs, training=True)
                loss = _masked_flow_loss(out, noise, action, mask, act_dim)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item()

        if local_rank == 0:
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            pbar.set_postfix(**{'total_loss': total_loss / (batch_idx + 1), 'lr': current_lr})
            pbar.update(1)
            with open(os.path.join(opt.logs, 'result.txt'), 'a+') as f:
                f.writelines("Epoch:%d [%d|%d] loss:%f \n" % (epoch, batch_idx + 1, len(train_loader), loss.mean()))
            if getattr(opt, 'wandb', False) and WANDB_AVAILABLE:
                step = (epoch - 1) * len(train_loader) + batch_idx
                wandb.log({'train/step_loss': loss.item(), 'lr': current_lr, 'epoch': epoch}, step=step)

    dist.barrier()
    epoch_loss = total_loss / len(train_loader)
    if epoch % opt.save_period == 0 and local_rank == 0:
        print('save model to logs')
        torch.save(net_without_ddp.state_dict(),
                   os.path.join(opt.logs, 'epoch_%d_loss_%f.pth') % (epoch, total_loss))

    if local_rank == 0:
        with open(os.path.join(opt.logs, 'result.txt'), 'a+') as f:
            f.writelines('\nEpoch: %d, total loss: %f, epoch loss: %f' % (epoch, total_loss, epoch_loss))

    return epoch_loss


def val(net, test_loader, criterion, epoch, opt, act_dim, chunksize, obs_state, local_rank):
    if local_rank == 0:
        print("Start validation")
        pbar = tqdm(total=len(test_loader), desc=f'Epoch {epoch}/{opt.epochs}', postfix=dict, mininterval=0.3)
    net.eval()
    total_loss = 0
    mae = torch.zeros(chunksize, act_dim).cuda(local_rank)

    with torch.no_grad():
        for batch_idx, (img1, img2, tactile_imgs, obs, action, mask, lang_embed) in enumerate(test_loader):
            img1 = img1.cuda(local_rank)
            img2 = img2.cuda(local_rank)
            obs = obs.cuda(local_rank)
            action = action.cuda(local_rank)
            mask = mask.cuda(local_rank).float()
            lang_embed = lang_embed.cuda(local_rank)
            tactile_imgs = tactile_imgs.cuda(local_rank)

            out = net(img1, img2, obs=obs, act=action, lang_embed=lang_embed, tactile_imgs=tactile_imgs, training=False)

            mask_full = mask.unsqueeze(-1).repeat(1, 1, act_dim)
            loss = (mask_full * criterion(out, action)).sum() / mask_full.sum()
            total_loss += loss.item()

            ae = (mask_full * torch.abs(out - action)).sum(0)  # (chunksize, act_dim)
            dist.all_reduce(ae)
            mae += ae
            if local_rank == 0:
                ae_print = [round(x / opt.batch_size / chunksize, 2) for x in ae.sum(0).tolist()]
                pbar.set_postfix(**{'val_loss': total_loss / (batch_idx + 1), 'AE': ae_print})
                pbar.update(1)

    dist.barrier()
    mae = mae / len(test_loader) / opt.batch_size
    mae = torch.round(mae * 100) / 100
    epoch_loss = total_loss / len(test_loader)
    if local_rank == 0:
        print("\nVal epoch loss: %f" % epoch_loss)
        print('\nmae: ', mae)
        with open(os.path.join(opt.logs, 'result.txt'), 'a+') as f:
            f.writelines('\nVal epoch: %d, total loss: %f, epoch loss: %f \n MAE: %s \n\n' % (epoch, total_loss, epoch_loss, str(mae)))
        if getattr(opt, 'wandb', False) and WANDB_AVAILABLE:
            # mae is normalized-space L1 error, not denormalized physical units -- useful for
            # tracking relative progress per action dim, not for reading off real mm/rad error.
            per_dim_mae = mae.mean(dim=0)  # (act_dim,) averaged over chunk steps
            wandb.log({
                'val/epoch_loss': epoch_loss,
                'val/mae_mean': mae.mean().item(),
                'val/mae_per_dim': wandb.Histogram(per_dim_mae.cpu().numpy()),
                'epoch': epoch,
            })
    return epoch_loss
