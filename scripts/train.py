import os
import argparse
import yaml
import operator
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from datasets.synthia import get_dataloader as get_standard_loader
from models.registry import build_model, get_task_type
from utils.losses import UncertaintyLoss
from utils.logger import setup_logger

from datasets.synthia_video import get_video_dataloader as get_video_loader

def get_optimizer_groups(model, config):
    if 'optim_groups' in config['training']:
        groups = []
        for group in config['training']['optim_groups']:
            module = operator.attrgetter(group['attr'])(model)
            groups.append({'params': module.parameters(), 'lr': float(group['lr'])})
        return groups
    return [{'params': model.parameters(), 'lr': float(config['training']['learning_rate'])}]


def train(config_path, data_dir, save_dir, resume_checkpoint=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    model_name = config['model']['name']
    task_type = get_task_type(model_name)
    is_video = config['dataset'].get('type', 'standard') == 'video'

    logger = setup_logger(model_name, save_dir)
    logger.info(f"Training {model_name} on {device.type.upper()}")

    # Choice between two dataloaders.
    if is_video:
        train_loader = get_video_loader(data_dir, batch_size=config['training']['batch_size'], is_train=True)
    else:
        train_loader = get_standard_loader(data_dir, batch_size=config['training']['batch_size'], is_train=True)

    model = build_model(model_name, **config['model'].get('args', {})).to(device)
    loss_module = UncertaintyLoss().to(device)

    optim_params = get_optimizer_groups(model, config)
    optim_params.append({'params': loss_module.parameters(), 'lr': float(config['training'].get('loss_lr', 1e-4))})

    optimizer = AdamW(optim_params, weight_decay=float(config['training']['weight_decay']))
    scaler = GradScaler('cuda')

    best_loss = float('inf')
    start_epoch = 0

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch']
        best_loss = ckpt.get('best_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        loss_module.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{config['training']['epochs']}]")

        for batch in progress_bar:
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                if is_video:
                    # VIDEOMT
                    img1, mask1, dep1 = batch['img1'].to(device), batch['mask1'].to(device), batch['depth1'].to(device)
                    img2, mask2, dep2 = batch['img2'].to(device), batch['mask2'].to(device), batch['depth2'].to(device)

                    sem1, dep_out1, queries_t1 = model(img1, prev_queries=None)
                    sem2, dep_out2, _ = model(img2, prev_queries=queries_t1.detach())

                    l_seg1 = loss_module.ce(sem1, mask1)
                    l_dep1 = loss_module.depth(dep_out1, dep1)
                    l_seg2 = loss_module.ce(sem2, mask2)
                    l_dep2 = loss_module.depth(dep_out2, dep2)

                    loss = l_seg1 + l_dep1 + l_seg2 + l_dep2
                else:
                    # STANDARD
                    imgs, msks, depths = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    outputs = model(imgs)

                    seg_logits = outputs[0] if task_type == "multitask" else outputs
                    depth_logits = outputs[1] if task_type == "multitask" else torch.zeros_like(depths)

                    seg_logits = F.interpolate(seg_logits, size=msks.shape[-2:], mode="bilinear", align_corners=False)
                    if task_type == "multitask":
                        depth_logits = F.interpolate(depth_logits, size=depths.shape[-2:], mode="bilinear",
                                                     align_corners=False)

                    loss = loss_module(seg_logits, msks, depth_logits,
                                       depths) if task_type == "multitask" else loss_module.ce(seg_logits, msks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")

        ckpt_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': best_loss,
        }

        torch.save(ckpt_data, os.path.join(save_dir, f"{model_name}_epoch_{epoch + 1}.pth"))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_data, os.path.join(save_dir, f"{model_name}_BEST.pth"))
            logger.info(f"New best model saved with loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(args.config, args.data, args.save_dir, args.resume)