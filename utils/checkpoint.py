import os
import torch

def save_checkpoint(cfg, state, is_best):
    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)

    output_path = os.path.join(cfg.save_path, cfg.post)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    torch.save(state, os.path.join(output_path, f'checkpoint_epoch_{state["epoch"]}.pth'))
    if is_best:
        torch.save(state, os.path.join(output_path, 'checkpoint_best.pth'))
