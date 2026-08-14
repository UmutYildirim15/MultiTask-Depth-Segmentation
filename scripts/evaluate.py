import argparse
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.synthia import get_dataloader as get_standard_loader
from datasets.synthia_video import get_video_dataloader
from models.registry import build_model, get_task_type
from utils.metrics import SegmentationMetrics, BinnedDepthMetrics

CLASS_NAMES = [
    "sky", "building", "road", "sidewalk", "fence", "vegetation",
    "pole", "car", "sign", "pedestrian", "cyclist", "misc"
]


def evaluate_model(config_path, weights_path, data_dir):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config['model']['name']
    task_type = get_task_type(model_name)
    is_video = config['dataset'].get('type', 'standard') == 'video'

    if is_video:
        val_loader = get_video_dataloader(data_dir, batch_size=config['testing']['batch_size'], is_train=False)
    else:
        val_loader = get_standard_loader(data_dir, batch_size=config['testing']['batch_size'], is_train=False)

    model = build_model(model_name, **config['model'].get('args', {})).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()

    seg_evaluator = SegmentationMetrics(num_classes=19, ignore_index=255)
    depth_evaluator = BinnedDepthMetrics() if task_type == "multitask" else None

    print(f"\n{'=' * 60}\n Evaluating: {model_name} on {device.type.upper()}\n{'=' * 60}")

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Processing Frames"):
            if is_video:
                img1, img2 = batch['img1'].to(device), batch['img2'].to(device)
                mask2, dep2 = batch['mask2'].to(device), batch['depth2'].to(device)

                _, _, queries_t1 = model(img1, prev_queries=None)
                sem_logits, depth_logits, _ = model(img2, prev_queries=queries_t1)

                msks, depths = mask2, dep2
            else:
                imgs = batch[0].to(device)
                msks, depths = batch[1].to(device), batch[2].to(device)

                outputs = model(imgs)
                seg_logits = outputs[0] if task_type == "multitask" else outputs
                depth_logits = outputs[1] if task_type == "multitask" else None

            seg_logits = F.interpolate(seg_logits, size=msks.shape[-2:], mode="bilinear", align_corners=False)
            seg_evaluator.update(torch.argmax(seg_logits, dim=1), msks)

            if depth_logits is not None:
                depth_logits = F.interpolate(depth_logits, size=depths.shape[-2:], mode="bilinear", align_corners=False)
                depth_evaluator.update(depth_logits, depths)

    seg_res = seg_evaluator.compute()
    print(f"\n[SEGMENTATION] mIoU: {seg_res['mIoU'] * 100:.2f}% | Pixel Acc: {seg_res['pixel_acc'] * 100:.2f}%")
    for i, name in enumerate(CLASS_NAMES):
        val = seg_res["per_class_iou"][i]
        print(f"  {name:<15s}: {'N/A' if val != val else f'{val * 100:.2f}%'}")

    if depth_evaluator:
        depth_global, depth_binned = depth_evaluator.compute()
        print(
            f"\n[DEPTH GLOBAL] RMSE: {depth_global['RMSE']:.4f} | AbsRel: {depth_global['AbsRel']:.4f} | d1: {depth_global['d1'] * 100:.2f}%")
        print("[DEPTH BINNED (Distance-Based)]")
        for k, v in depth_binned.items():
            print(f"  {k:<10s} -> RMSE: {v['RMSE']:.4f} | AbsRel: {v['AbsRel']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    evaluate_model(args.config, args.weights, args.data)