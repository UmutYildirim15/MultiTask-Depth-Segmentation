import argparse
import yaml
import time
import torch
from models.registry import build_model, get_task_type


def measure_inference_speed(config_path, input_shape=(1, 3, 512, 512), iterations=100, warmup=10):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config['model']['name']

    model = build_model(model_name).to(device)
    model.eval()

    dummy_input = torch.randn(input_shape, device=device)
    is_cuda = device.type == "cuda"

    if is_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

    print(f"[{model_name}] Starting...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

        if is_cuda:
            torch.cuda.synchronize()

        print(f"[{model_name}] Running inference for {iterations} iterations...")
        total_time = 0.0

        for _ in range(iterations):
            if is_cuda:
                starter.record()
            else:
                start_t = time.perf_counter()

            _ = model(dummy_input)

            if is_cuda:
                ender.record()
                torch.cuda.synchronize()
                total_time += starter.elapsed_time(ender) / 1000.0
            else:
                total_time += time.perf_counter() - start_t

    avg_latency = (total_time / iterations) * 1000
    fps = iterations / total_time

    print(f"\n{'=' * 40}")
    print(f"Hardware: {device.type.upper()}")
    print(f"Model: {model_name}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    measure_inference_speed(args.config)