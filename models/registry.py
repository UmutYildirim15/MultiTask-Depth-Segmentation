from .baselines.mobilevit import MobileViTMultiTask
from .baselines.pidnet import PIDNet
from .baselines.resnet_mt import ResNet50MultiTask
from .baselines.segformer_mt import SegFormerMultiTask
from .baselines.swin_upernet import SwinUperNetMultiTask
from .baselines.topformer import TopFormerProxy
from .sgda_net import SGDANet
from .baselines.videomt import VidEoMT_ADAS

MODEL_REGISTRY = {
    "pidnet": {"class": PIDNet, "task": "segmentation"},
    "mobilevit": {"class": MobileViTMultiTask, "task": "multitask"},
    "swin_upernet": {"class": SwinUperNetMultiTask, "task": "multitask"},
    "segformer_mt": {"class": SegFormerMultiTask, "task": "multitask"},
    "resnet_mt": {"class": ResNet50MultiTask, "task": "multitask"},
    "topformer": {"class": TopFormerProxy, "task": "multitask"},
    "sgda_net": {"class": SGDANet, "task": "multitask"},
    "videomt": {"class": VidEoMT_ADAS, "task": "video_multitask"}
}

def build_model(name: str, num_classes: int = 19, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]["class"](num_classes=num_classes, **kwargs)

def get_task_type(name: str) -> str:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]["task"]