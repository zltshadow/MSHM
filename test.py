import os
import torch
import numpy as np
from mshm import MultipleSequenceHybridMamba
from monai.transforms import Compose, ScaleIntensity, ToTensor
from monai.utils import set_determinism
from monai.data.image_reader import PILReader

class LoadMulImage:
    def __init__(self, reader, resize=False):
        self.reader = reader()
        self.resize = resize

    def __call__(self, filename_list):
        t1_img = torch.from_numpy(np.array(self.reader.read(filename_list[0]))).unsqueeze(0)
        t2_img = torch.from_numpy(np.array(self.reader.read(filename_list[1]))).unsqueeze(0)
        t1c_img = torch.from_numpy(np.array(self.reader.read(filename_list[2]))).unsqueeze(0)
        return torch.cat((t1_img, t2_img, t1c_img), dim=0)

if __name__ == "__main__":
    set_determinism(seed=42)
    in_channels = 3
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 图像路径
    data_dir = "data/lld"
    sub_dirs = ["low", "high"]
    label_map = {"low": 0, "high": 1}

    # 加载模型
    model = MultipleSequenceHybridMamba(
        in_channels=in_channels,
        num_classes=num_classes,
        stem_channels=64,
        spatial_dims=2,
        mamba_encoder=True,
        cross_attn=True,
        mamba_fusion=True,
    ).to(device)

    checkpoint_path = "checkpoints/lld/MSHM.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # 加载图像工具和预处理
    load_data = LoadMulImage(reader=PILReader, resize=False)
    val_transforms = Compose([
        ScaleIntensity(),
        ToTensor(track_meta=False),
    ])

    # 推理每个子目录图像
    for sub_dir in sub_dirs:
        print(f"\n==> Inference on '{sub_dir}' sample:")
        img_paths = [
            os.path.join(data_dir, sub_dir, "t1.png"),
            os.path.join(data_dir, sub_dir, "t2.png"),
            os.path.join(data_dir, sub_dir, "t1c.png"),
        ]
        img = load_data(img_paths).unsqueeze(0).to(device)
        img = val_transforms(img)

        with torch.no_grad():
            output = model(img)
            pred_class = torch.argmax(output, dim=1).item()

        result = "Low-Grade (0)" if pred_class == 0 else "High-Grade (1)"
        print(f"Prediction for '{sub_dir}': {result} | Ground Truth: {label_map[sub_dir]}")

        if pred_class == label_map[sub_dir]:
            print("✅ Correct Prediction")
        else:
            print("❌ Incorrect Prediction")
