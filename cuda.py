import os
import torch

# GPU 1번을 사용하도록 지정
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 중인 디바이스:", device)
print("GPU 개수:", torch.cuda.device_count())
print("GPU 이름:", torch.cuda.get_device_name(0))
