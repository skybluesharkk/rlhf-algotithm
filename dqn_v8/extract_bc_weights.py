# extract_bc_weights.py
import torch

ckpt = torch.load("improved_bc_pretrained_student.pt",
                  weights_only=False,
                  map_location="cpu")   # GPU 없어도 OK
torch.save(ckpt["student_model"], "improved_bc_student_weights.pth")
print("✅  improved_bc_student_weights.pth  저장 완료!")
