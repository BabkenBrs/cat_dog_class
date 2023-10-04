import torch
from train import CNNRunner, val_batch_gen

ckpt_name = "model_base.ckpt"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


best_model = None
with open(ckpt_name, "rb") as f:
    best_model = torch.load(f)

opt = torch.optim.Adam(best_model.parameters(), lr=1e-3)
opt.zero_grad()

runner = CNNRunner(best_model, opt, device, ckpt_name)


val_stats = runner.validate(val_batch_gen, best_model, phase_name="val")
# test_stats = runner.validate(test_batch_gen, best_model, phase_name='test')
# Нужно научится сохранять то, что прогнозирует модель
print(val_stats)
