import torch
from model_real import Simple_Stupid_model
from train import CNNRunner, val_batch_gen

if __name__ == "__main__":
    # model_param = "model_base.pt"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    best_model = Simple_Stupid_model()
    # best_model.load_state_dict(torch.load(model_param))
    with open("model_base.pt", "rb") as model_param:
        best_model.load_state_dict(torch.load(model_param))

    opt = torch.optim.Adam(best_model.parameters(), lr=1e-3)
    opt.zero_grad()

    runner = CNNRunner(best_model, opt, device)

    val_stats = runner.validate(val_batch_gen, best_model, phase_name="val")
    # test_stats = runner.validate(test_batch_gen, best_model, phase_name='test')
