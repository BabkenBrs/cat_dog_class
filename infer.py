import pandas as pd
import torch
from torch import nn

from cat_dog_class.model_real import Simple_Stupid_model
from train import CNNRunner, test_batch_gen

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    best_model = Simple_Stupid_model()
    with open("model_base.pt", "rb") as model_param:
        best_model.load_state_dict(torch.load(model_param))

    opt = torch.optim.Adam(best_model.parameters(), lr=1e-3)
    opt.zero_grad()

    runner = CNNRunner(best_model, opt, device)

    predicted_y = []
    for X, _ in test_batch_gen:
        predicted_y += (nn.Softmax(dim=1)(best_model.forward(X))[:, 1]).tolist()

    pred_y_df = pd.DataFrame({"predicted_y": predicted_y})
    pred_y_df.to_csv("scoring.csv", index=False)

    test_stats = runner.validate(test_batch_gen, best_model, phase_name="test")
