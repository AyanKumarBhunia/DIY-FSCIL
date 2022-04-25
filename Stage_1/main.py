import torch
import time
from model import Sketch_Classification
from dataset_pickle import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Photo Classification")

    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.getcwd(),
        help="In order to access from condor",
    )
    parser.add_argument(
        "--sketch_data_dir",
        type=str,
        default="../multi_modal/final/sketchy_all.pickle",
        help="",
    )
    parser.add_argument(
        "--photo_data_dir",
        type=str,
        default="../multi_modal/final/extended_photo",
        help="",
    )

    parser.add_argument(
        "--saved_models", type=str, default="./models", help="Saved models directory"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="Sketchy", help="TUBerlin vs Sketchy"
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="Resnet",
        help="VGG / InceptionV3/ Resnet50",
    )
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--nThreads", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--eval_freq_iter", type=int, default=200)
    parser.add_argument("--print_freq_iter", type=int, default=10)
    parser.add_argument("--splitTrain", type=float, default=0.7)
    parser.add_argument(
        "--training", type=str, default="photo", help="sketch / photo / edge"
    )

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    # dataloader_Train_sketch, dataloader_Test_sketch = get_dataloader_sketch(hp)
    # dataloader_sketch_iterator = iter(dataloader_Train_sketch)

    model = Sketch_Classification(hp)
    model.to(device)
    step = 0
    best_accuracy = 0

    os.makedirs(hp.saved_models, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(hp.saved_models, "model_best_" + str(hp.training) + ".pth"),
    )

    for epoch in range(hp.max_epoch):

        for i_batch, batch in enumerate(dataloader_Train):
            loss = model.train_model(batch)
            step += 1

            if step % hp.print_freq_iter == 0:
                print(
                    "Epoch: {}, Iter: {}, Steps: {}, Loss: {}, Best Accuracy: {}".format(
                        epoch, i_batch, step, loss, best_accuracy
                    )
                )

            if step % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    accuracy = model.evaluate(dataloader_Test)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            hp.saved_models, "model_best_" + str(hp.training) + ".pth"
                        ),
                    )
