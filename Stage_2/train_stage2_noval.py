import argparse
import os
import torch

from dataloader_crossmodal import get_dataloader
# from dataloader_all_photos import get_dataloader

# from models import Incremental_Fewshot_Model
# from models_kd import Incremental_Fewshot_Model
# from models_onlykd import Incremental_Fewshot_Model
# from models_algcnn_kd import Incremental_Fewshot_Model
# from baseline4 import Incremental_Fewshot_Model
from incremental_model import Incremental_Fewshot_Model

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# 2nd stage trainer
parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_workers", type=int, default=8, help="number of data loading workers"
)
# @TODO
parser.add_argument("--device", type=str, default="cuda", help="enables cuda")
parser.add_argument(
    "--disp_step", type=int, default=199, help="display step during training"
)
parser.add_argument(
    "--saved_models", type=str, default="./models", help="Saved models directory"
)
parser.add_argument("--base_dir", type=str, default=os.getcwd(), help="")
parser.add_argument(
    "--data_dir", type=str, default="./datasets/cross_modal_v2", help=""
)  # @TODO
parser.add_argument(
    "--sketch_data_dir",
    type=str,
    default="../FewShotIncremental_resnet50/datasets/final/sketchy_all.pickle",
    help="",
)
parser.add_argument(
    "--photo_data_dir",
    type=str,
    default="../FewShotIncremental_resnet50/datasets/final/extended_photo",
    help="",
)

######################## train params ###########################
parser.add_argument("--train_stage", type=str, default="fewshot", help="")
parser.add_argument("--epochs", type=int, default=60, help="")
parser.add_argument("--lr", type=float, default=0.1, help="")
parser.add_argument("--optim", type=str, default="sgd", help="")
parser.add_argument("--backbone", type=str, default="resnet18", help="") # resnet50
parser.add_argument("--mode", type=str, default="training", help="")
# parser.add_argument("--train_stage", type=str, default="fewshot", help="")

######################## train args ##############################
parser.add_argument(
    "--train_nKnovel", type=int, default=5, help="number of novel classes for training"
)
parser.add_argument(
    "--train_nKbase", type=int, default=-1, help="number of base classes for training"
)
parser.add_argument("--train_nExemplars", type=int, default=5, help="")
parser.add_argument("--train_nTestNovel", type=int, default=5 * 3, help="")
parser.add_argument("--train_nTestBase", type=int, default=5 * 3, help="")
parser.add_argument("--train_batch_size", type=int, default=256, help="train batch_size")
parser.add_argument("--train_epoch_size", type=int, default=600, help="")

######################## test args ##############################
parser.add_argument(
    "--test_nKnovel", type=int, default=5, help="number of novel classes for testing"
)
parser.add_argument(
    "--test_nKbase", type=int, default=64, help="number of base classes for testing"
)
parser.add_argument("--test_nExemplars", type=int, default=5, help="")
parser.add_argument("--test_nTestNovel", type=int, default=15 * 5, help="")
parser.add_argument("--test_nTestBase", type=int, default=15 * 5, help="")
parser.add_argument("--test_batch_size", type=int, default=256, help="test batch_size")
parser.add_argument("--test_epoch_size", type=int, default=2000, help="")

args = parser.parse_args()
print(vars(args))
out_dir = os.path.join(args.saved_models, "stage2")
os.makedirs(out_dir, exist_ok=True)

########################################################################
# Get Dataloader: dataloader_Train, dataloader_Test with get_dataloader
########################################################################
# dataloader_Train, dataloader_Test = get_dataloader(hp)

dataloader_train, dataloader_test = get_dataloader(args)

trainer = Incremental_Fewshot_Model(args)
"""
weights_path = os.path.join(args.base_dir, 'pretrained_models/sketch.pth')
w = torch.load(weights_path, map_location='cpu')
trainer.feat_model.load_state_dict(w, strict=False)
"""
weights_path = os.path.join(args.base_dir, "pretrained_models/crossmodal.pth")  # @TODO
w = torch.load(weights_path, map_location="cpu")
trainer.feat_model.load_state_dict(w, strict=False)

train_stats, eval_stats = {}, {}
train_list, eval_list = [], []
best_novel_acc, best_novel_stats = 0.0, {}

# save models
feat_model_path = os.path.join(out_dir, "incremental_feat_model_latest.pth")
torch.save(trainer.feat_model.state_dict(), feat_model_path)
classifier_model_path = os.path.join(out_dir, "incremental_classifier_latest.pth")
torch.save(trainer.classifier.state_dict(), classifier_model_path)
gann_model_path = os.path.join(out_dir, "incremental_gann_latest.pth")
torch.save(trainer.gann.state_dict(), gann_model_path)

for curr_epoch in range(1, args.epochs):
    trainer.adjust_learning_rates(curr_epoch)
    train_stats = trainer.run_train_epoch(dataloader_train, curr_epoch)

    # save models
    feat_model_path = os.path.join(out_dir, "incremental_feat_model_latest.pth")
    torch.save(trainer.feat_model.state_dict(), feat_model_path)
    classifier_model_path = os.path.join(out_dir, "incremental_classifier_latest.pth")
    torch.save(trainer.classifier.state_dict(), classifier_model_path)
    gann_model_path = os.path.join(out_dir, "incremental_gann_latest.pth")
    torch.save(trainer.gann.state_dict(), gann_model_path)

    if dataloader_test is not None:
        with torch.no_grad():
            eval_stats = trainer.evaluate(dataloader_test)

    print(f"epoch {curr_epoch}: train: ", train_stats)
    print(f"epoch {curr_epoch}: test: ", eval_stats)

    train_list.append(train_stats)
    eval_list.append(eval_stats)

    train_results_fname = os.path.join(out_dir, "train_results.csv")
    train_stats_df = pd.DataFrame(train_list)
    train_stats_df.to_csv(train_results_fname, index=False)

    eval_results_fname = os.path.join(out_dir, "eval_results.csv")
    eval_stats_df = pd.DataFrame(eval_list)
    eval_stats_df.to_csv(eval_results_fname, index=False)

    if best_novel_acc < eval_stats["AccuracyNovel"]:
        feat_model_path = os.path.join(out_dir, "incremental_feat_model_best.pth")
        torch.save(trainer.feat_model.state_dict(), feat_model_path)

        classifier_model_path = os.path.join(out_dir, "incremental_classifier_best.pth")
        torch.save(trainer.classifier.state_dict(), classifier_model_path)
        gann_model_path = os.path.join(out_dir, "incremental_gann_best.pth")
        torch.save(trainer.gann.state_dict(), gann_model_path)

        prev_best = eval_stats["AccuracyNovel"]
        print(f"Model Updated. Scores updated from {best_novel_acc} to {prev_best}")
        best_novel_acc = eval_stats["AccuracyNovel"]
        best_novel_stats = eval_stats

    """
    feat_model_path = os.path.join(
        out_dir, "incremental_feat_model_latest_{}.pth".format(curr_epoch)
    )
    torch.save(trainer.feat_model.state_dict(), feat_model_path)

    classifier_model_path = os.path.join(
        out_dir, "incremental_classifier_latest_{}.pth".format(curr_epoch)
    )
    torch.save(trainer.classifier.state_dict(), classifier_model_path)

    gann_model_path = os.path.join(out_dir, "incremental_gann_latest_{}.pth".format(curr_epoch))
    torch.save(trainer.gann.state_dict(), gann_model_path)
    """
print("Best Novel: ", best_novel_stats)
