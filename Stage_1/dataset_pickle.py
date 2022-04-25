import torch
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import cv2
from random import randint
from PIL import Image
import random
import numpy as np
import random
from glob import glob
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import copy
from natsort import natsorted

import random
import torchvision.transforms.functional as F
from rasterize import rasterize_Sketch
import numpy as np
import random
from collections import defaultdict
import copy
import glob
import math

train_classes_split = [
    "airplane",
    "alarm_clock",
    "ant",
    "ape",
    "apple",
    "armor",
    "axe",
    "banana",
    "bear",
    "bee",
    "beetle",
    "bell",
    "bench",
    "bicycle",
    "blimp",
    "bread",
    "butterfly",
    "camel",
    "candle",
    "cannon",
    "car_(sedan)",
    "castle",
    "cat",
    "chair",
    "chicken",
    "church",
    "couch",
    "crab",
    "crocodilian",
    "cup",
    "deer",
    "dog",
    "duck",
    "elephant",
    "eyeglasses",
    "fan",
    "fish",
    "flower",
    "frog",
    "geyser",
    "guitar",
    "hamburger",
    "hammer",
    "harp",
    "hat",
    "hedgehog",
    "hermit_crab",
    "horse",
    "hot-air_balloon",
    "hotdog",
    "hourglass",
    "jack-o-lantern",
    "jellyfish",
    "kangaroo",
    "knife",
    "lion",
    "lizard",
    "lobster",
    "motorcycle",
    "mushroom",
    "owl",
    "parrot",
    "penguin",
    "piano",
]

val_classes_split = [
    "pickup_truck",
    "pig",
    "pineapple",
    "pistol",
    "pizza",
    "pretzel",
    "rabbit",
    "racket",
    "ray",
    "rifle",
    "rocket",
    "sailboat",
    "saxophone",
    "scorpion",
    "sea_turtle",
    "seal",
    "shark",
    "sheep",
    "shoe",
    "snail",
    "snake",
    "spider",
    "spoon",
    "squirrel",
    "starfish",
    "strawberry",
    "swan",
    "table",
    "tank",
    "teapot",
    "teddy_bear",
    "tiger",
    "trumpet",
    "turtle",
    "umbrella",
    "violin",
    "volcano",
    "wading_bird",
    "wine_bottle",
    "zebra",
]

unseen_classes = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]


def buildLabelIndex():
    labels = train_classes_split + val_classes_split + unseen_classes
    label2inds = {}
    for idx, label in enumerate(labels):
        label2inds[label] = idx

    return label2inds


def get_data_lists(dict_):

    data = []
    data_labels = []

    for key in dict_.keys():
        value = dict_[key]
        data.extend(value)
        data_labels.extend([key] * len(value))

    return np.array(data), data_labels


class Sketch_Dataset(data.Dataset):
    def __init__(self, args, mode="Train"):

        self.args = args
        self.mode = mode
        self.label2index = buildLabelIndex()

        coordinate_path = os.path.join(args.base_dir, args.sketch_data_dir)
        with open(coordinate_path, "rb") as fp:
            (
                train_sketch,
                test_sketch,
                self.negetiveSampleDict,
                self.Coordinate,
            ) = pickle.load(fp)

        # select 64 classes
        train_set = [x for x in train_sketch if x.split("/")[0] in train_classes_split]
        test_set = [x for x in test_sketch if x.split("/")[0] in train_classes_split]
        self.Train_Sketch = train_set + test_set

        self.Train_Sketch_classes = defaultdict(list)
        for x in self.Train_Sketch:
            cls = x.split("/")[0]
            cls = self.label2index[cls]
            self.Train_Sketch_classes[cls].append(x)

        # select 64 classes
        self.train_classes = [
            x
            for x in self.negetiveSampleDict.keys()
            if x.split("/")[0] in train_classes_split
        ]
        self.train_classes.sort()  # 64

        # cross modal data
        photo_classes = glob.glob(
            os.path.join(self.args.base_dir, self.args.photo_data_dir, "*")
        )

        # select base 64
        photo_train_classes = [
            x.split("/")[-1]
            for x in photo_classes
            if x.split("/")[-1] in train_classes_split
        ]
        photo_train_classes.sort()

        self.train_photo_classes_dict = defaultdict(list)
        for cls_name in photo_train_classes:
            cls_path = os.path.join(
                self.args.base_dir, self.args.photo_data_dir, cls_name, "*"
            )
            cls_samples = glob.glob(cls_path)
            cls_name = self.label2index[cls_name]
            self.train_photo_classes_dict[cls_name].extend(cls_samples)

        (
            self.sketch_base_train_dict,
            self.sketch_base_val_dict,
            self.sketch_base_test_dict,
        ) = self._split_base_classes(self.Train_Sketch_classes)
        (
            self.photo_base_train_dict,
            self.photo_base_val_dict,
            self.photo_base_test_dict,
        ) = self._split_base_classes(self.train_photo_classes_dict)

        if self.mode.startswith("Train"):
            self.classes = self.train_classes
            self.photo_classes = photo_train_classes
            self.class_indexes = list(self.sketch_base_train_dict.keys())
            self.sketch_data_dict = self.sketch_base_train_dict
            self.photo_data_dict = self.photo_base_train_dict

            print("sketch: ", sum([len(v) for k, v in self.sketch_data_dict.items()]))
            print("Photo: ", sum([len(v) for k, v in self.photo_data_dict.items()]))

            data_photo, data_photo_labels = [], []
            for key in tqdm(self.photo_data_dict.keys()):
                value = self.photo_data_dict[key]
                for img_path in value:
                    img = Image.open(img_path).convert("RGB")
                    img = transforms.Resize((84, 84))(img)
                    data_photo.append(np.asarray(img))
                    data_photo_labels.append(key)
                    # break

            data_sketch, data_sketch_labels = [], []
            for key in tqdm(self.sketch_data_dict.keys()):
                value = self.sketch_data_dict[key]
                for sketch_path in value:
                    vector_x = self.Coordinate[sketch_path]
                    sketch_img = rasterize_Sketch(vector_x)
                    sketch_img = Image.fromarray(sketch_img).convert("RGB")
                    sketch_img = transforms.Resize((84, 84))(sketch_img)
                    data_sketch.append(np.asarray(sketch_img))
                    data_sketch_labels.append(key)
                    # break
            """
            for key in self.sketch_data_dict.keys():
                self.photo_data_dict[key].extend(self.sketch_data_dict[key])
            print('muti modal: ', sum([len(v) for k, v in self.photo_data_dict.items()]))
            """

            data, data_labels = get_data_lists(self.photo_data_dict)
            data = np.concatenate([data_photo, data_sketch], axis=0)
            data_labels = data_photo_labels + data_sketch_labels

            self.Train_List = data
            self.train_labels = data_labels

            self.input_transform = self.get_transform("Train")
            self.is_train = True
            self.phase = "train"
            print("Train: ", data.shape)
        elif self.mode.startswith("Test"):
            self.photo_data_dict = self.photo_base_val_dict
            print(
                "Test Photo: ", sum([len(v) for k, v in self.photo_data_dict.items()])
            )

            # data, data_labels = get_data_lists(self.photo_data_dict)
            data, data_labels = [], []
            for key in self.photo_data_dict.keys():
                value = self.photo_data_dict[key]
                for img_path in value:
                    img = Image.open(img_path).convert("RGB")
                    img = transforms.Resize((84, 84))(img)
                    data.append(np.asarray(img))
                    data_labels.append(key)
                    # break
            data = np.array(data)
            self.Test_List = data
            self.test_labels = data_labels

            self.input_transform = self.get_transform("Test")
            self.is_train = False
            self.phase = "test"
            print("Test: ", data.shape)

    @staticmethod
    def _split_base_classes(data_dict):
        train_dict, val_dict, test_dict = {}, {}, {}
        for key in data_dict.keys():
            img_paths = data_dict[key]
            img_paths.sort()

            LEN = len(img_paths)
            train_dict[key] = img_paths[0 : int(LEN * 0.6)]
            val_dict[key] = img_paths[int(LEN * 0.6) : int(LEN * 0.8)]
            test_dict[key] = img_paths[int(LEN * 0.8) :]
            setlist = [train_dict[key], val_dict[key], test_dict[key]]
            setlist = list(map(set, setlist))
            assert len(set.intersection(*setlist)) == 0
            # print(len(train_dict[key]), len(val_dict[key]), len(test_dict[key]))
        return train_dict, val_dict, test_dict

    def __getitem__(self, item):

        if self.mode == "Train":
            image = Image.fromarray(self.Train_List[item])
            # image = Image.open(self.Train_List[item])
            label = self.train_labels[item]
            sample = {
                "image": self.input_transform(image),
                "label": label,
            }

            return sample
        else:
            image = Image.fromarray(self.Test_List[item])
            # image = Image.open(self.Test_List[item])
            label = self.test_labels[item]
            sample = {
                "image": self.input_transform(image),
                "label": label,
            }

            return sample

    def __len__(self):
        if self.mode == "Train":
            return len(self.Train_List)
        elif self.mode == "Test":
            return len(self.Test_List)

    @staticmethod
    def get_transform(type):
        transform_list = []
        if type is "Train":
            transform_list.extend([transforms.Resize((84, 84))])
        elif type is "Test":
            transform_list.extend([transforms.Resize((84, 84))])
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transforms.Compose(transform_list)


def collate_self(batch):
    return batch


def get_dataloader(hp):

    dataset_Train = Sketch_Dataset(hp, mode="Train")
    dataset_Test = Sketch_Dataset(hp, mode="Test")

    dataloader_Train = data.DataLoader(
        dataset_Train,
        batch_size=hp.batchsize,
        shuffle=True,
        num_workers=int(hp.nThreads),
    )

    dataloader_Test = data.DataLoader(
        dataset_Test,
        batch_size=hp.batchsize,
        shuffle=False,
        num_workers=int(hp.nThreads),
    )

    return dataloader_Train, dataloader_Test
