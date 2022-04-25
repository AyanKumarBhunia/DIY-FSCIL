import torch
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F
from rasterize import rasterize_Sketch
import numpy as np
import random
from collections import defaultdict
import glob
import math
import torchnet as tnt

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
    """_summary_

    Returns:
        _type_: _description_
    """
    labels = train_classes_split + val_classes_split + unseen_classes
    label2inds = {}
    for idx, label in enumerate(labels):
        # if label not in label2inds:
        #    label2inds[label] = []
        # label2inds[label].append(idx)
        label2inds[label] = idx

    return label2inds


class Sketchy_Dataset(data.Dataset):
    """_summary_

    Args:
        data (_type_): _description_
    """
    def __init__(self, args, mode="Train"):
        """_summary_

        Args:
            args (_type_): _description_
            mode (str, optional): _description_. Defaults to "Train".
        """
        self.args = args
        self.mode = mode
        # self.training = copy.deepcopy(args.training)
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

        # select 40 classes
        train_set = [x for x in train_sketch if x.split("/")[0] in val_classes_split]
        test_set = [x for x in test_sketch if x.split("/")[0] in val_classes_split]
        self.val_sketch = train_set + test_set

        self.val_sketch_classes_dict = defaultdict(list)
        for x in self.val_sketch:
            cls = x.split("/")[0]
            cls = self.label2index[cls]
            self.val_sketch_classes_dict[cls].append(x)

        self.val_classes = [
            x
            for x in self.negetiveSampleDict.keys()
            if x.split("/")[0] in val_classes_split
        ]

        self.train_classes.sort()  # 64
        self.val_classes.sort()  # 40
        self.test_classes = unseen_classes
        self.test_classes.sort()  # 21

        self.Test_1_Sketch = [
            x for x in train_sketch if x.split("/")[0] in unseen_classes
        ]  # 11413 samples
        self.Test_2_Sketch = [
            x for x in test_sketch if x.split("/")[0] in unseen_classes
        ]  # 1213 samples

        self.Test_Sketch = self.Test_1_Sketch + self.Test_2_Sketch
        self.Test_Sketch_classes_dict = defaultdict(list)
        for x in self.Test_Sketch:
            cls = x.split("/")[0]
            cls = self.label2index[cls]
            self.Test_Sketch_classes_dict[cls].append(x)

        # cross modal data
        photo_classes = glob.glob(
            os.path.join(self.args.base_dir, self.args.photo_data_dir, "*")
        )
        # print(len(photo_classes), "**" * 30)
        # select base 64
        photo_train_classes = [
            x.split("/")[-1]
            for x in photo_classes
            if x.split("/")[-1] in train_classes_split
        ]
        photo_train_classes.sort()

        photo_val_classes = [
            x.split("/")[-1]
            for x in photo_classes
            if x.split("/")[-1] in val_classes_split
        ]
        photo_val_classes.sort()

        photo_test_classes = [
            x.split("/")[-1]
            for x in photo_classes
            if x.split("/")[-1] in unseen_classes
        ]
        photo_test_classes.sort()

        self.train_photo_classes_dict = defaultdict(list)
        for cls_name in photo_train_classes:
            cls_path = os.path.join(
                self.args.base_dir, self.args.photo_data_dir, cls_name, "*"
            )
            cls_samples = glob.glob(cls_path)
            cls_name = self.label2index[cls_name]
            self.train_photo_classes_dict[cls_name].extend(cls_samples)

        self.val_photo_classes_dict = defaultdict(list)
        for cls_name in photo_val_classes:
            cls_path = os.path.join(
                self.args.base_dir, self.args.photo_data_dir, cls_name, "*"
            )
            cls_samples = glob.glob(cls_path)
            cls_name = self.label2index[cls_name]
            self.val_photo_classes_dict[cls_name].extend(cls_samples)

        self.test_photo_classes_dict = defaultdict(list)
        for cls_name in photo_test_classes:
            cls_path = os.path.join(
                self.args.base_dir, self.args.photo_data_dir, cls_name, "*"
            )
            cls_samples = glob.glob(cls_path)
            cls_name = self.label2index[cls_name]
            self.test_photo_classes_dict[cls_name].extend(cls_samples)

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
            # self.sketch_data_dict = self.Train_Sketch_classes
            # self.photo_data_dict = self.train_photo_classes_dict

            self.input_transform = self.get_transform("Train")
            self.is_train = True
            self.phase = "train"

            print("Total Training sketch classes {}".format(len(self.train_classes)))
            print("Total Testing sketch classes {}".format(len(self.test_classes)))
            print("Total validation sketch classes {}".format(len(self.val_classes)))

            print("Total Training photo classes {}".format(len(photo_train_classes)))
            print("Total Testing photo classes {}".format(len(photo_test_classes)))
            print("Total validation photo classes {}".format(len(photo_val_classes)))

        elif self.mode.startswith("val"):
            self.classes = self.val_classes
            self.photo_classes = photo_val_classes
            self.class_indexes_novel = list(self.val_sketch_classes_dict.keys())
            self.class_indexes_base = list(self.photo_base_val_dict.keys())

            self.sketch_data_dict = self.val_sketch_classes_dict
            self.photo_data_dict = self.val_photo_classes_dict

            self.photo_data_dict_base = self.photo_base_val_dict

            self.input_transform = self.get_transform("Test")
            self.is_train = False
            self.phase = "val"
        else:
            self.classes = self.test_classes
            self.photo_classes = photo_test_classes
            self.class_indexes_novel = list(self.Test_Sketch_classes_dict.keys())
            self.class_indexes_base = list(self.photo_base_test_dict.keys())

            self.sketch_data_dict = self.Test_Sketch_classes_dict
            self.photo_data_dict = self.test_photo_classes_dict

            self.photo_data_dict_base = self.photo_base_test_dict

            self.input_transform = self.get_transform("Test")
            self.is_train = False
            self.phase = "test"

        self.total_class = len(self.classes)

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
        pass

    def __len__(self):
        if self.mode == "Train":
            return self.args.batchsize * self.args.total_train_batches
        elif self.mode == "Test":
            return self.args.batchsize * self.args.total_test_batches

    @staticmethod
    def get_transform(type):
        transform_list = []
        if type is "Train":
            transform_list.extend([transforms.Resize((512, 512))])
        elif type is "Test":
            transform_list.extend([transforms.Resize((512, 512))])
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


class FewShotDataloader:
    def __init__(
        self,
        dataset,
        nKnovel=5,  # number of novel categories.
        nKbase=-1,  # number of base categories.
        nExemplars=1,  # number of training examples per novel category.
        nTestNovel=15 * 5,  # number of test examples for all the novel categories.
        nTestBase=15 * 5,  # number of test examples for all the base categories.
        batch_size=1,  # number of training episodes per batch.
        num_workers=4,
        epoch_size=2000,  # number of batches per epoch.
    ):

        self.dataset = dataset
        self.phase = self.dataset.phase

        max_possible_nKnovel = len(self.dataset.classes)

        assert nKnovel >= 0 and nKnovel < max_possible_nKnovel
        self.nKnovel = nKnovel

        max_possible_nKbase = len(self.dataset.classes)
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if self.phase == "train" and nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel  # 59, 5
        else:
            max_possible_nKbase = 64

        assert nKbase >= 0 and nKbase <= max_possible_nKbase
        self.nKbase = nKbase

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase == "test") or (self.phase == "val")

    def sample_episode(self):
        nKnovel = self.nKnovel  # 5 for train/test
        nKbase = self.nKbase  # -1 for train and 64 for test
        nTestNovel = self.nTestNovel  # 5x3 for Train 5x15 for Test
        nTestBase = self.nTestBase  # 5x3 for Train 5x15 for Test
        nExemplars = self.nExemplars

        """
        support: exemplars, exemplars_labels: sketch
        query: test, test_labels: photos
        """

        if self.is_eval_mode:
            label_ids = self.dataset.class_indexes_base
            # print(label_ids, nKbase, nKnovel)
            Kbase = random.sample(label_ids, nKbase)

            label_ids = self.dataset.class_indexes_novel
            Knovel = random.sample(label_ids, nKnovel)

            nKnovel = len(Knovel)
            Tnovel, Exemplars = [], []
            nEvalExamplesPerClass = int(math.ceil(float(nTestNovel) / nKnovel))

            for Knovel_idx in range(len(Knovel)):
                imds_exemplars = random.sample(
                    self.dataset.sketch_data_dict[Knovel[Knovel_idx]], nExemplars
                )
                imds_tnovel = random.sample(
                    self.dataset.photo_data_dict[Knovel[Knovel_idx]],
                    nEvalExamplesPerClass,
                )

                Tnovel += [(img_id, nKbase + Knovel_idx) for img_id in imds_tnovel]
                Exemplars += [
                    (img_id, nKbase + Knovel_idx) for img_id in imds_exemplars
                ]
            assert len(Tnovel) == nTestNovel
            assert len(Exemplars) == len(Knovel) * nExemplars
            random.shuffle(Exemplars)

            Tbase = []
            if len(Kbase) > 0:
                KbaseIndices = np.random.choice(
                    np.arange(len(Kbase)), size=nTestBase, replace=True
                )
                KbaseIndices, NumImagesPerCategory = np.unique(
                    KbaseIndices, return_counts=True
                )

                for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                    imd_ids = random.sample(
                        self.dataset.photo_data_dict_base[Kbase[Kbase_idx]], NumImages
                    )
                    Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]

            assert len(Tbase) == nTestBase

            Test = Tbase + Tnovel
            random.shuffle(Test)
            Kall = Kbase + Knovel

            return Exemplars, Test, Kall, nKbase

        else:
            label_ids = self.dataset.class_indexes
            cats_ids = random.sample(label_ids, nKnovel + nKbase)
            assert len(cats_ids) == (nKnovel + nKbase)
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

            nKnovel = len(Knovel)
            Tnovel, Exemplars = [], []
            nEvalExamplesPerClass = int(math.ceil(float(nTestNovel) / nKnovel))

            for Knovel_idx in range(len(Knovel)):
                imds_exemplars = random.sample(
                    self.dataset.sketch_data_dict[Knovel[Knovel_idx]], nExemplars
                )
                imds_tnovel = random.sample(
                    self.dataset.photo_data_dict[Knovel[Knovel_idx]],
                    nEvalExamplesPerClass + nExemplars,
                )
                photos_exemplars = imds_tnovel[:nExemplars]
                imds_tnovel = imds_tnovel[nExemplars:]
                photos_per_class = nExemplars // 2
                sketches_per_class = nExemplars // 2
                # photos_per_class = 3
                # sketches_per_class = 2
                imds_exemplars = (
                    imds_exemplars[:sketches_per_class]
                    + photos_exemplars[photos_per_class:]
                )

                Tnovel += [(img_id, nKbase + Knovel_idx) for img_id in imds_tnovel]
                Exemplars += [
                    (img_id, nKbase + Knovel_idx) for img_id in imds_exemplars
                ]
            assert len(Tnovel) == nTestNovel
            assert len(Exemplars) == len(Knovel) * nExemplars
            random.shuffle(Exemplars)

            Tbase = []
            if len(Kbase) > 0:
                KbaseIndices = np.random.choice(
                    np.arange(len(Kbase)), size=nTestBase, replace=True
                )
                KbaseIndices, NumImagesPerCategory = np.unique(
                    KbaseIndices, return_counts=True
                )

                for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                    imd_ids = random.sample(
                        self.dataset.photo_data_dict[Kbase[Kbase_idx]], NumImages
                    )
                    Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]

            assert len(Tbase) == nTestBase

            Test = Tbase + Tnovel
            random.shuffle(Test)
            Kall = Kbase + Knovel

            return Exemplars, Test, Kall, nKbase

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            Exemplars, Test, Kall, nKbase = self.sample_episode()

            Xt = []
            for img_path, _ in Test:
                img = Image.open(img_path).convert("RGB")
                img = self.dataset.input_transform(img)
                Xt.append(img)
            Xt = torch.stack(Xt, dim=0)
            Yt = torch.LongTensor([label for _, label in Test])

            Kall = torch.LongTensor(Kall)
            Xe = []
            for sketch_path, _ in Exemplars:
                try:
                    vector_x = self.dataset.Coordinate[sketch_path]
                    sketch_img = rasterize_Sketch(vector_x)
                    sketch_img = Image.fromarray(sketch_img).convert("RGB")
                    # @TODO add seperate transforms
                    sketch_img = self.dataset.input_transform(sketch_img)
                    Xe.append(sketch_img)
                except:
                    # Photo
                    img = Image.open(sketch_path).convert("RGB")
                    img = self.dataset.input_transform(img)
                    Xe.append(img)

            Xe = torch.stack(Xe, dim=0)
            Ye = torch.LongTensor([label for _, label in Exemplars])
            return Xe, Ye, Xt, Yt, Kall, nKbase

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function
        )
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            # num_workers=(0 if self.is_eval_mode else self.num_workers),
            num_workers=self.num_workers,
            shuffle=(False if self.is_eval_mode else True),
        )

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(math.ceil(float(self.epoch_size) / self.batch_size))


def get_dataloader(args):
    dataset_train = Sketchy_Dataset(args, mode="Train")
    dloader_train = FewShotDataloader(
        dataset=dataset_train,
        nKnovel=args.train_nKnovel,
        nKbase=args.train_nKbase,
        nExemplars=args.train_nExemplars,  # num training examples per novel category
        nTestNovel=args.train_nTestNovel,  # num test examples for all the novel categories
        nTestBase=args.train_nTestBase,  # num test examples for all the base categories
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        epoch_size=args.train_epoch_size
        * args.train_batch_size,  # num of batches per epoch
    )

    dataset_test = Sketchy_Dataset(args, mode="test")
    dloader_test = FewShotDataloader(
        dataset=dataset_test,
        nKnovel=args.test_nKnovel,
        nKbase=args.test_nKbase,
        nExemplars=args.test_nExemplars,  # num testing examples per novel category
        # nTestNovel=args.test_nTestNovel * args.test_nKnovel,  # @TODO
        nTestNovel=args.test_nTestNovel,  # @TODO
        # num test examples for all the novel categories
        # nTestBase=args.test_nTestBase * args.test_nKnovel,  # @TODO
        nTestBase=args.test_nTestBase,  # @TODO
        # num test examples for all the base categories
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        epoch_size=args.test_epoch_size,  # num of batches per epoch
    )
    return dloader_train, dloader_test
