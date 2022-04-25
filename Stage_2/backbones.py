import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as backbone_


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, userelu=True):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module(
            "Conv",
            nn.Conv2d(
                in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
            ),
        )
        self.layers.add_module("BatchNorm", nn.BatchNorm2d(out_planes))

        if userelu:
            self.layers.add_module("ReLU", nn.ReLU(inplace=True))

        self.layers.add_module(
            "MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        out = self.layers(x)

        return out


class ConvNet(nn.Module):
    def __init__(self, opt=None):
        super(ConvNet, self).__init__()

        if opt is None:
            opt = {
                "userelu": False,
                "in_planes": 3,
                "out_planes": [64, 64, 128, 128],
                "num_stages": 4,
            }

        self.in_planes = opt["in_planes"]
        self.out_planes = opt["out_planes"]
        self.num_stages = opt["num_stages"]
        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert type(self.out_planes) == list and len(self.out_planes) == self.num_stages

        num_planes = [
            self.in_planes,
        ] + self.out_planes
        userelu = opt["userelu"] if ("userelu" in opt) else True

        conv_blocks = []
        for i in range(self.num_stages):
            if i == (self.num_stages - 1):
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i + 1], userelu=userelu)
                )
            else:
                conv_blocks.append(ConvBlock(num_planes[i], num_planes[i + 1]))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.size(0), -1)
        return out


# resnet18 + GAP
class Resnet_Network(nn.Module):
    def __init__(self):
        super(Resnet_Network, self).__init__()
        backbone = backbone_.resnet18(pretrained=True)  # resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ["avgpool", "fc"]:
                self.features.add_module(name, module)

        self.pool_method = nn.AdaptiveMaxPool2d(1)  # as default
        # num_class = 125
        # self.classifier = nn.Linear(2048, num_class)

    def forward(self, input, bb_box=None):
        x = self.features(input)
        x = self.pool_method(x)
        x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x


class LinearDiag(nn.Module):
    def __init__(self, num_features, bias=False):
        super(LinearDiag, self).__init__()
        weight = torch.FloatTensor(num_features).fill_(
            1
        )  # initialize to the identity transform
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_features).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter("bias", None)

    def forward(self, X):
        assert X.dim() == 2 and X.size(1) == self.weight.size(0)
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out


class FeatExemplarAvgBlock(nn.Module):
    def __init__(self, nFeat):
        super(FeatExemplarAvgBlock, self).__init__()

    def forward(self, features_train, labels_train):
        # torch.Size([8, 25, 512]), torch.Size([8, 25, 5])
        # Val torch.Size([1, 25, 512]) torch.Size([1, 25, 5])
        labels_train_transposed = labels_train.transpose(1, 2)  # torch.Size([8, 5, 25])
        # Val torch.Size([1, 5, 25])
        """
        # labels_train[0, :][:, 0] == 0
        # labels_train[0, :][:, 0][labels_train[0, :][:, 0] == 1]
        for b_idx in range(labels_train.shape[0]):
            for c_idx in range(5):
                l = labels_train[b_idx, :][:, c_idx][labels_train[b_idx, :][:, c_idx]==1]
                f = features_train[b_idx, :][:, c_idx][labels_train[b_idx, :][:, c_idx]==1]
                print(b_idx, c_idx, l.shape, f.shape)
        """
        weight_novel = torch.bmm(
            labels_train_transposed, features_train
        )  # torch.Size([8, 5, 512])
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel)
        )  # torch.Size([8, 5, 1]) --> torch.Size([8, 5, 512])
        return weight_novel


class AttentionBasedBlock(nn.Module):
    def __init__(self, nFeat, nK, scale_att=10.0):
        super(AttentionBasedBlock, self).__init__()
        self.nFeat = nFeat
        self.queryLayer = nn.Linear(nFeat, nFeat)
        self.queryLayer.weight.data.copy_(
            torch.eye(nFeat, nFeat) + torch.randn(nFeat, nFeat) * 0.001
        )
        self.queryLayer.bias.data.zero_()

        self.scale_att = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_att), requires_grad=True
        )
        wkeys = torch.FloatTensor(nK, nFeat).normal_(0.0, np.sqrt(2.0 / nFeat))
        self.wkeys = nn.Parameter(wkeys, requires_grad=True)

    def forward(self, features_train, labels_train, weight_base, Kbase):
        batch_size, num_train_examples, num_features = features_train.size()
        nKbase = weight_base.size(1)  # [batch_size x nKbase x num_features]
        labels_train_transposed = labels_train.transpose(1, 2)
        nKnovel = labels_train_transposed.size(
            1
        )  # [batch_size x nKnovel x num_train_examples]

        features_train = features_train.view(
            batch_size * num_train_examples, num_features
        )
        Qe = self.queryLayer(features_train)
        Qe = Qe.view(batch_size, num_train_examples, self.nFeat)
        Qe = F.normalize(Qe, p=2, dim=Qe.dim() - 1, eps=1e-12)

        wkeys = self.wkeys[Kbase.view(-1)]  # the keys of the base categoreis
        wkeys = F.normalize(wkeys, p=2, dim=wkeys.dim() - 1, eps=1e-12)
        # Transpose from [batch_size x nKbase x nFeat] to
        # [batch_size x self.nFeat x nKbase]
        wkeys = wkeys.view(batch_size, nKbase, self.nFeat).transpose(1, 2)

        # Compute the attention coeficients
        # batch matrix multiplications: AttentionCoeficients = Qe * wkeys ==>
        # [batch_size x num_train_examples x nKbase] =
        #   [batch_size x num_train_examples x nFeat] * [batch_size x nFeat x nKbase]
        AttentionCoeficients = self.scale_att * torch.bmm(Qe, wkeys)
        AttentionCoeficients = F.softmax(
            AttentionCoeficients.view(batch_size * num_train_examples, nKbase)
        )
        AttentionCoeficients = AttentionCoeficients.view(
            batch_size, num_train_examples, nKbase
        )

        # batch matrix multiplications: weight_novel = AttentionCoeficients * weight_base ==>
        # [batch_size x num_train_examples x num_features] =
        #   [batch_size x num_train_examples x nKbase] * [batch_size x nKbase x num_features]
        weight_novel = torch.bmm(AttentionCoeficients, weight_base)
        # batch matrix multiplications: weight_novel = labels_train_transposed * weight_novel ==>
        # [batch_size x nKnovel x num_features] =
        #   [batch_size x nKnovel x num_train_examples] * [batch_size x num_train_examples x num_features]
        weight_novel = torch.bmm(labels_train_transposed, weight_novel)
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel)
        )

        return weight_novel


class Classifier(nn.Module):
    def __init__(self, opt=None, train_stage="fewshot"):
        super(Classifier, self).__init__()
        if opt is None:
            if train_stage == "base_classification":
                opt = {
                    "classifier_type": "cosine",
                    "weight_generator_type": "none",
                    "nKall": 64,
                    "nFeat": 128 * 5 * 5,
                    "scale_cls": 10,
                    "backbone": "resnet18",
                }
            elif train_stage == "fewshot":
                opt = {
                    "classifier_type": "cosine",
                    "weight_generator_type": "attention_based",
                    "nKall": 64,
                    "nFeat": 128 * 5 * 5,
                    "scale_cls": 10,
                    "scale_att": 10.0,
                    "backbone": "resnet18",
                }
            else:
                raise f"{train_stage} not supported."

        if "backbone" in opt:
            if opt["backbone"] == "resnet50":
                opt["nFeat"] = 2048
            elif opt["backbone"] == "resnet18":
                opt["nFeat"] = 512
            else:
                opt["nFeat"] = 128 * 5 * 5

        self.weight_generator_type = opt["weight_generator_type"]
        # self.weight_generator_type = "none" # @TODO
        self.classifier_type = opt["classifier_type"]
        assert self.classifier_type == "cosine" or self.classifier_type == "dotproduct"

        nKall = opt["nKall"]
        nFeat = opt["nFeat"]
        self.nFeat = nFeat
        self.nKall = nKall

        # 64 x 3200
        weight_base = torch.FloatTensor(nKall, nFeat).normal_(0.0, np.sqrt(2.0 / nFeat))
        self.weight_base = nn.Parameter(weight_base, requires_grad=True)
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        scale_cls = opt["scale_cls"] if ("scale_cls" in opt) else 10.0
        self.scale_cls = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_cls), requires_grad=True
        )

        if self.weight_generator_type == "none":
            # If the weight generator type is `none` then feature averaging
            # is being used. However, in this case the generator does not
            # involve any learnable parameter and thus does not require
            # training.
            self.favgblock = FeatExemplarAvgBlock(nFeat)
        elif self.weight_generator_type == "feature_averaging":
            self.favgblock = FeatExemplarAvgBlock(nFeat)
            self.wnLayerFavg = LinearDiag(nFeat)
        elif self.weight_generator_type == "attention_based":
            # scale_att = opt["scale_att"] if ("scale_att" in opt) else 10.0
            self.favgblock = FeatExemplarAvgBlock(nFeat)
            # self.attblock = AttentionBasedBlock(nFeat, nKall, scale_att=scale_att)
            self.wnLayerFavg = LinearDiag(nFeat)
            # self.wnLayerWatt = LinearDiag(nFeat)
        else:
            raise ValueError(
                "Not supported/recognized type {0}".format(self.weight_generator_type)
            )

    def get_classification_weights(
        self, Kbase_ids, features_train=None, labels_train=None
    ):
        # ***********************************************************************
        # ******** Get the classification weights for the base categories *******
        batch_size, nKbase = Kbase_ids.size()
        weight_base = self.weight_base[Kbase_ids.view(-1)]  # Val torch.Size([64, 512])
        weight_base = weight_base.view(
            batch_size, nKbase, -1
        )  # Val torch.Size([1, 64, 512])
        # ***********************************************************************

        if features_train is None or labels_train is None:
            # If training data for the novel categories are not provided then
            # return only the classification weights of the base categories.
            return weight_base

        # ***********************************************************************
        # ******* Generate classification weights for the novel categories ******
        _, num_train_examples, num_channels = features_train.size()
        nKnovel = labels_train.size(2)
        if self.classifier_type == "cosine":
            features_train = F.normalize(
                features_train, p=2, dim=features_train.dim() - 1, eps=1e-12
            )
        if self.weight_generator_type == "none":
            weight_novel = self.favgblock(features_train, labels_train)
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        elif self.weight_generator_type == "feature_averaging":
            weight_novel_avg = self.favgblock(features_train, labels_train)
            weight_novel = self.wnLayerFavg(
                weight_novel_avg.view(batch_size * nKnovel, num_channels)
            )
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        elif self.weight_generator_type == "attention_based":
            weight_novel_avg = self.favgblock(
                features_train, labels_train
            )  # torch.Size([8, 5, 512]), Val torch.Size([1, 5, 512])
            weight_novel_avg = self.wnLayerFavg(
                weight_novel_avg.view(batch_size * nKnovel, num_channels)
            )  # torch.Size([40, 512]), Val torch.Size([5, 512])
            weight_novel = weight_novel_avg
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        else:
            raise ValueError(
                "Not supported / recognized type {0}".format(self.weight_generator_type)
            )
        # ***********************************************************************

        # Concatenate the base and novel classification weights and return them.
        weight_both = torch.cat(
            [weight_base, weight_novel], dim=1
        )  # torch.Size([8, 64, 512]) # Val torch.Size([1, 69, 512])
        # weight_both shape: [batch_size x (nKbase + nKnovel) x num_channels]

        return weight_both

    def apply_classification_weights(self, features, cls_weights):
        if self.classifier_type == "cosine":
            features = F.normalize(features, p=2, dim=features.dim() - 1, eps=1e-12)
            cls_weights = F.normalize(
                cls_weights, p=2, dim=cls_weights.dim() - 1, eps=1e-12
            )

        cls_scores = self.scale_cls * torch.baddbmm(
            1.0, self.bias.view(1, 1, 1), 1.0, features, cls_weights.transpose(1, 2)
        )
        return cls_scores

    def forward(
        self,
        features_test,
        Kbase_ids,
        features_train=None,
        labels_train=None,
        gann=None,
        gann2=None,
    ):
        cls_weights = self.get_classification_weights(
            Kbase_ids, features_train, labels_train
        )
        # torch.Size([8, 59]) torch.Size([8, 25, 3200]) torch.Size([8, 25, 5])
        # cls_weights: torch.Size([8, 64, 3200])
        if gann:
            # print('using gaann')
            cls_weights = gann(cls_weights, cls_weights, cls_weights)
        if gann2:
            cls_weights = gann2(cls_weights, cls_weights, cls_weights)
        cls_scores = self.apply_classification_weights(features_test, cls_weights)
        # cls_scores =  cls_scores * 16
        # torch.Size([8, 30, 64])
        return cls_scores


class Resnet_Network_Teacher(nn.Module):
    def __init__(self):
        super(Resnet_Network_Teacher, self).__init__()
        backbone = backbone_.resnet18(pretrained=False)  # resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ["avgpool", "fc"]:
                self.features.add_module(name, module)

        self.pool_method = nn.AdaptiveMaxPool2d(1)  # as default

        num_class = 64
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(512, num_class)

    def forward(self, input):
        x = self.features(input)
        x = self.pool_method(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
