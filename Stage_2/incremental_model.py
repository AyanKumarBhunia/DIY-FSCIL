from torch.autograd import Variable
import torch.nn as nn

# from Networks import ConvNet, Classifier, Resnet_Network
from torch import optim
import torch
import torch.nn.functional as F
from utils import *
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gradient_surgery import get_agreement_func
import torchvision.models as backbone_
from torch.nn.parameter import Parameter
import math
import scipy.sparse as sp
from backbones import * 

SURGERY = True
GANN = True
SKIP_GRAD_ISSUES = True
USE_KD = True

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.FloatTensor(in_features, out_features), requires_grad=True
        )  # .cuda()
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


def corrcoef(tensor, rowvar=True):
    """Get Pearson product-moment correlation coefficients (np.corrcoef)"""
    covariance = cov(tensor, rowvar=rowvar)
    variance = covariance.diagonal(0, -1, -2)
    if variance.is_complex():
        variance = variance.real
    stddev = variance.sqrt()
    covariance /= stddev.unsqueeze(-1)
    covariance /= stddev.unsqueeze(-2)
    if covariance.is_complex():
        covariance.real.clip_(-1, 1)
        covariance.imag.clip_(-1, 1)
    else:
        covariance.clip_(-1, 1)
    return covariance


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.5):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        # output = self.dropout(self.fc(output))
        # output = self.layer_norm(output + residual)
        output = self.fc(self.dropout(output))
        output = output + residual
        return output


def loss_fn_kd(outputs, teacher_outputs, params=None):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    if params is None:
        params = {
            "alpha": 0.95,
            "temperature": 6,
        }
    alpha = 0.95
    T = 20  # 6
    # alpha, T = 0.9, 10
    """
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    """
    KD_loss = nn.KLDivLoss()(
        F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)
    ) * (alpha * T * T)
    return KD_loss


class Incremental_Fewshot_Model(nn.Module):
    def __init__(self, hp):
        super(Incremental_Fewshot_Model, self).__init__()

        print("Loading: ", hp.backbone)
        # use resnet50
        if hp.backbone == "resnet18":
            self.feat_model = Resnet_Network()
        else:
            self.feat_model = ConvNet()
        self.classifier = Classifier(train_stage=hp.train_stage)
        self.gann = MultiHeadAttention(1, 512, 512, 512)
        self.teacher = Resnet_Network_Teacher()
        self.teacher.to(hp.device)
        self.teacher.eval()
        self.teacher_classifier_weight = self.teacher.classifier.weight
        self.teacher_classifier_bias = self.teacher.classifier.bias

        weights_path = os.path.join(
            hp.base_dir, "pretrained_models/crossmodal.pth"
        )  # @TODO
        w = torch.load(weights_path, map_location="cpu")
        self.teacher.load_state_dict(w, strict=False)
        self.teacher.eval()

        if hp.mode == "evaluation":
            feat_model_path = os.path.join(
                hp.saved_models,
                "stage2",
                "incremental_feat_model_{}.pth".format(hp.pretrained_model_type),
            )
            if hp.device == "cpu":
                feat_model_weights = torch.load(
                    feat_model_path, map_location=torch.device("cpu")
                )
            else:
                feat_model_weights = torch.load(feat_model_path)

            classifier_model_path = os.path.join(
                hp.saved_models,
                "stage2",
                "incremental_classifier_{}.pth".format(hp.pretrained_model_type),
            )
            if hp.device == "cpu":
                classifier_weights = torch.load(
                    classifier_model_path, map_location=torch.device("cpu")
                )
            else:
                classifier_weights = torch.load(classifier_model_path)

            gann_model_path = os.path.join(
                hp.saved_models,
                "stage2",
                "incremental_gann_{}.pth".format(hp.pretrained_model_type),
            )
            if hp.device == "cpu":
                gann_weights = torch.load(
                    gann_model_path, map_location=torch.device("cpu")
                )
            else:
                gann_weights = torch.load(gann_model_path)

            print(
                "Loading weights from {}, {} and {}".format(
                    feat_model_path, classifier_model_path, gann_model_path
                )
            )
            self.feat_model.load_state_dict(feat_model_weights)
            self.classifier.load_state_dict(classifier_weights)
            self.gann.load_state_dict(gann_weights)
            print("Pretrained models loaded from stage2 for testing.....")
        elif hp.train_stage == "fewshot":
            pass
            """
            feat_model_path = os.path.join(hp.saved_models, 'stage1', "feat_model_best.pth")
            if hp.device == 'cpu':
                feat_model_weights = torch.load(feat_model_path, map_location=torch.device('cpu'))
            else:
                feat_model_weights = torch.load(feat_model_path)

            classifier_model_path = os.path.join(hp.saved_models, 'stage1', "classifier_best.pth")
            if hp.device == 'cpu':
                classifier_weights = torch.load(classifier_model_path, map_location=torch.device('cpu'))
            else:
                classifier_weights = torch.load(classifier_model_path)
            print('Loading weights from {} and {}'.format(feat_model_path, classifier_model_path))
            self.feat_model.load_state_dict(feat_model_weights)
            self.classifier.load_state_dict(classifier_weights, strict=False)
            print('Pretrained models loaded from stage1.....')
            """

        self.feat_model.to(hp.device)
        self.classifier.to(hp.device)
        self.gann.to(hp.device)
        self.grad_fn = get_agreement_func("agr-sum")

        self.hp = hp
        self._init_optimizers()

        self.curr_epoch = 0
        self.criterion = nn.CrossEntropyLoss()

        self.nKbase = torch.LongTensor()
        self.allocate_tensors()
        # self.activate_dropout = (
        #    opt["activate_dropout"] if ("activate_dropout" in opt) else False
        # )

    def _init_optimizers(self):
        if self.hp.optim == "adam":
            if self.hp.train_stage == "base_classification":
                # self.train_params = list(self.feat_model.parameters()) + list(
                #                            self.classifier.parameters()
                #                        )
                params = [
                    {"params": self.feat_model.parameters(), "lr": self.hp.lr},
                    {"params": self.classifier.parameters(), "lr": self.hp.lr},
                ]
                self.optimizer = optim.Adam(params, self.hp.lr)
            elif self.hp.train_stage == "fewshot":
                params = [
                    # {"params": self.feat_model.parameters(), "lr": self.hp.lr},
                    {"params": self.classifier.parameters(), "lr": self.hp.lr},
                ]
                self.optimizer = optim.Adam(params, self.hp.lr)
            else:
                raise "train_stage {} not supported ".format(self.hp.train_stage)
        elif self.hp.optim == "sgd":
            if self.hp.train_stage == "fewshot":
                params = [
                    # {"params": self.feat_model.parameters(), "lr": self.hp.lr},
                    {"params": self.classifier.parameters(), "lr": self.hp.lr},
                    {"params": self.gann.parameters(), "lr": self.hp.lr},
                ]
                model_parameters = filter(
                    lambda p: p.requires_grad, self.classifier.parameters()
                )
                classifier_params = sum([np.prod(p.size()) for p in model_parameters])
                model_params = filter(lambda p: p.requires_grad, self.gann.parameters())
                gann_params = sum([np.prod(p.size()) for p in model_params])
                print("Training Params: ", classifier_params, gann_params)
                # self.optimizer = optim.Adam(params, self.hp.lr)  # @TODO
                self.optimizer = optim.SGD(
                    params,
                    lr=self.hp.lr,
                    momentum=0.9,
                    nesterov=True,
                    weight_decay=5e-4,
                )
            else:
                raise "train_stage {} not supported ".format(self.hp.train_stage)
        else:
            raise "optimizer {} not supported ".format(self.hp.optim)

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors["images_train"] = torch.FloatTensor()
        self.tensors["labels_train"] = torch.LongTensor()
        self.tensors["labels_train_1hot"] = torch.FloatTensor().to(self.hp.device)
        self.tensors["images_test"] = torch.FloatTensor()
        self.tensors["labels_test"] = torch.LongTensor()
        self.tensors["Kids"] = torch.LongTensor()

    def run_train_epoch(self, data_loader, epoch):
        self.dloader = data_loader
        self.dataset_train = data_loader.dataset

        self.feat_model.train()
        self.classifier.train()

        train_stats = DAverageMeter()
        self.bnumber = len(data_loader())
        for idx, batch in enumerate(tqdm(data_loader(epoch))):
            self.biter = idx  # batch iteration.
            # self.global_iter = self.curr_epoch * len(data_loader) + self.biter
            train_stats_this = self.train_step(batch)
            if idx % self.hp.disp_step == 0:
                print("Train: ", idx, "avg: ", train_stats, train_stats_this)
            train_stats.update(train_stats_this)
        return train_stats.average()

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def set_tensors(self, batch):
        self.nKbase = self.dloader.nKbase
        self.nKnovel = self.dloader.nKnovel

        if self.nKnovel > 0:
            train_test_stage = "fewshot"
            assert len(batch) == 6
            images_train, labels_train, images_test, labels_test, K, nKbase = batch
            if nKbase.shape[0] == 1:
                self.nKbase = nKbase.squeeze().item()
            else:
                self.nKbase = nKbase.squeeze()[0]

            self.tensors["images_train"].resize_(images_train.size()).copy_(
                images_train
            ).to(self.hp.device)
            self.tensors["labels_train"].resize_(labels_train.size()).copy_(
                labels_train
            ).to(self.hp.device)
            labels_train = self.tensors["labels_train"].to(self.hp.device)

            nKnovel = 1 + labels_train.max() - self.nKbase

            labels_train_1hot_size = list(labels_train.size()) + [
                nKnovel,
            ]
            labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
            self.tensors["labels_train_1hot"].resize_(labels_train_1hot_size).fill_(
                0
            ).scatter_(
                len(labels_train_1hot_size) - 1, labels_train_unsqueeze - self.nKbase, 1
            ).to(
                self.hp.device
            )
            self.tensors["images_test"].resize_(images_test.size()).copy_(
                images_test
            ).to(self.hp.device)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(
                labels_test
            ).to(self.hp.device)
            self.tensors["Kids"].resize_(K.size()).copy_(K).to(self.hp.device)
        else:
            train_test_stage = "base_classification"
            assert len(batch) == 4
            images_test, labels_test, K, nKbase = batch
            self.nKbase = nKbase.squeeze()[0]
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)

        return train_test_stage

    def process_batch(self, batch, do_train):
        process_type = self.set_tensors(batch)

        if process_type == "fewshot":
            record = self.process_batch_fewshot_without_forgetting(do_train=do_train)
        elif process_type == "base_classification":
            record = self.process_batch_base_category_classification(do_train=do_train)
        else:
            raise ValueError("Unexpected process type {0}".format(process_type))

        return record

    def get_grads(self):
        grads = []

        # for p in self.parameters():
        #    grads.append(p.grad.data.clone().flatten())
        """
        for p in self.feat_model.parameters():
            grads.append(p.grad.data.clone().flatten())
        for p in self.classifier.parameters():
            grads.append(p.grad.data.clone().flatten())
        for p in self.gnn.parameters():
            grads.append(p.grad.data.clone().flatten())
        """
        for p in self.classifier.parameters():
            grads.append(p.grad.data.clone().flatten())
        # model_params = filter(lambda p: p.requires_grad, self.gann.parameters())
        # gann_params = sum([np.prod(p.size()) for p in model_params])
        # print(self.gann.parameters(), gann_params)
        for param in self.gann.parameters():
            if SKIP_GRAD_ISSUES:
                try:
                    grads.append(param.grad.data.clone().flatten())
                except:
                    pass
            else:
                grads.append(param.grad.data.clone().flatten())
        return torch.cat(grads)

    def update_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.classifier.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end
        for k, p in enumerate(self.gann.parameters()):
            if SKIP_GRAD_ISSUES:
                try:
                    dims = p.shape
                    end = start + dims.numel()
                    p.grad.data = new_grads[start:end].reshape(dims)
                    start = end
                except:
                    pass
            else:
                dims = p.shape
                end = start + dims.numel()
                p.grad.data = new_grads[start:end].reshape(dims)
                start = end

        """
        for k, p in enumerate(self.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end
        """

        """
        for k, p in enumerate(self.Network.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end
        """

    def replace_to_rotate(self, support, query):
        # print(support.shape, query.shape)
        rot_list = [0, 0, 90, 180, 270]
        # rot_list = [0, 0, 90]
        sel_rot = random.choice(rot_list)
        if sel_rot == 90:  # rotate 90 degree
            support = support.transpose(3, 4).flip(3)
            query = query.transpose(3, 4).flip(3)
        elif sel_rot == 180:  # rotate 180 degree
            support = support.flip(3).flip(4)
            # query = query.flip(3).flip(4)
        elif sel_rot == 270:  # rotate 270 degree
            support = support.transpose(3, 4).flip(4)
            # query = query.transpose(3, 4).flip(4)

        # print('updated: ', support.shape, query.shape)
        return support, query

    def process_batch_fewshot_without_forgetting(self, do_train=True):
        images_train = self.tensors["images_train"]  # torch.Size([8, 25, 3, 84, 84])
        labels_train = self.tensors["labels_train"]  # torch.Size([8, 25])
        labels_train_1hot = self.tensors["labels_train_1hot"]  # torch.Size([8, 25, 5])
        images_test = self.tensors["images_test"]  # torch.Size([8, 30, 3, 84, 84])
        labels_test = self.tensors["labels_test"]  # torch.Size([8, 30])
        Kids = self.tensors["Kids"]  # 64, [0, .., 63]
        nKbase = self.nKbase
        #  torch.Size([1, 5, 3, 84, 84]) torch.Size([1, 30, 3, 84, 84])
        #  torch.Size([1, 5, 3, 84, 84]) torch.Size([1, 750, 3, 84, 84])
        # print('Do Train: ', do_train, images_train.shape, images_test.shape)

        criterion = self.criterion

        do_train_feat_model = True  # @TODO
        """
        do_train_feat_model = do_train and self.optimizers["feat_model"] is not None
        if not do_train_feat_model:
            feat_model.eval()
            if do_train and self.activate_dropout:
                # Activate the dropout units of the feature extraction model
                # even if the feature extraction model is freezed (i.e., it is
                # in eval mode).
                activate_dropout_units(feat_model)
        """

        if do_train:  # zero the gradients
            # if do_train_feat_model:
            #    self.optimizers["feat_model"].zero_grad()
            self.optimizer.zero_grad()
            self.feat_model.train()
            self.classifier.train()
            self.gann.train()

            images_train, images_test = self.replace_to_rotate(
                images_train, images_test
            )
        # ***********************************************************************
        # *********************** SET TORCH VARIABLES ***************************
        domain_grads = []
        with torch.set_grad_enabled(do_train):
            is_volatile = not do_train or not do_train_feat_model
            images_test_var = Variable(images_test, volatile=is_volatile).to(
                self.hp.device
            )
            labels_test_var = Variable(labels_test, requires_grad=False).to(
                self.hp.device
            )
            Kbase_var = (
                None
                if (nKbase == 0)
                else Variable(Kids[:, :nKbase].contiguous(), requires_grad=False)
            )
            labels_train_1hot_var = Variable(labels_train_1hot, requires_grad=False).to(
                self.hp.device
            )
            images_train_var = Variable(images_train, volatile=is_volatile).to(
                self.hp.device
            )
            # ***********************************************************************

            loss_record = {}
            # ***********************************************************************
            # ************************* FORWARD PHASE: ******************************

            # ************ EXTRACT FEATURES FROM TRAIN & TEST IMAGES ****************
            (
                batch_size,
                num_train_examples,
                channels,
                height,
                width,
            ) = images_train.size()
            num_test_examples = images_test.size(1)
            features_train_var = self.feat_model(
                images_train_var.view(
                    batch_size * num_train_examples, channels, height, width
                )
            )  # torch.Size([200, 3200]) # Val [25, 512]
            features_test_var = self.feat_model(
                images_test_var.view(
                    batch_size * num_test_examples, channels, height, width
                )
            )  # torch.Size([240, 3200]) # Val [750, 512]
            features_train_var = features_train_var.view(
                [
                    batch_size,
                    num_train_examples,
                ]
                + list(features_train_var.size()[1:])
            )  # torch.Size([8, 25, 3200]) # Val torch.Size([1, 25, 512])
            features_test_var = features_test_var.view(
                [
                    batch_size,
                    num_test_examples,
                ]
                + list(features_test_var.size()[1:])
            )  # torch.Size([8, 30, 3200]) # Val torch.Size([1, 750, 512])
            if (not do_train_feat_model) and do_train:
                # Make sure that no gradients are backproagated to the feature
                # extractor when the feature extraction model is freezed.
                features_train_var = Variable(
                    features_train_var.data, volatile=False
                ).to(self.hp.device)
                features_test_var = Variable(features_test_var.data, volatile=False).to(
                    self.hp.device
                )
            # ***********************************************************************

            # ************************ APPLY CLASSIFIER *****************************
            if self.nKbase > 0:  # tensor(59)
                cls_scores_var = self.classifier(
                    features_test=features_test_var,
                    Kbase_ids=Kbase_var,
                    features_train=features_train_var,
                    labels_train=labels_train_1hot_var,
                    # gann=self.gann,
                    gann=self.gann if GANN else None,
                )  # torch.Size([8, 30, 64]) # Val torch.Size([1, 750, 69])
            else:
                cls_scores_var = self.classifier(
                    features_test=features_test_var,
                    features_train=features_train_var,
                    labels_train=labels_train_1hot_var,
                    # gann=self.gann if GANN else None,
                )

            cls_scores_var = cls_scores_var.view(batch_size * num_test_examples, -1)
            # torch.Size([240, 64])
            labels_test_var = labels_test_var.view(batch_size * num_test_examples)
            # torch.Size([240])
            # ***********************************************************************

            # ************************* COMPUTE LOSSES ******************************
            loss_cls_all = criterion(cls_scores_var, labels_test_var)
            loss_total = loss_cls_all
            """
            # grad surgery
            if do_train:
                loss_total.backward()
                domain_grads.append(self.get_grads())
                self.optimizer.zero_grad()
            """
            if do_train:
                if USE_KD:
                    feat = self.teacher.features(
                        images_test_var.view(
                            batch_size * num_test_examples, channels, height, width
                        )
                    )
                    feat = nn.AdaptiveMaxPool2d(1)(feat)
                    feat = torch.flatten(feat, 1)
                    updated_weights = self.teacher_classifier_weight[Kids.view(-1)]
                    updated_bias = self.teacher_classifier_bias[Kids.view(-1)]
                    self.teacher.classifier.weight = nn.Parameter(updated_weights)
                    self.teacher.classifier.bias = nn.Parameter(updated_bias)
                    soft_labels = self.teacher.classifier(feat)
                    kd_loss = loss_fn_kd(cls_scores_var, soft_labels, params=None)

                    loss_total = (loss_cls_all) * (1 - 0.95) + kd_loss
                else:
                    loss_total = loss_cls_all
                # loss_total = loss_cls_all + nn.MSELoss()(cls_scores_var, soft_labels)
                # delta = torch.abs(cls_scores_var - soft_labels)
                # kd_loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
                # loss_total = loss_cls_all * (1 - 0.95) + 0.05 * kd_loss
                # loss_total = loss_cls_all * 0.8  + 0.2 * RKDLoss()(cls_scores_var, soft_labels)
                # loss_total = (loss_cls_all) * (1 - 0.9) + kd_loss
                # grad surgery
                # loss_total = loss_cls_all
                # """
                if SURGERY:
                    loss_total.backward()
                    domain_grads.append(self.get_grads())
                    self.optimizer.zero_grad()
                # """
            else:
                loss_total = loss_cls_all

            loss_record["loss"] = loss_total.item()

        if self.nKbase > 0:
            loss_record["AccuracyBoth"] = top1accuracy(
                cls_scores_var.data, labels_test_var.data
            )

            preds_data = cls_scores_var.data.cpu()
            labels_test_data = labels_test_var.data.cpu()
            base_ids = torch.nonzero(labels_test_data < self.nKbase).view(-1)
            novel_ids = torch.nonzero(labels_test_data >= self.nKbase).view(-1)
            preds_base = preds_data[base_ids, :]  # Val torch.Size([375, 69])
            preds_novel = preds_data[novel_ids, :]  # Val torch.Size([375, 69])

            loss_record["AccuracyBase"] = top1accuracy(
                preds_base[:, :nKbase], labels_test_data[base_ids]
            )
            loss_record["AccuracyNovel"] = top1accuracy(
                preds_novel[:, nKbase:], (labels_test_data[novel_ids] - nKbase)
            )
        else:
            loss_record["AccuracyNovel"] = top1accuracy(
                cls_scores_var.data, labels_test_var.data
            )
        # ***********************************************************************

        # ***********************************************************************
        # ************************* BACKWARD PHASE ******************************
        if do_train:
            # """
            # grad surgery
            if SURGERY:
                new_grads = self.grad_fn(
                    domain_grads
                )  # Modify gradients according to grad_fn
                self.update_grads(new_grads)  # Update gradients
                self.optimizer.step()  # Update model parameters
            else:
                loss_total.backward()
                self.optimizer.step()
            """
            loss_total.backward()
            self.optimizer.step()
            """
        # ***********************************************************************

        if not do_train:
            if self.biter == 0:
                self.test_accuracies = {"AccuracyNovel": []}
            self.test_accuracies["AccuracyNovel"].append(loss_record["AccuracyNovel"])
            if self.biter == (self.bnumber - 1):
                # Compute the std and the confidence interval of the accuracy of
                # the novel categories.
                stds = np.std(np.array(self.test_accuracies["AccuracyNovel"]), 0)
                ci95 = 1.96 * stds / np.sqrt(self.bnumber)
                loss_record["AccuracyNovel_std"] = stds
                loss_record["AccuracyNovel_cnf"] = ci95

        return loss_record

    def adjust_learning_rates(self, epoch):
        # LUT = [(20, 0.1), (40, 0.006), (50, 0.0012), (60, 0.00024)]
        LUT = [(2, 0.1), (5, 0.01), (10, 0.006), (20, 0.0012), (30, 0.00024)]
        # LUT = [(10, 0.006), (20, 0.0012), (30, 0.00024)]
        lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])

        for param_group in self.optimizer.param_groups:
            if lr != param_group["lr"]:
                print(
                    "Epcoh: {} lr updated from {} to {}".format(
                        epoch, param_group["lr"], lr
                    )
                )
            param_group["lr"] = lr

    def evaluate(self, dloader):
        self.dloader = dloader
        self.dataset_eval = dloader.dataset

        self.feat_model.eval()
        self.classifier.eval()
        self.gann.eval()

        eval_stats = DAverageMeter()
        self.bnumber = len(dloader)
        for idx, batch in enumerate(tqdm(dloader())):
            self.biter = idx
            eval_stats_this = self.evaluation_step(batch)
            eval_stats.update(eval_stats_this)
            if idx % self.hp.disp_step == 0:
                print("Eval: ", idx, "avg: ", eval_stats, eval_stats_this)

        return eval_stats.average()

    """
    def _to_one_hot(self, y, num_classes=64):
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype).to(self.hp.device)

        return zeros.scatter(scatter_dim, y_tensor, 1).squeeze()
    """


# def inv_correlation(y_true, y_pred):
#    """Computes 1 minus the dot product between corresponding pairs of samples in two tensors."""
#    return 1.0 - torch.sum(y_true * y_pred, axis=-1)


# def l2norm(x):
#    """L2-normalizes a tensor along the last axis."""
#    x_norm = torch.norm(x.clone(), 2, 1, keepdim=True)
#    return x / x_norm
