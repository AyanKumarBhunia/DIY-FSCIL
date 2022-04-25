from network import *
from torch import optim
import torch
import time
from gradient_surgery import get_agreement_func

# from pcgrad import PCGrad
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sketch_Classification(nn.Module):
    def __init__(self, hp):
        super(Sketch_Classification, self).__init__()
        self.Network = eval(hp.backbone_name + "_Network(hp)")
        self.train_params = self.parameters()
        self.optimizer = optim.Adam(self.train_params, hp.learning_rate)
        # self.optimizer = PCGrad(optim.Adam(self.train_params, hp.learning_rate))
        self.loss = nn.CrossEntropyLoss()
        self.hp = hp
        self.grad_fn = get_agreement_func("agr-sum")

    def get_grads(self):
        grads = []
        for p in self.Network.parameters():
            grads.append(p.grad.data.clone().flatten())
        return torch.cat(grads)

    def update_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.Network.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end

    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        domain_grads = []
        train_loss, train_acc = 0.0, 0.0

        with torch.set_grad_enabled(True):
            inputs, targets = batch["image"].to(device), batch["label"].to(device)

            outputs = self.Network(inputs)
            loss = self.loss(outputs, targets)
            loss.backward()

            domain_grads.append(self.get_grads())

            predictions = torch.max(outputs, 1)[1]
            train_loss += loss.item()
            train_acc += (predictions == targets).float().mean().item()

            self.optimizer.zero_grad()

        # train_loss /= len(train_batches)
        # train_acc /= len(train_batches)

        new_grads = self.grad_fn(domain_grads)  # Modify gradients according to grad_fn
        self.update_grads(new_grads)  # Update gradients
        self.optimizer.step()  # Update model parameters
        return train_loss
        return train_loss, train_acc

        # output = self.Network(batch['sketch_img'].to(device), [x.to(device) for x in batch['sketch_boxes']])
        output = self.Network(batch["image"].to(device))
        loss = self.loss(output, batch["label"].to(device))

        # losses = [loss, loss_sketch]
        # assert len(losses) == 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Network.parameters(), 10)
        self.optimizer.step()

        # self.optimizer.pc_backward(losses) # calculate the gradient can apply gradient modification
        self.optimizer.step()  # apply gradient step
        return loss.item()
        return loss.item() + loss_sketch.item()

    def evaluate(self, dataloader_Test):
        self.eval()
        correct = 0
        test_loss = 0
        start_time = time.time()
        for i_batch, batch in enumerate(dataloader_Test):

            output = self.Network(batch["image"].to(device))
            test_loss += self.loss(output, batch["label"].to(device)).item()
            prediction = output.argmax(dim=1, keepdim=True).to("cpu")
            correct += prediction.eq(batch["label"].view_as(prediction)).sum().item()

        test_loss /= len(dataloader_Test.dataset)
        accuracy = 100.0 * correct / len(dataloader_Test.dataset)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time_Takes: {}\n".format(
                test_loss,
                correct,
                len(dataloader_Test.dataset),
                accuracy,
                (time.time() - start_time),
            )
        )

        return accuracy
