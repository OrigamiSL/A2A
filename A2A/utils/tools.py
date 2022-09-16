import numpy as np
import torch
import torch.nn.functional as F


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        args.learning_rate = args.learning_rate * 0.5
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


def loss_process(args, pred, true, criterion=None, flag=0, dataset=None):
    if flag == 2:
        if dataset:
            weight_pred = pred.reshape(-1, pred.shape[-1])
            weight_pred = dataset.inverse_transform(weight_pred.detach().cpu().numpy())
            weight_pred = dataset.standard_transformer(weight_pred)
            return weight_pred
        else:
            return pred

    if flag == 0:
        if args.area_loss:
            thres = 0.5 * args.jitter  # Area Loss
            if args.aug == 'Scaling':
                true_up = true * (1 + thres)
                true_down = true * (1 - thres)
            else:  # 'Jittering'
                true_up = true + thres
                true_down = true - thres
            up_mat = torch.ge(true_up, pred).int()
            down_mat = torch.ge(pred, true_down).int()
            total_mat = up_mat + down_mat
            pred = torch.where(total_mat == 2, true, pred)
            loss1 = criterion(pred, true)
            loss2 = criterion(pred, true_down)
            loss3 = criterion(pred, true_up)
            A_mat = torch.ge(loss1, loss2).int()
            loss = torch.where(A_mat > 0, loss2, loss1)
            B_mat = torch.ge(loss, loss3).int()
            loss = torch.where(B_mat > 0, loss3, loss)
        else:
            loss = criterion(pred, true)
        return loss
    else:
        loss2 = criterion(pred, true)
        return loss2.detach().cpu().numpy()


def Cost_loss(z_list, aug_num=3, order_num=1, instance_loss=True):
    assert (len(z_list) > 0)
    assert (z_list[0] is not None)
    loss = torch.zeros(len(z_list)).to(z_list[0].device)
    alpha = 0.5 if instance_loss else 1
    for index, z in enumerate(z_list):
        B, V, D = z.shape
        z_ins = z.clone()
        # temporal loss
        z = z.contiguous().permute(1, 0, 2)  # B, V, D --> V, B, D
        z = sim(z)  # [V, B, D] * [V, D, B] --> [V, B, B]
        ins = torch.split(z, (aug_num + order_num), 1)
        assert (B % (aug_num + order_num) == 0)
        for i in range(B // (aug_num + order_num)):
            initial_ins = ins[i][:, 0, :]  # V, B
            soft_ins = F.softmax(initial_ins, dim=-1)  # loss function
            logs = torch.split(soft_ins, aug_num + order_num, 1)
            sum_ins = torch.sum(logs[i][:, :aug_num], dim=1)
            log_ins = -torch.log(sum_ins)
            loss[index] += alpha * torch.mean(log_ins) / (B // (aug_num + order_num))
        # instance loss
        if instance_loss:
            z_ins = z_ins.contiguous()  # B, V, D
            z_ins = sim(z_ins)  # [B, V, D] * [B, D, V] --> [B, V, V]
            ins = torch.split(z_ins, (aug_num + order_num), 0)
            for i in range(B // (aug_num + order_num)):
                initial_ins = ins[i][0, :, :]  # V, V
                soft_ins = F.softmax(initial_ins, dim=-1)  # loss function
                diag_ins = torch.diag(soft_ins)
                log_ins = torch.mean(-torch.log(diag_ins))
                loss[index] += alpha * log_ins / (B // (aug_num + order_num))
        return torch.mean(loss)


def sim(z):
    # z1-->inner product, z2-->norm
    z1 = torch.matmul(z, z.transpose(1, 2))
    z_sum = torch.sum(z ** 2, dim=-1, keepdim=True)
    z_sum = torch.sqrt(z_sum)
    z2 = torch.matmul(z_sum, z_sum.transpose(1, 2))
    return z1 / z2
