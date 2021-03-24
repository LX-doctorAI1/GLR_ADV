import numpy as np
from torchvision import transforms
from tqdm import tqdm
import pickle as pkl
import torch
import pickle
import time
import torchvision
from torch.utils.data import DataLoader
import dataLoader

device = torch.device('cuda:6')


def evalHot(y, pred):
    """
    评估效果
    :param y:真实值的独热编码
    :param pred: 预测值的输出
    :return: 正确的个数
    """
    _y = torch.argmax(y, dim=-1)
    _pred = torch.argmax(pred, dim=-1)
    N = np.sum((_y == _pred).cpu().numpy())
    return N


def KMeansRepeatX(X, repeat, train=True):
    """
    :param X:Raw data \\in R^{batch_size X n_dim}
    :param repeat:重复的次数、采样数
    :return: 加了偏置项和重复数据的样本 维度[batch_size,repeat,n_dum+1]
    """
    X = X.reshape(len(X), -1)
    if train:
        repeatX = torch.cat([X] * repeat, dim=0).to(device)
        one_shape = tuple(repeatX.shape[:-1]) + (1,)
        one = torch.ones(size=one_shape, dtype=torch.float).to(device)
        return torch.cat([repeatX, one], dim=-1)
    else:
        one = torch.ones(tuple(X.shape[:-1]) + (1,), dtype=torch.float).to(device)
        return torch.cat([X, one], dim=-1)


def OneHotLabel(Y, n):
    """
    :param Y:序列型标签
    :param n: 标签数目
    :return: 标签的独热编码
    """
    y = torch.zeros([len(Y), n]).to(device)
    y[torch.arange(0, len(Y)), Y] = 1
    return y


def KMeansRepeatY(Y, repeat):
    # print(Y.shape)
    repeatY = torch.cat([Y] * repeat, dim=0)
    return repeatY


class Activation:
    """
    包含激活函数
    """

    @staticmethod
    def logistic(z):
        return 1 / (1 + torch.exp(-z))

    @staticmethod
    def softmax(z):
        stable_exps = torch.exp(z)
        return stable_exps / stable_exps.sum(dim=-1, keepdim=True)

    @staticmethod
    def threshold(z):
        z[z < 0] = 0
        return torch.sign(z)

    @staticmethod
    def relu(z):
        z[z < 0] = 0
        return z


def CELoss(Y, T):
    """
    :param Y:模型输出
    :param T: 样本标签
    :return: 交叉熵损失
    """
    return -(T * torch.log(Y)).sum(dim=-1)

def ZeroOneLoss(Y, T):
    """
    :param Y:模型输出
    :param T: 样本标签
    :return: 0-1 损失
    """
    _Y = torch.argmax(Y, dim=-1)
    _T = torch.argmax(T, dim=-1)
    loss = torch.as_tensor(_Y != _T, dtype=torch.float, device=device)
    return loss

class Layer:
    def __init__(self, n_input, n_output, sigma, activation):
        """
        :param n_input:输入维度
        :param n_output: 输出维度
        :param sigma: 方差
        :param activation: 激活函数
        """
        self.w = torch.randn(size=[n_input, n_output]).to(device)  # 多出来的是bias
        self.w *= (2 / self.w.shape[0] ** 0.5)
        self.sigma = sigma
        self.n_input = n_input
        self.n_output = n_output
        self.input = None
        self.output = None
        self.noise = None
        self.activation = activation
        self.bp_grad = None
        self.lr_grad = None
        self.batch_bp_grad = None
        self.batch_lr_grad = None
        print('该层方差为:{}'.format(sigma))
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_last_mt = 0
        self.adam_last_vt = 0
        self.t = 0
        self.epsilon = 1e-8
        self.nadam_alpha = 0.9
        self.nadam_beta = 0.999
        self.nadam_s = 0
        self.nadam_r = 0
        self.momentum = 0
        self.momentum_beta = 0.9
        self.adagard_r = 0.


    def get_params(self):
        return self.w.cpu()

    def forward(self, x, train=False, BP=False):
        self.input = x
        if BP:
            # print(self.input.shape)
            # print(self.w.shape)
            self.output = self.input.matmul(self.w)
            if self.activation:
                self.output = self.activation(self.output)
            return self.output
        else:
            if not train:
                self.output = self.input.matmul(self.w)
                if self.activation:
                    self.output = self.activation(self.output)
                return self.output
            else:
                # self.noise = torch.randn([len(self.input), self.n_output]) * self.sigma
                self.noise = np.random.laplace(loc=0, scale=self.sigma, size=[len(self.input), self.n_output])
                self.noise = torch.from_numpy(self.noise).to(device).to(torch.float32)
                self.output = self.input.matmul(self.w) + self.noise
                if self.activation:
                    self.output = self.activation(self.output)
                return self.output

    def backward(self, target, BP=True):
        """
        :param target: BP训练模式下，target是残差；LR训练模式下，target是损失
        :param BP: 是否为BP训练
        :return: BP训练模式下，返回残差；LR训练模式下，返回损失
        """
        if BP:
            eta = target
            if self.activation == Activation.softmax:
                eta = self.output - eta
            elif self.activation == Activation.logistic:
                eta = self.output * (1 - self.output) * eta
            else:
                print('尚未注册！\n')
                exit()
            batch_size = len(self.input)
            grad = self.input.T.matmul(eta)
            self.bp_grad = grad / batch_size
            return torch.einsum('ij,kj->ik', eta, self.w)
        else:
            term = self.input * target[:, np.newaxis]
            batch_grad = torch.einsum('ni, nj->nij', term, torch.sign(self.noise))
            batch_grad /= self.sigma
            batch_grad = torch.mean(batch_grad, dim=0)
            self.lr_grad = batch_grad
            return target

    def update_params(self, learning_rate, BP=True, method='sgd', weight_decay=1e-4):
        if BP:
            grad = self.bp_grad
        else:
            grad = self.lr_grad
        grad += weight_decay * self.w
        if method == 'sgd':
            self.w -= learning_rate * grad
            return
        elif method == 'adam':
            self.t += 1
            self.adam_last_mt = self.adam_beta1 * self.adam_last_mt + (1 - self.adam_beta1) * grad
            self.adam_last_vt = self.adam_beta2 * self.adam_last_vt + (1 - self.adam_beta2) * grad ** 2
            adam_mt_cap = self.adam_last_mt / (1 - self.adam_beta1 ** self.t)
            adam_vt_cap = self.adam_last_vt / (1 - self.adam_beta2 ** self.t)
            self.w -= learning_rate * adam_mt_cap / (torch.sqrt(adam_vt_cap) + self.epsilon)
            return
        elif method == 'nadam':
            self.nadam_s = self.nadam_alpha * self.nadam_s + (1 - self.nadam_alpha) * grad
            self.nadam_r = self.nadam_beta * self.nadam_r + (1 - self.nadam_beta) * grad ** 2
            sqrt_term = 1 - self.nadam_beta ** self.t
            sqrt_term = sqrt_term ** 0.5
            lr = learning_rate * sqrt_term / (1 - self.nadam_alpha ** self.t)
            delta = lr * (self.nadam_alpha * self.nadam_s + (1 - self.nadam_alpha) * grad) / torch.sqrt(self.nadam_r)
            self.w -= delta
        elif method == 'momentum':
            self.t += 1
            if self.t < 100:
                self.w -= learning_rate * grad
                return
            self.momentum = self.momentum_beta * self.momentum - learning_rate * grad
            self.w += self.momentum
            return
        elif method == 'adagrad':
            self.adagard_r += grad * grad
            self.w -= learning_rate * grad / (self.epsilon + torch.sqrt(self.adagard_r))
            return
        else:
            print('优化方法尚未注册')
            exit()

    def load_weight(self, w):
        self.w = w

class Network(object):
    def __init__(self, n_input, units_per_layers: list, activation_per_layers: list, sigma):
        assert len(units_per_layers) == len(activation_per_layers)
        self.n_layers = len(units_per_layers)
        self.params = [(n_input, units_per_layers[0], sigma, activation_per_layers[0])]
        for i in range(self.n_layers - 1):
            self.params.append(
                (units_per_layers[i], units_per_layers[i + 1], sigma,
                 activation_per_layers[i + 1]))
        self.layers = [Layer(*self.params[i]) for i in range(self.n_layers)]
        print('模型层数为:{} 各层及对应的激活函数为:{}'.format(len(self.layers),
                                               [(units_per_layers[i], activation_per_layers[i]) for i in
                                                range(self.n_layers)]))

    def forward(self, X, train=True, BP=False):
        z = X
        for layer in self.layers:
            # print(BP)
            z = layer.forward(z, train, BP)
        return z

    def backward(self, target, BP=True):
        """
        :param target:BP训练方式下target是标签 LR训练方式下target是损失
        :param BP: 是否为BP模式
        :return: None
        """
        if BP:
            for i in range(self.n_layers - 1, -1, -1):
                target = self.layers[i].backward(target, BP)
        else:
            for layer in self.layers:
                layer.backward(target, BP)

    def update_params(self, learning_rate, BP=True, method='sgd', weight_decay=1e-4):
        for layer in self.layers:
            layer.update_params(learning_rate, BP, method, weight_decay)

    def load_weights(self, w_list):
        for layer_index in range(len(self.layers)):
            self.layers[layer_index].load_weight(w_list[layer_index])
        print('模型加载参数成功！\n')

    def save_weights(self, path):
        w_list = []
        for layer in self.layers:
            w_list.append(layer.get_params())
        with open(path, 'wb') as file:
            pkl.dump(w_list, file)
        print('模型保存参数成功！\n')

if __name__ == "__main__":
    mnist = 'MNIST'
    task = mnist
    train_loss = []
    test_loss = []
    acc = []
    time_list = []
    epoch_train_estimation_relative_error = []
    epoch_test_estimation_relative_error = []
    repeat_n = 100
    is_01_loss = True
    net_arc = [50, 10]
    learning_rate = 1e-2
    sigma = 1.0
    alg_start = 0.
    alg_end = 0.
    net_act = [Activation.threshold,  Activation.softmax]
    assert len(net_arc) == len(net_act)
    n_layers = len(net_arc)
    if task == mnist:
        print('run mnist')
        batch_size = 128
        n_input = 28 * 28 + 1
        n_output = 10
        seed = None
        epoches = 20
        loss = 0.
        num_classes = 10
        reuse = False
        BP_train = False
        print('LRT tarin on')
        transform = transforms.Compose([transforms.ToTensor()])
        start_epoch = 0
        data_dir = './'
        tranform = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.FashionMNIST(data_dir, download=True, train=True, transform=tranform)
        val_dataset = torchvision.datasets.FashionMNIST(root=data_dir, download=True, train=False, transform=tranform)

        # train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=0)
        # test_dataloader = DataLoader(dataset=val_dataset, batch_size=128, num_workers=0, shuffle=False)
        train_dataloader, test_dataloader = dataLoader.LoadMNIST('../data/MNIST', tranform, batch_size, False)
        net = Network(n_input, net_arc, net_act, sigma)
        # train_img, train_label = loadMNIST_RAM(train_dataloader, repeat_n)
        print('数据准备完成!')
        trainLoss = 0.
        testLoss = 0.
        print('epoch to run:{} learning rate:{}'.format(epoches, learning_rate))
        start = time.time()
        print('模型信息:\narc:{}\nact:{}\nK:{}'.format(net_arc, net_act, repeat_n))
        alg_start = time.time()
        for epoch in range(start_epoch, start_epoch + epoches):
            loss = 0.
            nbatch = 0.
            N = 0.
            n = 0.
            trainLoss = 0.
            train_estimation_relative_error = 0
            for batch, [trainX, trainY] in enumerate(tqdm(train_dataloader, ncols=10)):
                # break

                nbatch += 1
                trainX = trainX.to(device)
                trainY = trainY.to(device)
                trainY = OneHotLabel(trainY, num_classes)
                batch_train_repeatX, batch_train_repeatY = KMeansRepeatX(trainX, repeat_n), KMeansRepeatY(trainY,
                                                                                                          repeat_n)
                pre = net.forward(batch_train_repeatX, train=True, BP=BP_train)

                if not is_01_loss:
                    loss = CELoss(pre, batch_train_repeatY)
                else:
                    loss = ZeroOneLoss(pre, batch_train_repeatY)
                trainLoss += torch.mean(loss).detach().cpu().numpy()
                if BP_train:
                    net.backward(batch_train_repeatY, BP_train)
                    net.update_params(learning_rate, BP_train, method='adam')
                else:
                    net.backward(loss, BP_train)
                    net.update_params(learning_rate, BP_train, method='adagrad')
            trainLoss /= nbatch
            train_loss.append(trainLoss)
            epoch_train_estimation_relative_error.append(train_estimation_relative_error / nbatch)
            # trainAcc = n / N
            print('train epoch:{} loss:{}'.format(epoch, trainLoss))
            if ((epoch + 1) % 10 == 0):
                learning_rate *= 0.8
                print('学习率衰减至{}'.format(learning_rate))
            loss = 0.
            N = 0.
            n = 0.
            nbatch = 0.
            test_estimation_relative_error = 0
            for batch, [testX, testY] in enumerate(tqdm(test_dataloader, ncols=10)):
                nbatch += 1
                testX = testX.to(device)
                testY = testY.to(device)
                testX = KMeansRepeatX(testX, 1, False)
                testY = OneHotLabel(testY, num_classes)

                pre = net.forward(testX, train=False)
                if not is_01_loss:
                    loss = torch.mean(CELoss(pre, testY)).detach().cpu().numpy()
                else:
                    loss = torch.mean(ZeroOneLoss(pre, testY))
                testLoss += loss
                N += len(testX)
                n += evalHot(testY, pre)
            testLoss /= nbatch
            test_loss.append(testLoss)
            testAcc = n / N
            acc.append(testAcc)
            epoch_test_estimation_relative_error.append(test_estimation_relative_error / nbatch)
            print('test epoch:{} loss:{} acc:{}'.format(epoch, testLoss, n / N))
            time_list.append(time.time() - start)
            # net.save_weights(os.path.join('./models_mnist', model_path), epoch, learning_rate)
        net.save_weights('mnist_model/FashionLRTZ.pkl')
        alg_end = time.time()

    print('train_loss:{}\n test_loss:{}\n acc:{}'.format(train_loss, test_loss, acc))
    print('time:{}'.format(time_list))
    with open('FashionLRTZ.pkl'.format(sigma), 'wb') as file:
        pkl.dump(
            [train_loss, test_loss, acc], file)
    print(alg_end-alg_start)
