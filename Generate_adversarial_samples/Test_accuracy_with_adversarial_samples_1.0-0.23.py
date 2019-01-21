import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms
from tqdm import *
from skimage import transform, data
from sklearn import metrics
import pickle

# define nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # my network is composed of only affine layers
        self.fc1 = nn.Linear(14 * 14, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        # self.r   = nn.Parameter(data=torch.randn(5,5), requires_grad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# DATA LOADERS
def flat_trans(x):
    x.resize_(28 * 28) #图片缩放
    return x

if __name__ == '__main__':

    net = Net()

    # Load pre-trained weights
    weights_dict = {}
    with open("weights_accuracy=0.95.pkl", "rb") as f:
        weights_dict = pickle.load(f)
    for param in net.named_parameters():
        if param[0] in weights_dict.keys():
            # print("Copying: ", param[0])
            param[1].data = weights_dict[param[0]].data
    print("Weights Loaded!")

    # load the adversarial samples
    adv_dict = {}
    with open("adversarial_samples_with_FGSM.pkl", "rb") as f:
        adv_dict = pickle.load(f)
    xs = adv_dict["xs"]
    y_trues = adv_dict["y_trues"]
    y_preds = adv_dict["y_preds"]
    noises = adv_dict["noises"]
    y_preds_adversarial = adv_dict["y_preds_adversarial"]

    print("===== xs.length : {} =====".format(len(xs)))

    # get the adversarial samples
    xs = np.array(xs).reshape(len(xs), 196)
    noises = np.array(noises).reshape(len(xs), 196)
    y_trues = np.array(y_trues).reshape(len(xs))
    advs = xs + noises

    # TEST the initial_x
    init_images, init_labels = xs, y_trues
    init_images = Variable(torch.FloatTensor(init_images), requires_grad=False)
    init_outputs = net(init_images)
    init_y_pred = np.argmax(init_outputs.data.numpy(), axis=1)
    init_test_accuracy = metrics.accuracy_score(y_trues, init_y_pred)
    print('The accuracy on the initial samples is {:.2f}'.format(init_test_accuracy))  # 1.0

    # TEST the adv_x
    adv_images, adv_labels = advs, y_trues
    adv_images = Variable(torch.FloatTensor(adv_images), requires_grad=False)
    adv_outputs = net(adv_images)
    adv_y_pred = np.argmax(adv_outputs.data.numpy(), axis=1)
    adv_test_accuracy = metrics.accuracy_score(y_trues, adv_y_pred)
    print('The accuracy on the adversarial_samples_m1 is {:.2f}'.format(adv_test_accuracy))  #0.23










