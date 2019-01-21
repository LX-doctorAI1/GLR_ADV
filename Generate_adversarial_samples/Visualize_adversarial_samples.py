import numpy as np
import matplotlib.pyplot as plt 
import pickle 
import sys, os 

# pkl_loc = sys.argv[1]
with open("adversarial_samples_with_FGSM.pkl", "rb") as f:
    adv_data_dict = pickle.load(f) 

xs = adv_data_dict["xs"]
y_trues = adv_data_dict["y_trues"]
y_preds = adv_data_dict["y_preds"]
noises  = adv_data_dict["noises"]
y_preds_adversarial = adv_data_dict["y_preds_adversarial"]  
# print(len(xs)) #1750

# visualize randomly some of adversarial samples
idxs = np.random.choice(range(50), size=(20,), replace=False) #生成包含20个不重复数值的数组,[0,50)
for matidx, idx in enumerate(idxs):
    orig_im = xs[idx].reshape(14,14)
    adv_im  = orig_im + noises[idx].reshape(14,14)
    disp_im = np.concatenate((orig_im, adv_im), axis=1)
    plt.subplot(5,4,matidx+1)
    plt.imshow(disp_im, "gray")
    plt.xticks([])
    plt.yticks([])
# plt.savefig("initial_samples.png")
plt.show()

# visualize randomly some of initial samples
# idxs = np.random.choice(range(50), size=(10,), replace=False) #生成包含20个不重复数值的数组,[0,50)
# # print(idxs)
# for matidx, idx in enumerate(idxs):
#     orig_im = xs[idx].reshape(14,14)
#     # adv_im  = orig_im + noises[idx].reshape(14,14)
#     # disp_im = np.concatenate((orig_im, adv_im), axis=1)
#     plt.subplot(1,10,matidx+1)
#     plt.imshow(orig_im, "gray")
#     plt.xticks([])
#     plt.yticks([])
# # plt.savefig("adversarial_samples.png")
# plt.show()























