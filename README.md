# A New Likelihood Ratio Method for Training Artificial Neural Networks

### Introduction
We investigate a new approach to compute the gradients of artificial neural networks (ANNs), based on the so-called push-out likelihood ratio method. Unlike the widely used backpropagation (BP) method that requires continuity of the loss function and the activation function, our approach bypasses this requirement by injecting artificial noises into the signals passed along the neurons. We show how this approach has a similar computational complexity as BP, and moreover is more advantageous in terms of removing the backward recursion and eliciting transparent formulas. We also formalize the connection between BP, a pivotal technique for training ANNs, and infinitesimal perturbation analysis, a classic path-wise derivative estimation approach, so that both our new proposed methods and BP can be better understood in the context of stochastic gradient estimation. Our approach allows efficient training for ANNs with more flexibility on the loss and activation functions, and shows empirical improvements on the robustness of ANNs under adversarial attacks and corruptions of natural noises.

### Citation

If you find generalized likelihood ratio method useful in your research, please consider citing:

    @article{peng2019stochastic,
        Author = {Yijie Peng, Li Xiao, Bernd Heidergott,Jeff L. Hong, Henry Lam},
        Title = {Stochastic Gradient Estimation for Artificial Neural Networks},
        Journal = {Preprint with DOI: 10.2139/ssrn.3318847},
        Year = {2019}
    }
    
      @article{Li2019brain-like,
        Author = {Li Xiao, Yijie Peng,Jeff L. Hong, Zewu Ke,Shuhuai Yang},
        Title = {Training Artificial Neural Networks by Generalized Likelihood Ratio Method: Exploring Brain-like Learning to Improve Robustness},
        Journal = {IEEE 16th International Conference on Automation Science and Engineering (CASE)},
        Year = {2020}
    } 
 
### Requirements: software

The code is developed based on python 3.7.4

### Dataset Used:

MNIST: can be downloaded online by pytorch  (torchvision.datasets.MNIST)

Fashion-MNIST:can be downloaded online by pytorch (torchvision.datasets.FashionMNIST)     

Tiny-ImageNet: can be downloaded from url: http://cs231n.stanford.edu/tiny-imagenet-200.zip  

### Training with generalized likelihood method:

Dir: Train&Test

./Train&Test/LRS.py: Training with GLR method using cross entropy loss and sigmoid function

./Train&Test/LRSZ.py: Training with GLR method using 0-1 loss and sigmoid function

./Train&Test/LRT.py: Training with GLR method using cross entropy loss and threhold function

./Train&Test/LRTZ.py: Training with GLR method using 0-1 loss and threhold function




### Test Adversarial effect:

Dir: RobustnessTest

./RobustnessTest/generate_adv.py: generate adversarial samples using FGSM method

./RobustnessTest/noise_generate_adv.py:generate noise corrupted images

./RobustnessTest/testAdv.py:evaluate the robustness of the models on noise samples


Please contact Li Xiao(xiaoli@ict.ac.cn) for any problem about the code.
