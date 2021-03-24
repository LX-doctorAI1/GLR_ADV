# Training Artificial Neural Networks by Generalized Likelihood Ratio Method: Exploring Brain-like Learning to Improve Robustness

### Introduction
In this work, we propose a generalized likelihood ratio method capable of training the artificial neural networks with some biological brain-like mechanisms,.e.g., (a) learning by the loss value, (b) learning via neurons with discontinuous activation and loss functions. The traditional back propagation method cannot train the artificial neural networks with aforementioned brain-like learning mechanisms. Numerical results show that the robustness of various artificial neural networks trained by the new method is significantly improved when the input data is affected by both the natural noises and adversarial attacks.

### Citation

If you find generalized likelihood ratio method useful in your research, please consider citing:

    @article{peng2019stochastic,
        Author = {Yijie Peng, Li Xiao, Bernd Heidergott,Jeff L. Hong, Henry Lam},
        Title = {Stochastic Gradient Estimation for Artificial Neural Networks},
        Journal = {Preprint with DOI: 10.2139/ssrn.3318847},
        Year = {2019}
    }
    
      @article{Li2019brain-like,
        Author = {Li Xiao, Yijie Peng,Jeff L. Hong, Zewu Ke},
        Title = {Training Artificial Neural Networks by Generalized Likelihood Ratio Method: Exploring Brain-like Learning to Improve Robustness},
        Journal = {CoRR},
        volume   = {abs/1902.00358},
        Year = {2019}
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
