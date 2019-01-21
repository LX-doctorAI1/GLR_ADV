# Training artificial neural networks by generalized likelihood ratio method:exploring brain-like learning to improve adversarial defensiveness

### Introduction
Recent work in deep learning has shown that the artificial neural networks  are vulnerable to adversarial attacks, where a very small perturbation of the inputs can drastically alter the classification result. In this work, we propose a generalized likelihood ratio method capable of training the artificial neural networks with some biological brain-like mechanisms,.e.g., (a) learning by the loss value, (b) learning via neurons with discontinuous activation and loss functions. The traditional back propagation method cannot train the artificial neural networks with aforementioned brain-like learning mechanisms. Numerical results show that various artificial neural networks trained by the new method can significantly improve the defensiveness against the adversarial attacks. Code is available: \url{https://github.com/LX-doctorAI/GLR_ADV}.

### Citation

If you find generalized likelihood ratio method useful in your research, please consider citing:

    @article{ren15fasterrcnn,
        Author = {Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun},
        Title = {{Faster R-CNN}: Towards Real-Time Object Detection with Region Proposal Networks},
        Journal = {arXiv preprint arXiv:1506.01497},
        Year = {2015}
    }
    
### Requirements: software

The code is developed based on python 3.7.1

### Training with generalized likelihood method:
./Training_by_GLR_and_BP/GLR/Training_by_GLR_with_threshold.py: Training with GLR method using cross entropy loss and threshold function
./Training_by_GLR_and_BP/GLR/Training_by_GLR_with_0_1_loss_and_threshold.py: Training with GLR method using 0-1 loss and threshold function

### Generate Adversarial Samples:
./Generate_adversarial_samples/Training_by_BP.py:Training using BP for MLP with two hidden layers
./Generate_adversarial_samples/Generate_5k_samples.py:randomly sampled 5k sample images
./Generate_adversarial_samples/Generate_adversarial_sample_by_FGSM.py: generate adversarial samples using FGSM
./Generate_adversarial_samples/Visualize_adversarial_samples.py: visualize adversarial samples

### Test Adversarial effect:
./Test_with_adversarial_samples/BP/Test_accuracy=0.99-0.28.py: using MLP trained with BP for testing, with an accuracy of 0.99 on the original samples, and an accuracy of 0.28 on the adversarial samples
./Test_with_adversarial_samples/Threshold/Test_accuracy=0.95-0.52.py:using the same structure trained with GLR for testing, with an accuracy of 0.95 on the original samples, and an accuracy of 0.52 on the adversarial samples
./Test_with_adversarial_samples/Threshold_with_0_1_loss/Test_accuracy=0.95-0.58.py:using the same structure trained with GLR for testing, with an accuracy of 0.86 on the original samples, and an accuracy of 0.58 on the adversarial samples



