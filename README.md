# data-augmentation-with-gan-and-vae :100:

[Vincent Fortin](https://github.com/vincentfortin) and I are using the [UTK Faces dataset](http://aicip.eecs.utk.edu/wiki/UTKFace) to for the project in the [_Machine Learning I_](https://www.hec.ca/en/courses/detail/?cours=MATH80629A) project. 

Unbalanced classes is one of the most frequent struggle when dealing with real data. Is it better to down/upsample, or do nothing at all? Another approach is to generate samples resembling the smallest class. In this project, we are using Variational AutoEncoders (VAEs) and Generative Adversarial Networks (GANs) to generate samples of the smallest class. Using human faces, we will determine if a convolutional neural network (CNN) will be trained better with generated samples, or without.  

## PROGRESS
1. [First we trained a VAE](https://github.com/nicolas-gervais/data-augmentation-with-gan-and-vae/blob/master/Variational%20Auto%20Encoder%20on%20Human%20Faces.ipynb) to generate human faces
2. [Then we trained a ConvNet with Pytorch](https://github.com/nicolas-gervais/data-augmentation-with-gan-and-vae/blob/master/Pytorch%20ConvNet%20Distinguishing%20Men%20and%20Women.ipynb) but it didn't work.
3. So we tried with Keras to see if our architecture was the problem. It's not. [We reached 90% accuracy](https://github.com/nicolas-gervais/data-augmentation-with-gan-and-vae/blob/master/Keras%20CNN%20Benchmark.ipynb). 
## TO DO
- [x] Train a Keras Model
- [ ] Create a GAN to generate human faces
    - Here is the [adversarial auto encoder](https://github.com/nicolas-gervais/data-augmentation-with-gan-and-vae/blob/master/Adversarial%20Auto%20Encoder.ipynb)
- [ ] Explore other generative methods
- [ ] Train CNNs to see if the accuracy is better with the generative methods
- [ ] Fix the Pytorch CNN
## PROJECT PLAN
1. Create various sample generators
2. Establish a benchmark CNN classifier, trained with 10% of the female samples (smaller class)
3. Train classifiers on 10% of the female samples, and add generated samples. Finally, compare performance.
    - VAE
    - GAN
    - other
4. Compare performance, plot 
## MISC
- Try random erasing
