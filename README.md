# Generating Images using Deep Convolutional GANs - DCGANs

## 1. Motivation
Neural Networks have been able to classify and predict success for a variety of tasks but when
it comes to *generating objects* that do not exist or imagination, vanilla neural networks do not have that great an ability. 


**Generative models** like _GANs_ - _Generative Adversarial Networks_ can actually generate new images by learning the distribution of the dataset used to train. In other words, they have imagination and therefore, are ideal for this project.


## 2. Methodology

###c) Gathering data: ​
We will be using public data - starting from the CIFAR dataset, we will look through this dataset and probably others for a source of unsupervised learning, where in the algorithm will learn the distribution of the data and learn to generate similar but new images.

###b) Processing: ​
Standard normalisation for better accuracy. Batch normalisation will be
applied to reduce over-fitting. Model that will be best for this application: ​A generative model - in our case the GAN, will be ideal for this.

####    Architecture guidelines for stable DCGANs - 
* Use batchnorm in both the generator and the discriminator.
* Use ReLU activation in generator for all layers except for the output.
*	For the output, we'll be using a Tanh activation fucntion
* Use LeakyReLU activation in the discriminator for all layers.


## 3. Evaluation: ​ 
We will minimise two losses i.e the Generator’s and the Discriminators - They will be trained together and will improve each others’ accuracy. When the predictions of the discriminator display close to a 50-50 chance of real/fake, that will be the goal for this project.
