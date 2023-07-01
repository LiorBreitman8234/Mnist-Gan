# Gan For Mnist
- In this repository, I created a GAN framework for the Mnist data set.
- I create it using pytorch
- For the generator, I used the Conv transpose layers
- For the Discriminator, I used a very simple Convolutional net.
- I Trained multiple times, and, at the end I saw that the best amount of epochs s.t. the generator is good enough but doesn't collapse is around 50 epochs
![How Fake Images Look](pics/fake.png)