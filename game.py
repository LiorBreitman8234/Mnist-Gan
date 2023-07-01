#libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

from PIL import Image,ImageTk
import math
# imports
import os, sys

# third party imports
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler,random_split
import tkinter as tk
from tkinter import messagebox
import random

class CGenerator(nn.Module):
    def __init__(self, size_noise,num_channels=32):
        super(CGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(size_noise, 1024, 2, 1, 0, bias=False),#outputs(1024,2,2) images
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 2, 1, 0, bias=False),#outputs(512,3,3) images
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),#outputs(256,6,6)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),#outputs(128,12,12)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),#outputs(64,24,24)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 5, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        img = self.net(noise)
        return img
    
class PictureGuesser:
    def __init__(self,generator,loader):
        self.loader = loader
        self.gen = generator
        self.attempts = 0
        self.successes = 0
        self.window = tk.Tk()
        self.title_label = tk.Label(self.window, text="Is the picture real or fake?")
        self.title_label.pack()

        self.picture_label = tk.Label(self.window)
        self.picture_label.config(width=100, height=100)
        self.picture_label.pack()

        self.real_button = tk.Button(self.window, text="Real", command=lambda: self.check_guess(True))
        self.real_button.pack()

        self.fake_button = tk.Button(self.window, text="Fake", command=lambda: self.check_guess(False))
        self.fake_button.pack()

        self.attempts_label = tk.Label(self.window, text="Attempts: 0")
        self.attempts_label.pack()

        self.successes_label = tk.Label(self.window, text="Successes: 0")
        self.successes_label.pack()
        self.label = None
        self.generate_random_picture()

    def generate_random_picture(self):
        # Here, you would implement the logic to generate a random picture
        # For simplicity, let's just display a random label: "Real" or "Fake"
        self.label = random.choice(["Real", "Fake"])
        if self.label == "Real":
            image, label = next(iter(self.loader))
            image = transforms.ToPILImage()(image.squeeze())
        else:
            noise = torch.randn(1,128,1,1)
            image = self.gen(noise).squeeze()
            image = transforms.ToPILImage()(image)
        image = image.resize((100, 100))
        photo = ImageTk.PhotoImage(image)
        self.picture_label.config(image=photo)
        self.picture_label.image = photo

    def check_guess(self, guess):
        self.attempts += 1
        self.attempts_label.config(text="Attempts: " + str(self.attempts))

        if (self.label == "Real" and guess) or (self.label == "Fake" and not guess):
            self.successes += 1
            self.successes_label.config(text="Successes: " + str(self.successes))
            messagebox.showinfo("Correct", "Your guess is correct!")
        else:
            messagebox.showinfo("Incorrect", "Your guess is incorrect!")

        self.generate_random_picture()

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,drop_last=True)
    genarator = CGenerator(128)
    genarator = torch.load("mnist_gen.pth",map_location=torch.device("cpu"))
    game = PictureGuesser(genarator,train_loader)
    game.run()
