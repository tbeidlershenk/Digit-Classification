from ConvNeuralNetwork import ConvNeuralNetwork
from mnist import MNIST
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import os
from PIL import Image
from PlotModel import create_graph, update_graph

mnist_data = MNIST('data')

def train(epochs, batch_size, alpha):
    
    training_images, training_labels = mnist_data.load_training()
    train_i = np.reshape(training_images, (60000, 1, 28, 28))

    model = ConvNeuralNetwork()
    loss = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=alpha)
    num_batches = int(60000/batch_size)
    create_graph(batch_size)
    
    for e in range(epochs):
        for b in range(num_batches):          
            first = batch_size * b
            last = first + batch_size
            # create 4d tensor of batch_size images
            image_batch = torch.tensor(train_i[first:last,:,:,:], dtype=torch.float32)
            label_batch = torch.tensor(training_labels[first:last], dtype=torch.long)
            # forward pass
            outputs = model(image_batch)
            # calculate loss
            curr_loss = loss(outputs, label_batch)
            # clear gradients
            optimizer.zero_grad()
            # backward pass
            curr_loss.backward()
            # update parameters
            optimizer.step()
            update_graph((e*num_batches)+b, curr_loss.item())

            print("Epoch " + str(e) + ", Batch " + str(b))

    return model

def test(model):
    test_images, test_labels = mnist_data.load_testing()
    test_i = np.reshape(test_images, (10000, 1, 28, 28))

    count = 0.0
    for i in range(len(test_i)):
        # get the image
        tns = torch.tensor(test_i[i:i+1,:,:,:], dtype=torch.float32)
        # get prediction and actual
        pred = model.predict(tns).item()
        actual = test_labels[i]
        
        print("Guess = " + str(pred) + ", Actual = " + str(actual))
        if pred == actual:
            count += 1
    pct = count/10000
    print("Accuracy: " + str(pct))

def test_sketches(model):
    rel_path = "data/sketches/images/"
    files = os.listdir(rel_path)
    print(files)
    numCorrect = 0
    for img in files:
        # Extract label from filename
        img = rel_path + img
        label = int(img.split('_')[1][1])
        image_src = Image.open(img).convert('L')
        image_data = 255-np.asarray(image_src)
        image_data_reshaped = np.reshape(image_data,(1,1,28,28))
        guess = model.predict(torch.tensor(image_data_reshaped, dtype=torch.float32))
        if (guess == label): numCorrect += 1
        print("Guess = " + str(guess) + ", Actual = " + str(label))
    pct = float(numCorrect)/len(files)
    print("Sketch Accuracy: " + str(pct))


def main():
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    alpha = float(sys.argv[3])
    model = train(epochs, batch_size, alpha)
    test(model)
    test_sketches(model)
    if ("Y" == input("Save model? Y/N: ")):
        time = str(datetime.now())
        print("Model saved to data/models/ as saved_model-"+time+".pt")
        torch.save(model,'data/models/saved_model-'+time+'.pt') 

if __name__ == '__main__':
    main()
