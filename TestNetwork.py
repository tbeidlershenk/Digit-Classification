from NeuralNetwork import NeuralNetwork
import sys
from mnist import MNIST
import numpy as np
import torch
from torch import nn
from datetime import datetime
import PlotModel as pm
import matplotlib.pyplot as plt

def main():
    mnist_data = MNIST('data')

    # python3 TestNetwork.py train epochs batch_size alpha
    if sys.argv[1] == 'train':
        model = train(mnist_data, int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
        accuracy = test(mnist_data, model)
        print("Accuracy =",accuracy)
        if (input("Save model? Y/N ")=="Y"):
            torch.save(model,'data/models/saved_model-'+str(datetime.now())+'.pt') 
    elif sys.argv[1] != None:
        model = torch.load(sys.argv[2])
            
        # python3 TestNetwork.py load modelfile.pt
        if sys.argv[1] == 'load':
            accuracy = test(mnist_data, model)
            print("Accuracy =",accuracy)
        
        # python3 TestNetwork.py input modelfile.pt imagefile.bmp
        elif sys.argv[1] == 'input':
            path = sys.argv[3]
            pm.input_image(model, path)
        
    else:
        sys.exit(1)

def train(mnist_data, epochs, batch_size, alpha):
    
    training_images, training_labels = mnist_data.load_training()
    train_images = torch.tensor(training_images, dtype=torch.float32)
    train_labels = torch.tensor(training_labels, dtype=torch.long)
    train_size = len(training_images)

    layers = np.array([784, 200, 100, 10])
    size = len(layers)
    hidden = layers[1:size-1]
    batches = int(float(train_size)/batch_size)

    model = NeuralNetwork(layers[0], hidden, layers[size-1])
    loss = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=alpha)

    pm.create_graph(batch_size)
    for e in range (epochs):
        model = run_epoch(e, model, loss, optimizer, batches, batch_size, train_images, train_labels)
    
    return model

def test(mnist_data, model):
    test_images, test_labels = mnist_data.load_testing()
    test_images = torch.tensor(test_images, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)  
    return run_testset(model, test_images, test_labels)

def run_epoch(e, model, loss, optimizer, batches, batch_size, train_images, train_labels):
    ind = 0
    for b in range(batches):
        print("Epoch = " + str(e) + ", Batch = " + str(b))
        batchLoss = 0
        corr = 0.0
        # Run one batch
        for x in range(batch_size):
            image = train_images[ind]
            label = train_labels[ind]
            # Feed Forward
            output = model.forward(image)
            batchLoss += loss(output,label)
            ind += 1
            if (model.predict(image)==label):
                corr += 1
        
        # Clear gradients
        optimizer.zero_grad()
        # Backpropagate
        batchLoss.backward()
        # Optimizes based on gradient descent
        optimizer.step()
        
        # Add data to graph
        pct = corr/batch_size
        num = e * batches + b
        print(num)
        pm.update_graph(num, pct)

    return model

def run_testset(model, test_images, test_labels):
    
    corr=0.0
    test_size = len(test_images)

    for x in range(test_size):
        
        image = test_images[x]
        label = test_labels[x]
        pred = model.predict(image)
        
        if (pred==label):
            corr += 1
        else:
            print('incorrectly guessed')
        
        print("Guess="+str(pred) + " and Actual=" +str(label))
    
    accuracy = str(float(corr)/test_size)
    return accuracy

if __name__ == '__main__':
    main()

