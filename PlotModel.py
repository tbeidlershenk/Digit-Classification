import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 17}
plt.rc('font', **font)
batches = []
accuracies = []
bars = None
plot_freq = 1

def create_graph(batch_size):
    global bars
    plt.title("Loss / Batch")
    bars = plt.plot(batches, accuracies, linewidth=2.0)
    #plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()]) 
    plt.draw()
    plt.show(block=False)
    plt.pause(0.001)

def update_graph(batch, value):
    global bars
    if (batch % plot_freq == 0):
        batches.append(batch)
        accuracies.append(value)
        bars = plt.plot(batches, accuracies, linewidth=2.0)
        #plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()]) 
        plt.draw()
        plt.pause(0.001)

def input_image(model, path):
    image_src = Image.open(path).convert('L')
    image_data = 255-np.asarray(image_src)
    image_data_reshaped = np.reshape(image_data,784)
    guess=model.predict(torch.tensor(image_data_reshaped, dtype=torch.float32))
    print(guess)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = plt.imshow(image_data,cmap='Greys')
    ax.set_title('Model guessed:'+str(guess.item()))
    plt.show()