import torch
import torch.nn.functional as F
import torch.distributions as distributions
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from scipy import special

#Load data
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,
)
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor()
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Add noise to data
for i in range(len(train_data.data)):
    if (i % 2 == 0):
        train_data.data[i] = torch.randn(*train_data.data[i].shape)
        train_data.targets[i] = np.random.uniform(1,10)

for i in range(len(test_data.data)):
    if (i % 2 == 0):
        test_data.data[i] = torch.randn(*test_data.data[i].shape)
        test_data.targets[i] = np.random.uniform(1,10)

from torch.utils.data import DataLoader
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
}

figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

#CNN Architecture
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

cnn = CNN().to(device)
print(cnn)

loss_func = nn.CrossEntropyLoss()
from torch import optim
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)

#GA Architecture
class GA(torch.nn.Module):
    def __init__(self):#, state_size, action_size, hidden_dims):#, dropout = 0.5):
        super(GA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 2 classes
        self.out = nn.Linear(32 * 7 * 7, 2)


    def create_population(self, pop_size):
        population = []

        for i in range(pop_size):
            filter1 = np.random.uniform(-1, 1, size=(16, 1, 5, 5))
            filter2 = np.random.uniform(-1, 1, size=(32, 16, 5, 5))
            weights3 = np.random.uniform(-1, 1, size=(1568, 2))
            bias1 = np.random.uniform(-1, 1, size=(16))
            bias2 = np.random.uniform(-1, 1, size=(32))
            bias3 = np.random.uniform(-1, 1, size=(2))
            weights = [filter1, bias1, filter2, bias2, weights3, bias3]
            population.append(weights)

        return population

    def set_weights(self, child):
        w1 = torch.FloatTensor(child[0].T)
        b1 = torch.FloatTensor(child[1].T)
        w2 = torch.FloatTensor(child[2].T)
        b2 = torch.FloatTensor(child[3].T)
        w3 = torch.FloatTensor(child[4].T)
        b3 = torch.FloatTensor(child[5].T)

        self.conv1.weight = nn.Parameter(w1.to(device))
        self.conv1.bias = nn.Parameter(b1.to(device))
        self.conv2.weight = nn.Parameter(w2.to(device))
        self.conv2.bias = nn.Parameter(b2.to(device))
        self.out.weight = nn.Parameter(w3.to(device))
        self.out.bias = nn.Parameter(b3.to(device))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)

        #x = self.fc1(x)
        #x = self.dropout(x)
        #x = F.relu(x)
        #x = self.fc2(x)
        return output

    def get_action(self, observations):
        state = torch.tensor(observations).unsqueeze(0)
        action_pred = self.forward(state)
        action_prob = F.softmax(action_pred)
        possible_actions = distributions.Categorical(logits=action_prob)
        action = possible_actions.sample()
        return action

#Reproduce two genomes
def reproduce(x, y):
    flat_filter1_x = x[0].flatten()
    flat_bias1_x = x[1].flatten()
    flat_filter2_x = x[2].flatten()
    flat_bias2_x = x[3].flatten()
    flat_weights3_x = x[4].flatten()
    flat_bias3_x = x[5].flatten()
    
    flat_filter1_y = y[0].flatten()
    flat_bias1_y = y[1].flatten()
    flat_filter2_y = y[2].flatten()
    flat_bias2_y = y[3].flatten()
    flat_weights3_y = y[4].flatten()
    flat_bias3_y = y[5].flatten()
    
    filter1_split = np.random.randint(0, flat_filter1_x.size)
    bias1_split  = np.random.randint(0, flat_bias1_x.size)
    filter2_split = np.random.randint(0, flat_filter2_x.size)
    bias2_split = np.random.randint(0, flat_bias2_x.size)
    weights3_split = np.random.randint(0, flat_weights3_x.size)
    bias3_split = np.random.randint(0, flat_bias3_x.size)
    
    
    new_filter1 = np.concatenate((flat_filter1_x[0:filter1_split], flat_filter1_y[filter1_split:]))
    new_bias1 = np.concatenate((flat_bias1_x[0:bias1_split], flat_bias1_y[bias1_split:]))
    new_filter2 = np.concatenate((flat_filter2_x[0:filter2_split], flat_filter2_y[filter2_split:]))
    new_bias2 = np.concatenate((flat_bias2_x[0:bias2_split], flat_bias2_y[bias2_split:]))
    new_weights3 = np.concatenate((flat_weights3_x[0:weights3_split], flat_weights3_y[weights3_split:]))
    new_bias3 = np.concatenate((flat_bias3_x[0:bias3_split], flat_bias3_y[bias3_split:]))
    
    reshape_f1 = new_filter1.reshape(np.shape(x[0]))
    reshape_b1 = new_bias1.reshape(np.shape(x[1]))
    reshape_f2 = new_filter2.reshape(np.shape(x[2]))
    reshape_b2 = new_bias2.reshape(np.shape(x[3]))
    reshape_w3 = new_weights3.reshape(np.shape(x[4]))
    reshape_b3 = new_bias3.reshape(np.shape(x[5]))
    
    return [reshape_f1, reshape_b1, reshape_f2, reshape_b2, reshape_w3, reshape_b3]
    
#Mutate a genome
def mutate(child):
    flat_filter1 = child[0].flatten()
    flat_bias1 = child[1].flatten()
    flat_filter2 = child[2].flatten()
    flat_bias2 = child[3].flatten()
    flat_weights3 = child[4].flatten()
    flat_bias3 = child[5].flatten()
    
    filter1_mutate = np.random.randint(0, flat_filter1.size)
    bias1_mutate = np.random.randint(0, flat_bias1.size)
    filter2_mutate = np.random.randint(0, flat_filter2.size)
    bias2_mutate= np.random.randint(0, flat_bias2.size)
    weights3_mutate = np.random.randint(0, flat_weights3.size)
    bias3_mutate= np.random.randint(0, flat_bias3.size)
    
    flat_filter1[filter1_mutate] = np.random.randn()
    flat_bias1[bias1_mutate] = np.random.randn()
    flat_filter2[filter2_mutate] = np.random.randn()
    flat_bias2[bias2_mutate] = np.random.randn()
    flat_weights3[weights3_mutate] = np.random.randn()
    flat_bias3[bias3_mutate] = np.random.randn()
    
    reshape_f1 = flat_filter1.reshape(np.shape(child[0]))
    reshape_b1 = flat_bias1.reshape(np.shape(child[1]))
    reshape_f2 = flat_filter2.reshape(np.shape(child[2]))
    reshape_b2 = flat_bias2.reshape(np.shape(child[3]))
    reshape_w3 = flat_weights3.reshape(np.shape(child[4]))
    reshape_b3 = flat_bias3.reshape(np.shape(child[5]))
    
    return [reshape_f1, reshape_b1, reshape_f2, reshape_b2, reshape_w3, reshape_b3]

ga_agent = GA().to(device)
pop_size = 25
population = ga_agent.create_population(pop_size)
mutation_prob = 0.1

from tqdm import tqdm
num_epochs = 100
noise_factor = 1
def train(num_epochs, cnn, ga, population, loaders):

    cnn.train()

    # Train the model
    total_step = len(loaders['train'])

    #keep track of best agent and best score
    best_score = 0
    best_agent = []

    for epoch in range(num_epochs):
        new_population = []
        population_scores = [0]*len(population)
        with tqdm(total=len(loaders['train'])) as pbar:
            for i, (images, labels) in enumerate(loaders['train']):
                correct = 0

                labels_copy = torch.clone(labels)
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(images).to(device)   # batch x
                b_y = Variable(labels).to(device)   # batch y
                output = cnn(b_x.to(device))[0]
                loss = loss_func(output, b_y)

                pred_labels = torch.max(output, 1).indices
                correct += (pred_labels.to(device) == labels.to(device)).sum().item()

                accuracy = correct / len(b_y)

                #population_scores = [0]*len(population)
                #new_population = []
                for k in range(len(population)):
                    rewards = 0
                    correct_2 = 0
                    keep_imgs = []
                    ga.set_weights(population[k])
                    for j in range(len(b_x)):
                        keep = ga.get_action(b_x[j].to(device))
                        keep_imgs.append(keep)
                        if (j != (len(b_x) -1)):
                            rewards += 0

                    new_batch = []
                    new_labels = []
                    for j in range(len(keep_imgs)):
                        if keep_imgs[j] == 1:
                            new_batch.append(torch.FloatTensor(images[j]))
                            new_labels.append(labels_copy[j].item())

                    if (len(new_batch) == 0):
                        accuracy_2 = 0
                    else:
                        t = torch.stack(new_batch)
                        l = new_labels.copy()

                        b_x_2 = Variable(t)
                        b_y_2 = Variable(torch.Tensor(l))

                        #get output again based on those images
                        output_2 = cnn(b_x_2.to(device))[0]

                        #compare the accuracy between those outputs
                        pred_labels_2 = torch.max(output_2, 1).indices
                        #print(pred_labels_2)
                        #print(torch.Tensor(new_labels))
                        correct_2 += (pred_labels_2.to(device) == torch.Tensor(new_labels).to(device)).sum().item()

                        accuracy_2 = correct_2 / len(b_y_2)
                    rewards = 100*(accuracy_2 - accuracy)
                    population_scores[k] += rewards

                    ''' if (rewards > best_score):
                        best_agent = population[k]
                        best_score = rewards'''

                # clear gradients for this training step
                optimizer.zero_grad()

                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                optimizer.step()

                pbar.update()

            dist = special.softmax(population_scores)
                #print(dist)

            for n in range(len(population) - 2):
                #choose x based on rand fitness
                x  = np.random.choice(np.arange(0, pop_size), p=dist)

               #choose y based on rand fitness
                y  = np.random.choice(np.arange(0, pop_size), p=dist)

                child = reproduce(population[x], population[y])
                    
                if (np.random.uniform() < mutation_prob):#small random probability
                    child = mutate(child)
                child = mutate(population[x])
                new_population.append(child)

            #sort_score = population_scores
            ind = np.argpartition(population_scores, -2)[-2:]
            best1 = ind[0]
            best2 = ind[1]
            if population_scores[best1] > best_score:
                best_score = population_scores[best1]/600
                best_agent = population[best1]

            #best = np.argmax(population_scores)
            new_population.append(population[best1])
            new_population.append(population[best2])

            if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, Population average: {:.4f}. Best score: {:.4f}.'
                           .format(epoch + 1, num_epochs, loss.item(), accuracy, (sum(population_scores)/600)/len(population_scores), best_score))

            population = new_population
            
    return population, best_agent

population, best = train(num_epochs, cnn, ga_agent, population, loaders)

def test():
    # Test the model
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images.to(device))
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y.to(device) == labels.to(device)).sum().item() / float(labels.size(0))

    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)

test()

def test_noisy_with_agent(ga, best):
    # Test the model
    cnn.eval()
    best = 0
    for k in range(len(population)):
        with torch.no_grad():
            total = 0
            acc = 0
            count = 0
            for images, labels in loaders['test']:
                count += 1
                correct_2 = 0
                labels_copy = torch.clone(labels)
                b_x = Variable(images).to(device)   # batch x
                b_y = Variable(labels).to(device)   # batch y

                keep_imgs = []
                ga.set_weights(population[k])
                for j in range(len(b_x)):
                    keep = ga.get_action(b_x[j].to(device))
                    keep_imgs.append(keep)

                new_batch = []
                new_labels = []
                for j in range(len(keep_imgs)):
                    if keep_imgs[j] == 1:
                        new_batch.append(torch.FloatTensor(images[j]))
                        new_labels.append(labels_copy[j].item())


                if (len(new_batch) == 0):
                    accuracy_2 = 0
                else:
                    t = torch.stack(new_batch)
                    l = new_labels.copy()

                    b_x_2 = Variable(t)
                    b_y_2 = Variable(torch.Tensor(l))

                    #get output again based on those images
                    output_2 = cnn(b_x_2.to(device))[0]

                    #compare the accuracy between those outputs
                    pred_labels_2 = torch.max(output_2, 1).indices
                    correct_2 += (pred_labels_2.to(device) == torch.Tensor(new_labels).to(device)).sum().item()

                    accuracy_2 = correct_2 / len(b_y_2)

                acc += accuracy_2
            if acc > best:
                best = acc

    print('Test Accuracy of the model on the 10000 test images: %.2f' % (best/count))
    return new_batch, new_labels

new_images, new_labels = test_noisy_with_agent(ga_agent, best)
