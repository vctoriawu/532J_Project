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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

#Add noise to dataset
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

#CNN architecture
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

#GA architecture
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
            weights = np.array([filter1, bias1, filter2, bias2, weights3, bias3]) #make numpy
            population.append(weights)
        
        return population
    
    def set_weights(self, child):
        w1 = torch.FloatTensor(child[0].T)
        b1 = torch.FloatTensor(child[1].T)
        w2 = torch.FloatTensor(child[2].T)
        b2 = torch.FloatTensor(child[3].T)
        w3 = torch.FloatTensor(child[4].T)
        b3 = torch.FloatTensor(child[5].T)

        self.conv1.weight = nn.Parameter(w1.to(device))#.to(device)
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
        action_prob = F.softmax(action_pred, dim=1)
        possible_actions = distributions.Categorical(logits=action_prob)
        action = possible_actions.sample()
        return action

#Reproduce two parents    
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

from sklearn.neighbors import NearestNeighbors
#KNN code taken from
#https://github.com/BY571/GARNE-Genetic-Algorithm-with-Recurrent-Network-and-Novelty-Exploration/blob/55e6b88e549c2aff7117f4f2c556f9d290a391a1/GA_Addons/novelty.py#L6

def get_kNN(archive, bc, n_neighbors):
    """
    Searches and samples the K-nearest-neighbors from the archive and a new behavior characterization
    returns the summed distance between input behavior characterization and the bc in the archive
    
    """
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    archive = np.array(archive)
    neigh.fit(archive)
    distances, idx = neigh.kneighbors(X = bc.reshape(1,-1), n_neighbors=n_neighbors)

    return sum(distances.squeeze(0))

def add_bc_to_archive(bc_storage, archive, archive_prob):
    """
    For each behavior characterization in the storage it gets added to the archive by a given probability
    bc_storage = list of bc from the current population
    Probability = ARCHIVE_PROB
    
    """
    for bc in bc_storage:
        if np.random.random() <= archive_prob:
            archive.append(bc)
        
    return archive


ga_agent = GA().to(device)
pop_size = 25
population = ga_agent.create_population(pop_size)
mutation_prob = 0.1
k_neighbors = 5

#Flatten a genome - helper function for other GA operations
def flatten_child(child):
    flat_filter1 = child[0].flatten()
    flat_bias1 = child[1].flatten()
    flat_filter2 = child[2].flatten()
    flat_bias2 = child[3].flatten()
    flat_weights3 = child[4].flatten()
    flat_bias3 = child[5].flatten()
    return np.concatenate((flat_filter1, flat_bias1, flat_filter2, flat_bias2, flat_weights3, flat_bias3))

import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

num_epochs =25
noise_factor = 1
def train(num_epochs, cnn, ga, population, loaders):
    
    cnn.train()
    
    #keep track of best agent and best score
    best_score = 0
    best_agent = []
    
    #store behaviour characteristics
    bc_storage = []
    archive = []
    
    for gene in population:
        archive.append(flatten_child(gene))
        
    # Train the model
    total_step = len(loaders['train'])
    
    for epoch in range(num_epochs):
        S = np.minimum(k_neighbors, len(archive))
        population_rewards = []
        new_population = []
        population_scores = [0]*len(population)
        for k in range(len(population)):
            rewards = 0
            ga.set_weights(population[k])

            #calculate novelty
            bc = flatten_child(population[k])
            distance = get_kNN(archive=archive, bc=bc, n_neighbors=S)
            novelty_ = distance / S
            # calc new reward _weighted reward_novelty
            reward = novelty_
            population_rewards.append(reward)

        with tqdm(total=len(loaders['train'])) as pbar:
            for i, (images, labels) in enumerate(loaders['train']):
                correct = 0

                labels_copy = torch.clone(labels)
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(images)   # batch x
                b_y = Variable(labels)   # batch y
                output = cnn(b_x.to(device))[0]
                loss = loss_func(output, b_y.to(device))

                pred_labels = torch.max(output, 1).indices
                correct += (pred_labels.to(device) == labels.to(device)).sum().item()

                accuracy = correct / len(b_y)

                for k in range(len(population)):
                    correct_2 = 0
                    score = 0
                    keep_imgs = []
                    ga.set_weights(population[k])


                    for j in range(len(b_x)):
                        keep = ga.get_action(b_x[j].to(device))
                        keep_imgs.append(keep)
                        if (j != (len(b_x) -1)):
                            score += 0

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

                optimizer.zero_grad()

                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                optimizer.step()
   
                pbar.update()

            dist = special.softmax(population_rewards)

            for n in range(len(population) - 2):
                #choose x based on rand fitness
                x  = np.random.choice(np.arange(0, pop_size), p=dist)

               #choose y based on rand fitness
                y  = np.random.choice(np.arange(0, pop_size), p=dist)

                child = reproduce(population[x], population[y])
                #print(np.shape(child))

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

            #add bc to archive
            archive = add_bc_to_archive(bc_storage=bc_storage, archive=archive, archive_prob=1)

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

def test_noisy_with_agent(ga, population):
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
            if (acc) > best:
                best = acc

    print('Test Accuracy of the model on the 10000 test images: %.2f' % (best/count))
    return new_batch, new_labels
        
new_images, new_labels = test_noisy_with_agent(ga_agent, population)
