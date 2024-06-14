---
layout: post
title:  "Data Exploration Using Classification Confidence Scores from a CNN"
date:   2024-05-14 17:22:15 +1100
categories: Computer Vision
---
CNN confidence scores during image classification tell us the probability with which a specific image belongs to a class. Given that these scores are the outputs of a CNN and are generated after the network has fully processed the inputs, it may be possible to use them to further explore the input dataset and the relationship between different classes. Building on this idea, in this post I'll propose and implement a method for clustering CIFAR-10 dataset using confidence scores from a LeNet5 model.  

## CNN Training/Testing
First step is to train the model on the dataset. I'm using PyTorch to define the following LeNet5 model:
```python
class LeNet5(torch.nn.Module):
    def __init__(self, in_features=3, out_classes=10):
        super(LeNet5, self).__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(in_channels=in_features, out_channels=6, kernel_size=5, stride=1),
                                            torch.nn.Tanh(),
                                            torch.nn.MaxPool2d(2,2),
                                            torch.nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5, stride=1),
                                            torch.nn.Tanh(),
                                            torch.nn.MaxPool2d(2,2),
                                            torch.nn.Flatten(),
                                            )
        self.classifier = torch.nn.Sequential(torch.nn.Linear(in_features=16*5*5,out_features=120),
                                             torch.nn.Tanh(),
                                             torch.nn.Linear(in_features=120, out_features=84),
                                             torch.nn.Tanh(),
                                             torch.nn.Linear(in_features=84, out_features=out_classes))
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

Next we need to train and test the model and save the confidence scores. Notice how during testing, the outputs from the last layer of our model, which represent the logits, are passed to the Softmax function to get the confidence scores:
```python
class Classifier:
    def __init__(self, model, dataloaders, device):
        self.model = model
        self.train_loader, self.test_loader = dataloaders
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.config = {
            'lr':1e-3,
            'epochs': 100
        }
        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.config['lr'])
    
    def get_accuracy(self, output, y):
        pred_labels = torch.argmax(output, dim=1)
        return (pred_labels==y).sum().item() / len(y)
    
    def train_step(self):
        running_loss=0.
        running_accuracy=0.
        
        for x,y in self.train_loader:
            self.optim.zero_grad()
            x = x.to(self.device, dtype=torch.float)
            y = y.to(self.device, dtype=torch.long)
            
            output = self.model(x)
            loss = self.loss_fn(output, y)
            
            loss.backward()
            self.optim.step()
            
            running_loss += loss.item()
            running_accuracy += self.get_accuracy(output, y)
            
            del x, y, output
        
        train_loss = running_loss/len(self.train_loader)
        train_accuracy = running_accuracy/len(self.test_loader)
        
        return train_loss, train_accuracy

    def test(self):
        running_loss = 0.
        running_accuracy = 0.
        with torch.no_grad():
            for x,y in self.test_loader:
                x = x.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.long)

                output = self.model(x)
                confidence_scores = torch.nn.functional.softmax(output, dim=1)
                loss = self.loss_fn(output, y)

                running_loss += loss.item()
                running_accuracy = self.get_accuracy(output, y)

                del x, y, output
        
        test_loss = running_loss/len(self.test_loader)
        test_accuracy = running_accuracy/len(self.test_loader)
        
        return test_loss, test_accuracy, confidence_scores
    
    def train(self):
        train_losses,train_accs = [], []
        valid_losses, valid_accs = [], []
        
        for epoch in range(self.config["epochs"]):
            print(f"Model is using {'cuda' if next(self.model.parameters()).is_cuda else 'cpu'}")
            self.model.train()
            
            train_loss, train_acc = self.train_step()
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            self.model.eval()
            
            valid_loss, valid_acc, confidence_scores = self.test()
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            
            print(f"------EPOCH {epoch+1}/{self.config['epochs']}------")
            print(f"Training: LOSS: {train_loss} | ACCURACY: {train_acc}")
            print(f"Validation: LOSS: {valid_loss} | ACCURACY: {valid_acc}\n\n")
            
            # CLEANUP
            gc.collect()
            torch.cuda.empty_cache()
            
        return (train_losses, train_accs), (valid_losses, valid_accs), confidence_scores
```
It's time to train the model and get those confidence scores I've been talking about all along:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LeNet5().to(device)
classifier = Classifier(model, (train_loader, test_loader), device)
classifer.train()
```

Now that we have the confidence scores per test image per class, we need to find a way of using them to measure the similarity/differnece between each class:

Transposing the above matrix gives us the confidence scores per class in each row:

so now each row can be considered as a probability distribution that represents a single class. We need to measure the pairwies distances between these probability distributions to find out how similar/different these classes are to each other. There are different options for doing this, but I chose [Jensen-Shannon divergence](https://towardsdatascience.com/how-to-understand-and-use-jensen-shannon-divergence-b10e11b03fd6) since it is a reliable and efficient algorithm for bigger datasets. Following code implements the above steps:

Okay we're almost there. The last step is to use the divergence matrix to cluster the dataset. There are different clustering algorihtms out there, but I went with the classic k-means since it's easy and requires the user to only set the number of clusters they're looking for (and not any other hyperparameters). Putting the above together, we can now cluster similar classes together: 

Last but not least, to check how the above algorithm performs, let's compare our results to the output of t-SNE on CIFAR-10, which is a standard method for dimensionality reduction, data visualisation. and identifying patterns in a dataset.