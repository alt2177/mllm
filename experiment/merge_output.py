import torch 
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prob_0 = torch.from_numpy(torch.load("gpt2_f_experiment_0_train.pt"))
prob_1 = torch.from_numpy(torch.load("gpt2_f_experiment_1_train.pt"))
prob_2 = torch.from_numpy(torch.load("gpt2_f_experiment_2_train.pt"))
prob_3 = torch.from_numpy(torch.load("gpt2_f_experiment_3_train.pt"))
prob_4 = torch.from_numpy(torch.load("gpt2_f_experiment_4_train.pt"))
prob = torch.cat([prob_0, prob_1, prob_2, prob_3, prob_4],dim=1).to(device)

print([prob_0, prob_1, prob_2, prob_3, prob_4])

eval_prob_0 = torch.from_numpy(torch.load("gpt2_f_experiment_0_test.pt"))
eval_prob_1 = torch.from_numpy(torch.load("gpt2_f_experiment_1_test.pt"))
eval_prob_2 = torch.from_numpy(torch.load("gpt2_f_experiment_2_test.pt"))
eval_prob_3 = torch.from_numpy(torch.load("gpt2_f_experiment_3_test.pt"))
eval_prob_4 = torch.from_numpy(torch.load("gpt2_f_experiment_4_test.pt"))
eval_prob = torch.cat([eval_prob_0, eval_prob_1, eval_prob_2, eval_prob_3, eval_prob_4],dim=1).to(device)

dataset = load_dataset("yelp_review_full")
eval_dataset = dataset["train"]
labels = eval_dataset['label']

test_dataset = dataset["test"]
test_labels = test_dataset['label']

labels = torch.tensor(labels,dtype=torch.long).to(device)
test_labels = torch.tensor(test_labels,dtype=torch.long).to(device)

class NeuralNetwork(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(NeuralNetwork,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim,50),
            nn.Tanh(),
	    nn.Linear(50, 40),
	    nn.Tanh(),
            nn.Linear(40,20),
            nn.Tanh(),
	    nn.Linear(20, 10),
	    nn.Tanh(),
            nn.Linear(10,output_dim)
                    )
    def forward(self,x):
        return self.layer(x)

model = NeuralNetwork(input_dim=25,hidden_dim=20,output_dim=5).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

def compute_accuracy(predictions,labels):
    pred_labels = np.argmax(predictions,axis=1)
    accuracy = (pred_labels == labels).mean()
    return accuracy

num_epochs=100

for epoch in range(num_epochs):
    model.train()
    predictions = model(prob)
    loss = loss_fn(predictions,labels)
    predictions_cpu = predictions.detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()
    accuracy = compute_accuracy(predictions_cpu,labels_cpu)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
    # Evaluation
    model.eval()
    eval_predictions = model(eval_prob)
    eval_loss = loss_fn(eval_predictions, test_labels)
    eval_predictions_cpu = eval_predictions.detach().cpu().numpy()
    eval_labels_cpu = test_labels.detach().cpu().numpy()
    eval_accuracy = compute_accuracy(eval_predictions_cpu, eval_labels_cpu)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Accuracy: {accuracy.item():.4f}, Eval Loss: {eval_loss.item():.4f}, Eval Accuracy: {eval_accuracy.item():.4f}')



avg_probs = (eval_prob_0 + eval_prob_1 + eval_prob_2 + eval_prob_3 + eval_prob_4) / 5
pred_labels = torch.argmax(avg_probs, dim=1).to(device)
correct_predictions = (pred_labels == test_labels).sum().item()
total_predictions = test_labels.size(0)
accuracy = correct_predictions / total_predictions

print(f'Accuracy: {accuracy:.4f}')
