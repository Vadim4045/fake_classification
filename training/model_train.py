import os
import csv
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

styles = ['Fabrication', 'Advertising', 'Manipulation', 'Propaganda', 'Satire', 'Parody']
directory = './raw_models/'
model_save_pattern = 'Roberta_1024_512_6'
stats_file_path = f"{directory}training_stats.csv"

TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-06
previous_epochs = 12

device = torch.device('cuda')

class CustomRobertaForSequenceClassification(nn.Module):
    def __init__(self, num_labels=6):
        super(CustomRobertaForSequenceClassification, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_labels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask).logits
        x = self.leaky_relu(outputs)
        x = self.leaky_relu(self.fc1(x))
        return self.fc2(x)

    def save_weights(self, path):
      weights = {
          'bert': self.roberta.state_dict(),
          'fc1': self.fc1.state_dict(),
          'fc2': self.fc2.state_dict()
      }
      torch.save(weights, path)

    def load_weights(self, path):
        weights = torch.load(path, map_location=torch.device('cpu'))
        self.roberta.load_state_dict(weights['bert'])
        self.fc1.load_state_dict(weights['fc1'])
        self.fc2.load_state_dict(weights['fc2'])


model = CustomRobertaForSequenceClassification(num_labels=6)
model.to(device)

if previous_epochs > 0:
    model_save_path_pattern = f"{directory}{model_save_pattern}_{previous_epochs:03d}_*.pth"
    matching_files = sorted(glob.glob(model_save_path_pattern), reverse=True)

    if matching_files:
        latest_file = matching_files[0]
        print(f"Found matching file: {latest_file}")
        model.load_weights(latest_file)
    else:
        print("No matching file found.")
        exit()
else:
    print("Starting training from scratch.")

train_dataset = torch.load(f'{directory}roberta_large_6_classes_204167_train.pt')
test_dataset = torch.load(f'{directory}roberta_large_6_classes_51042_test.pt')

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {previous_epochs + epoch + 1}/{previous_epochs + EPOCHS}', leave=False)
    for batch in progress_bar:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})
    avg_train_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_train_loss:.4f}")

    model.eval()
    total_accuracy_score = 0
    total_score = 0
    total_test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f'Epoch {previous_epochs + epoch + 1}/{previous_epochs + EPOCHS}', leave=False)
        for batch in progress_bar:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            labels = torch.argmax(labels, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct = (preds == labels).sum().item()
            total_accuracy_score += correct
            total_score += labels.size(0)

            test_loss = loss_fn(logits, labels)
            total_test_loss += test_loss.item()

            progress_bar.set_postfix({'testing_loss': '{:.3f}'.format(test_loss.item())})

        avg_test_loss = total_test_loss / len(test_loader)
        accuracy = total_accuracy_score / total_score
        print(f"Average test loss: {avg_test_loss:.4f}")
        print(f"Accuracy: {accuracy:.3f}")

    model_save_path = (f"{directory}{model_save_pattern}_{(previous_epochs + epoch + 1):03d}_"
                       f"{avg_train_loss:.3f}_{avg_test_loss:.3f}_{accuracy:.3f}.pth")
    model.save_weights(model_save_path)
    print(f"Model saved at {model_save_path}")

    print(classification_report(all_labels, all_preds, target_names=styles))
    plt.figure(figsize=(15, 12))
    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt="d", cmap="Blues", xticklabels=styles, yticklabels=styles)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(f"{directory}confusion_matrix_epoch_{previous_epochs + epoch + 1}.png")
    plt.close()

    report = classification_report(all_labels, all_preds, target_names=styles, output_dict=True)

    with open(stats_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [
            previous_epochs + epoch + 1,
            avg_train_loss,
            avg_test_loss,
            accuracy
        ]

        for label in styles:
            row.extend([
                report[label]['precision'],
                report[label]['recall'],
                report[label]['f1-score']
            ])

        writer.writerow(row)

    print(f"Statistics for epoch {previous_epochs + epoch + 1} saved to {stats_file_path}")
