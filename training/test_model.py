import torch
from tkinter import filedialog
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

max_batches = 100
styles = ['Fabrication', 'Advertising', 'Manipulation', 'Propaganda', 'Satire', 'Parody']
directory = './raw_models/'
model_save_pattern = 'RobertaBase_768_1024_1024_512_6'

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8

device = torch.device('mps')

class CustomRobertaForSequenceClassification(nn.Module):
    def __init__(self, num_labels=6):
        super(CustomRobertaForSequenceClassification, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=768)
        self.fc1 = nn.Linear(768, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, num_labels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask).logits
        x = self.leaky_relu(outputs)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        return self.fc4(x)

    def save_weights(self, path):
      weights = {
          'bert': self.roberta.state_dict(),
          'fc1': self.fc1.state_dict(),
          'fc2': self.fc2.state_dict(),
          'fc3': self.fc3.state_dict(),
          'fc4': self.fc4.state_dict()
      }
      torch.save(weights, path)

    def load_weights(self, path):
        weights = torch.load(path, map_location=torch.device('cpu'))
        self.roberta.load_state_dict(weights['bert'])
        self.fc1.load_state_dict(weights['fc1'])
        self.fc2.load_state_dict(weights['fc2'])
        self.fc3.load_state_dict(weights['fc3'])
        self.fc4.load_state_dict(weights['fc4'])

    # Добавляем метод get_model_signature
    def get_model_signature(self):
        # Извлекаем архитектуру RoBERTa и количество выходов
        roberta_config = self.roberta.config
        architecture = "base" if roberta_config.hidden_size == 768 else "large"
        num_labels = self.roberta.config.num_labels
        layer_info = [f"roberta_{architecture}_{num_labels}"]

        # Добавляем информацию о линейных слоях
        for name, module in self.named_children():
            if isinstance(module, nn.Linear):
                layer_info.append(f"{module.out_features}")

        # Генерация строки
        return "_".join(layer_info)


model = CustomRobertaForSequenceClassification(num_labels=6)
model.to(device)

file_path = filedialog.askopenfilename(initialdir='../training/raw_models/',filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")])
model.load_weights(file_path)
model_signature = model.get_model_signature()

data_path = filedialog.askopenfilename(initialdir='../prepare_data/datasets/',filetypes=[("Data sets", "*.pt"), ("All Files", "*.*")])
test_dataset = torch.load(data_path)

test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
loss_fn = nn.CrossEntropyLoss()

model.eval()
total_accuracy_score = 0
total_score = 0
total_test_loss = 0
all_preds = []
all_labels = []

with torch.no_grad():
    progress_bar = tqdm(test_loader, desc='Testing model', leave=False)
    for i, batch in enumerate(progress_bar):
        # Ограничение на количество батчей
        if i >= max_batches:
            break
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

# Имя сохранённого файла
file_name = f"Full_model_{model_signature}_{avg_test_loss:.4f}_{accuracy:.3f}.pth"
print(file_name)
save_path = f'{directory}{file_name}'
torch.save(model,save_path)

print(classification_report(all_labels, all_preds, target_names=styles))
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt="d", cmap="Blues", xticklabels=styles, yticklabels=styles)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


