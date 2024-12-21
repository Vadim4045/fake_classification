import os
import re
import csv
import torch
import logging
import threading
import numpy as np
import tkinter as tk
import torch.nn as nn
from tkinter.simpledialog import Dialog
from transformers import RobertaTokenizer
from tkinter import messagebox, filedialog
from transformers import RobertaForSequenceClassification

logging.basicConfig(level=logging.WARNING)

styles = ['Fabrication', 'Advertising', 'Manipulation', 'Propaganda', 'Satire', 'Parody']
device = torch.device('mps')
directory = './FinalModels/'
model_name = 'Roberta_1024_512_6_007_0.790_0.904_0.632.pth'
current_model_path = f'{directory}Model_{model_name}'

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

def debug_decorator(func):
    def wrapper(*args, **kwargs):
        logging.debug(f"Calling function '{func.__name__}' with args: {args}, kwargs: {kwargs}")
        return func(*args, **kwargs)
    return wrapper

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

def load_model_async():
    global model
    print("Loading model asynchronously...")
    # model = CustomRobertaForSequenceClassification(num_labels=6)
    # model.load_weights(f'{directory}{model_name}')
    # torch.save(model, f'{directory}Model_{model_name}')
    model = torch.load(current_model_path)
    model.eval()
    model.to(device)
    print("Model loaded successfully.")

def load_new_model():
    file_path = filedialog.askopenfilename(initialdir='./FinalModels/',filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")])
    if not file_path:
        return

    global directory, model_name
    directory, model_name = os.path.split(file_path)
    current_model_path = model_name

    threading.Thread(target=load_model_async, daemon=True).start()
    current_model_name.set(f"Current model: {current_model_path}")

def preprocess(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    return inputs['input_ids'].to(device), inputs['attention_mask'].to(device)


def get_class_probabilities(text):
    input_ids, attention_mask = preprocess(text)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)

    min_logit = torch.min(logits)
    logits = logits - min_logit
    probs = logits / torch.sum(logits, dim=1, keepdim=True)
    probs = probs.cpu().numpy()
    percentages = probs[0] * 100
    return percentages


def classify_paragraphs(paragraphs):
    for widget in result_frame.winfo_children():
        widget.destroy()

    total_probs = np.zeros(len(styles))

    for paragraph in paragraphs:
        percentages = get_class_probabilities(paragraph)
        total_probs += percentages

    avg_probs = total_probs / len(paragraphs)

    sorted_indices = sorted(range(len(avg_probs)), key=lambda j: avg_probs[j], reverse=True)

    for i in sorted_indices:
        style = styles[i]
        percentage = avg_probs[i]

        # Создание Canvas с динамической шириной
        canvas = tk.Canvas(result_frame, height=16)
        canvas.pack(fill=tk.X, pady=1)  # fill=tk.X позволяет растягивать Canvas по ширине родителя

        # Получаем ширину Canvas
        canvas.update_idletasks()  # Обновление для получения корректной ширины
        canvas_width = canvas.winfo_width()

        # Рисуем полосы с использованием текущей ширины
        canvas.create_rectangle(0, 0, canvas_width, 18, fill="light gray")
        canvas.create_rectangle(0, 0, percentage / 100 * canvas_width, 18, fill="lightblue")

        # Текст по центру
        canvas.create_text(
            canvas_width / 2, 11,  # Центрируем текст
            text=f"{style}: {percentage:.2f}%",
            fill="blue",
            font=('Helvetica', 12, 'bold')
        )


def infer_text():
    text = text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Text entry is empty!")
        return

    paragraphs = re.split(r'\n\s*\n*', text)
    print("Number of paragraphs:", len(paragraphs))
    print("Paragraphs:", paragraphs)
    classify_paragraphs(paragraphs)

@debug_decorator
def paste_from_clipboard():
    try:
        clipboard_text = root.clipboard_get().strip()
        text_entry.delete("1.0", tk.END)
        text_entry.insert(tk.END, clipboard_text)
        paragraphs = re.split(r'\n\s*\n*', clipboard_text)
        classify_paragraphs(paragraphs)
    except Exception as e:
        messagebox.showerror("Error", str(e))

@debug_decorator
def load_from_file():
    file_path = filedialog.askopenfilename(initialdir='./',filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if not file_path:
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        file_text = file.read().strip()
        text_entry.delete("1.0", tk.END)
        text_entry.insert(tk.END, file_text)

        paragraphs = re.split(r'\n\s*\n', file_text)
        classify_paragraphs(paragraphs)


@debug_decorator
def process_csv(column_index, file_path):
    try:
        directory, filename = os.path.split(file_path)
        name, ext = os.path.splitext(filename)
        output_file_path = os.path.join(directory, f"{name}_classified{ext}")

        with open(file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', newline='',
                                                                    encoding='utf-8') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            header = next(reader)
            writer.writerow(header + styles)

            for row in reader:
                text = row[column_index]
                paragraphs = re.split(r'\n\s*\n', text)
                total_probs = np.zeros(len(styles))

                for paragraph in paragraphs:
                    percentages = get_class_probabilities(paragraph)
                    total_probs += percentages

                avg_probs = total_probs / len(paragraphs)
                writer.writerow(row + avg_probs.tolist())

        messagebox.showinfo("Success", f"Classified data saved to {output_file_path}")

    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        root.deiconify()  # Show the main window after processing


class ColumnInputDialog(Dialog):
    def __init__(self, parent, title=None):
        self.result = None
        self.dialog_window = None
        super().__init__(parent, title=title)

    def body(self, master):
        tk.Label(master, text="Enter the column number to classify (starting from 1):").pack(pady=5)
        self.column_entry = tk.Entry(master)
        self.column_entry.pack(pady=5)

    def apply(self):
        try:
            self.result = int(self.column_entry.get()) - 1
        except ValueError:
            self.result = None

    def buttonbox(self):
        box = tk.Frame(self)
        w = tk.Button(box, text="OK", width=10, command=self.ok)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        box.pack()

    def ok(self):
        self.apply()
        self.destroy()


def load_from_csv():
    file_path = filedialog.askopenfilename(initialdir="./",filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
    if not file_path:
        return

    # Hide the main window before processing CSV
    # root.withdraw()

    dialog = ColumnInputDialog(root, "Input Column Number")
    root.wait_window(dialog.dialog_window)  # Wait for the custom dialog to close

    column_index = dialog.result

    if column_index is None or column_index < 0:
        messagebox.showerror("Error", "Invalid column number.")
        root.after(3000, root.deiconify)  # Show the main window again after 3 seconds
        return

    root.after(100, process_csv, column_index, file_path)  # Process CSV after a short delay


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 30
        y = y + self.widget.winfo_rooty() + 30
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "12", "normal"))
        label.pack(ipadx=2)

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


root = tk.Tk()
root.title("Classification utility")

current_model_name = tk.StringVar(value=f"Current model: {current_model_path}")

# Model selection frame
model_frame = tk.Frame(root)
model_frame.pack(pady=5)

model_label = tk.Label(model_frame, text="Current Model:")
model_label.pack(side=tk.LEFT, padx=5)

model_name_label = tk.Label(model_frame, textvariable=current_model_name, fg="blue")
model_name_label.pack(side=tk.LEFT, padx=5)

load_model_button = tk.Button(model_frame, text="Load New Model", command=load_new_model)
load_model_button.pack(side=tk.LEFT, padx=5)

text_entry = tk.Text(root)
text_entry.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

button_frame = tk.Frame(root)
button_frame.pack(pady=3)

paste_button = tk.Button(button_frame, text="Paste from Clipboard", command=paste_from_clipboard)
paste_button.pack(side=tk.LEFT, padx=2)

file_button = tk.Button(button_frame, text="Load from File", command=load_from_file)
file_button.pack(side=tk.LEFT, padx=2)

csv_button = tk.Button(button_frame, text="Load from CSV", command=load_from_csv)
csv_button.pack(side=tk.LEFT, padx=2)

ToolTip(load_model_button, "Chose model to load")
ToolTip(paste_button, "Paste text from clipboard and classify")
ToolTip(file_button, "Load text from a text file and classify")
ToolTip(csv_button, "Load text from a CSV file,\nclassify and save in new CSV\n  (may take long time)")

result_frame = tk.Frame(root)
result_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

threading.Thread(target=load_model_async, daemon=True).start()

root.mainloop()

