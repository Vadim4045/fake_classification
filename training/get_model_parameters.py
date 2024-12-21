import torch
from tkinter import filedialog

model_path = filedialog.askopenfilename(initialdir='../training/raw_models/',filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")])
checkpoint = torch.load(model_path, map_location="cpu")
# Итерация по ключам верхнего уровня
for key, value in checkpoint.items():
    print(f"\nКлюч верхнего уровня: {key}")
    if isinstance(value, torch.Tensor):
        print(f"  Размер: {value.shape}")
    elif isinstance(value, dict):  # Если это вложенный OrderedDict
        # Итерация по вложенным параметрам
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, torch.Tensor):
                print(f"  Вложенный ключ: {sub_key}, Размер: {sub_value.shape}")
            else:
                print(f"  Вложенный ключ: {sub_key}, Тип данных: {type(sub_value)}")
    else:
        print(f"  Тип данных: {type(value)}")