import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import StringVar, messagebox
from sklearn.metrics import confusion_matrix
import re 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer

# Veri işleme sınıfı
class DataProcessor:
    def __init__(self, data_path, clean_pattern=None, max_features=None, stop_words=None):
        self.data_path = data_path
        self.clean_pattern = clean_pattern
        self.max_features = max_features
        self.stop_words = stop_words
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words, max_features=self.max_features)
        self.label_map = {"figurative": 0, "sarcasm": 1, "irony": 2, "regular": 3}  # Etiket haritası

    def load_data(self):
        # Eğitim ve test verilerini birleştir
        data = pd.concat([pd.read_csv(self.data_path[0]),
                          pd.read_csv(self.data_path[1])], axis=0)
        data.dropna(inplace=True)  # Eksik verileri temizle
        self.data = data

    def preprocess(self):
        # Tweet verilerini ve etiketlerini ayır
        x = list(self.data["tweets"])
        y = list(self.data["class"])
        x_cleaned = self._clean_texts(x)  # Metin temizleme işlemi
        self.vectorizer.fit(x_cleaned)  # TF-IDF vektörleştirme
        x_vectorized = self.vectorizer.transform(x_cleaned).toarray()
        y_encoded = np.array([self.label_map[label] for label in y])  # Etiketleri encode et
        return x_vectorized, y_encoded

    def _clean_texts(self, texts):
        # Metin temizleme fonksiyonu
        cleaned = []
        for text in texts:
            if self.clean_pattern is not None:
                text = re.sub(self.clean_pattern, " ", text)
            cleaned.append(text.lower().strip())
        return cleaned

    def transform_text(self, text):
        # Yeni bir metni temizleyip vektörleştir
        cleaned_text = self._clean_texts([text])
        return self.vectorizer.transform(cleaned_text).toarray()

# PyTorch Dataset sınıfı
class TweetDataset(Dataset):
    def __init__(self, x_vectorized, y_encoded):
        self.x_vectorized = x_vectorized
        self.y_encoded = y_encoded

    def __len__(self):
        return len(self.x_vectorized)

    def __getitem__(self, index):
        return self.x_vectorized[index], self.y_encoded[index]

# Yapay Sinir Ağı modeli
class DenseNetwork(nn.Module):
    def __init__(self):
        super(DenseNetwork, self).__init__()
        self.fc1 = nn.Linear(7000, 1024)  # İlk tam bağlı katman
        self.drop1 = nn.Dropout(0.4)  # Dropout katmanı (overfitting önlemek için)
        self.fc2 = nn.Linear(1024, 256)  # İkinci tam bağlı katman
        self.drop2 = nn.Dropout(0.4)
        self.prediction = nn.Linear(256, 4)  # Çıkış katmanı

    def forward(self, x):
        x = F.relu(self.fc1(x.to(torch.float)))  # ReLU aktivasyon fonksiyonu
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.log_softmax(self.prediction(x), dim=1)  # Log-Softmax aktivasyon fonksiyonu
        return x

# Model eğitimi ve değerlendirme sınıfı
class Trainer:
    def __init__(self, model, train_loader, validation_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()  # Kayıp fonksiyonu
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-3)  # Optimizasyon algoritması
        self.epochs = 6  # Eğitim epoch sayısı

    def train(self):
        self.train_losses = []
        self.train_accuracies = []
        print(f"{'Epoch':<8}{'Train Loss':<15}{'Train Accuracy (%)':<20}")
        print("-" * 45)
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            epoch_true = 0
            epoch_total = 0
            for data_, target_ in self.train_loader:
                data_ = data_.to(self.device)
                target_ = target_.to(self.device).long()
                self.optimizer.zero_grad()
                outputs = self.model(data_)
                loss = self.criterion(outputs, target_)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                _, pred = torch.max(outputs, dim=1)
                epoch_true += torch.sum(pred == target_).item()
                epoch_total += target_.size(0)
            accuracy = 100 * epoch_true / epoch_total
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(accuracy)
            print(f"{epoch:<8}{epoch_loss:<15.4f}{accuracy:<20.2f}")

    def evaluate(self):
        test_true = 0
        test_total = 0
        test_loss = 0.0
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for data_, target_ in self.validation_loader:
                data_, target_ = data_.to(self.device), target_.to(self.device).long()
                outputs = self.model(data_)
                loss = self.criterion(outputs, target_).item()
                _, pred = torch.max(outputs, dim=1)
                test_true += torch.sum(pred == target_).item()
                test_total += target_.size(0)
                true_labels.extend(target_.cpu().numpy())
                pred_labels.extend(pred.cpu().numpy())
                test_loss += loss
        accuracy = 100 * test_true / test_total
        print("\nEvaluation Results:")
        print(f"{'Test Loss':<15}{'Test Accuracy (%)':<20}")
        print(f"{test_loss:<15.4f}{accuracy:<20.2f}")

        # Sınıflandırma raporu
        class_report = classification_report(true_labels, pred_labels, target_names=["Figurative", "Sarcasm", "Irony", "Regular"])
        print("\nClassification Report:")
        print(class_report)
        
        return accuracy, test_loss, true_labels, pred_labels

# Görselleştirme sınıfı
class Visualizer:
    @staticmethod
    def plot_training_results(train_losses, train_accuracies):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Kayıp - Loss')
        plt.title('Eğitim Kaybı - Training Loss')
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', color='orange', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Doğruluk-Accuracy (%)')
        plt.title('Eğitim Doğruluğu - Training Accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(true_labels, pred_labels, label_names):
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        plt.xlabel('Tahmin(predict) edilen etiketler')
        plt.ylabel('Doğruluk etiketleri')
        plt.title('Confusion Matrix')
        plt.show()

# Tkinter tabanlı kullanıcı arayüzü sınıfı
class App:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = {0: "Figurative", 1: "Sarcasm", 2: "Irony", 3: "Regular"}

    def predict(self, text):
        self.model.eval()
        vectorized_text = self.processor.transform_text(text)
        with torch.no_grad():
            input_tensor = torch.tensor(vectorized_text).to(self.device)
            output = self.model(input_tensor)
            _, predicted = torch.max(output, dim=1)
        return self.label_map[predicted.item()]

    def run_app(self):
        root = tk.Tk()
        root.title("Tweet Sınıflandırma")
        root.geometry("650x500")
        root.configure(bg="#f2f2f2")  # Arka plan rengi

        # Üst başlık
        header = tk.Label(root, text="Tweet Sınıflandırma", font=("Helvetica", 16, "bold"), bg="#f2f2f2", fg="#333")
        header.grid(row=0, column=0, columnspan=2, pady=10)

        # Giriş etiketi ve kutusu
        input_label = tk.Label(root, text="Tweet Girin:", font=("Helvetica", 12), bg="#f2f2f2")
        input_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        input_text = StringVar()
        input_entry = tk.Entry(root, textvariable=input_text, width=40, font=("Helvetica", 10))
        input_entry.grid(row=1, column=1, padx=10, pady=5)

        # Tahmin fonksiyonu
        def classify_tweet():
            tweet = input_text.get()
            if tweet.strip() == "":
                messagebox.showerror("Hata", "Lütfen bir tweet girin.")
                return
            prediction = self.predict(tweet)
            messagebox.showinfo("Tahmin", f"Tweet şu şekilde sınıflandırıldı: {prediction}")

        # Tahmin butonu
        classify_button = tk.Button(
            root,
            text="Sınıflandır",
            command=classify_tweet,
            font=("Helvetica", 12),
            bg="#4CAF50",
            fg="white",
            activebackground="#45a049",
            width=15,
        )
        classify_button.grid(row=2, column=0, columnspan=2, pady=20)

        # Çıkış butonu
        exit_button = tk.Button(
            root,
            text="Çıkış",
            command=root.destroy,
            font=("Helvetica", 12),
            bg="#f44336",
            fg="white",
            activebackground="#e53935",
            width=15,
        )
        exit_button.grid(row=3, column=0, columnspan=2, pady=10)

        root.mainloop()

# Veri işleme
data_processor = DataProcessor(["train.csv", "test.csv"], clean_pattern="[^a-zA-Z0-9]", max_features=7000, stop_words="english")
data_processor.load_data()
x_vectorized, y_encoded = data_processor.preprocess()

# Dataset ve DataLoader oluşturma
dataset = TweetDataset(x_vectorized, y_encoded)
train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.25, random_state=42)
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=128, sampler=test_sampler)

# Model ve Trainer oluşturma
model = DenseNetwork().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
trainer = Trainer(model, train_loader, validation_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
trainer.train()
accuracy, test_loss, true_labels, pred_labels = trainer.evaluate()

# Sonuçları görselleştir
Visualizer.plot_training_results(trainer.train_losses, trainer.train_accuracies)
Visualizer.plot_confusion_matrix(true_labels,pred_labels, ["Figurative", "Sarcasm", "Irony", "Regular"])

# Tkinter uygulamasını çalıştırma
app = App(model, data_processor)
app.run_app() 
