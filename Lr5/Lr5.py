import numpy as np
import pandas as pd
import os

# ==========================================
# 1. ЗЧИТУВАННЯ ДАТАСЕТУ
# ==========================================
print("▶ Зчитування датасету 'diabetes.csv'...")
if not os.path.exists("diabetes.csv"):
    raise FileNotFoundError("Помилка: Файл 'diabetes.csv' не знайдено! Переконайтеся, що він лежить у папці Lr5.")

df = pd.read_csv("diabetes.csv")

# Виділяємо ознаки (X) та цільову змінну (y)
# Ознаки: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
X_raw = df.drop(columns=['Outcome']).values
y = df['Outcome'].values
feature_names = df.columns[:-1].tolist()

print(f"  Дані успішно завантажено: {X_raw.shape[0]} пацієнтів, {X_raw.shape[1]} клінічних ознак.")
print(f"  Хворих (Клас 1): {np.sum(y)}, Здорових (Клас 0): {len(y) - np.sum(y)}\n")

# ==========================================
# 2. ФУНКЦІЇ ПІДГОТОВКИ ДАНИХ (З НУЛЯ)
# ==========================================
def train_test_split_custom(X, y, test_ratio=0.2):
    """Розбиття даних на тренувальну та тестову вибірки."""
    np.random.seed(42)  # Для відтворюваності результатів
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_size = int(X.shape[0] * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def standardize_custom(X_train, X_test):
    """Нормалізація ознак (Z-score standardization)."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled

print("▶ Підготовка та нормалізація даних...")
X_train, X_test, y_train, y_test = train_test_split_custom(X_raw, y)
X_train_scaled, X_test_scaled = standardize_custom(X_train, X_test)

# ==========================================
# 3. АЛГОРИТМ ЛОГІСТИЧНОЇ РЕГРЕСІЇ
# ==========================================
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """Функція активації сигмоїда."""
        z = np.clip(z, -250, 250) # Запобігає переповненню
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Градієнтний спуск
        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Обчислення градієнтів
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Оновлення ваг
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X, threshold=0.5):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return (y_predicted > threshold).astype(int)

# ==========================================
# 4. ТРЕНУВАННЯ ТА ОЦІНКА МОДЕЛІ
# ==========================================
print("▶ Навчання моделі (Gradient Descent)...")
model = CustomLogisticRegression(learning_rate=0.1, iterations=3000)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# Оцінка точності
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"\n======================================")
print(f" РЕЗУЛЬТАТИ КЛАСИФІКАЦІЇ")
print(f"======================================")
print(f"Точність моделі (Accuracy): {accuracy * 100:.2f}%\n")

# Інтерпретація ваг
print("--- ІНТЕРПРЕТОВАНІСТЬ: ВПЛИВ ОЗНАК ---")
for name, weight in zip(feature_names, model.weights):
    impact = "Підвищує ризик" if weight > 0 else "Знижує ризик"
    print(f"{name:25s} : Вага = {weight:>6.3f} ({impact})")