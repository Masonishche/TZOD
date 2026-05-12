import pandas as pd
import numpy as np

# 1. Завантаження датасету
df = pd.read_csv('ecommerce_sales_34500.csv')

# --- ШТУЧНЕ ДОДАВАННЯ ПРОПУСКІВ (ДЛЯ ТЕСТУВАННЯ) ---
# Робимо 15% значень у 'quantity' та 10% у 'discount' порожніми (NaN)
np.random.seed(42)
df.loc[df.sample(frac=0.15).index, 'quantity'] = np.nan
df.loc[df.sample(frac=0.10).index, 'discount'] = np.nan
# ---------------------------------------------------

print("--- ЗВІТ ПРОПУСКІВ ДО ІМПУТАЦІЇ ---")
missing_report = pd.DataFrame({
    'Кількість пропусків': df.isnull().sum(),
    'Відсоток (%)': (df.isnull().sum() / len(df)) * 100
})
# Фільтруємо, щоб показати лише ті колонки, де є пропуски
print(missing_report[missing_report['Кількість пропусків'] > 0], "\n")

# 2. Розрахунок середнього чеку ДО імпутації
# Ігноруємо рядки, де бракує quantity чи discount
df_clean = df.dropna(subset=['quantity', 'discount'])
avg_receipt_before = (df_clean['price'] * df_clean['quantity'] * (1 - df_clean['discount'])).mean()

# 3. Імпутація пропусків: заповнення quantity та discount медіаною по product_id
df['quantity'] = df['quantity'].fillna(df.groupby('product_id')['quantity'].transform('median'))
df['discount'] = df['discount'].fillna(df.groupby('product_id')['discount'].transform('median'))

# Перевірка, чи залишились пропуски
print("Кількість пропусків після імпутації ('quantity'):", df['quantity'].isnull().sum())
print("Кількість пропусків після імпутації ('discount'):", df['discount'].isnull().sum(), "\n")

# 4. Розрахунок середнього чеку ПІСЛЯ імпутації
avg_receipt_after = (df['price'] * df['quantity'] * (1 - df['discount'])).mean()

print("--- ПОРІВНЯННЯ СЕРЕДНЬОГО ЧЕКУ ---")
print(f"Середній чек ДО імпутації:    ${avg_receipt_before:.2f}")
print(f"Середній чек ПІСЛЯ імпутації: ${avg_receipt_after:.2f}")