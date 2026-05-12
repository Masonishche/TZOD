import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. ЗАВАНТАЖЕННЯ РЕАЛЬНИХ ДАНИХ
# ==========================================
file_name = "magnetic_data.csv"

print(f"▶ Зчитування експериментальних даних з '{file_name}'...")
if not os.path.exists(file_name):
    raise FileNotFoundError(f"Помилка: Файл '{file_name}' не знайдено! Проведіть експеримент та створіть файл.")

df = pd.read_csv(file_name)

print("\n--- Зібрані дані ---")
print(df)

# ==========================================
# 2. АНАЛІЗ ДАНИХ
# ==========================================
# Знаходимо максимальне значення для кожного пристрою (зазвичай на відстані 0 см)
max_field_per_device = df.groupby('Device')['Magnetic_Field_uT'].max()
print("\n--- Максимальне магнітне випромінювання (впритул) ---")
for device, value in max_field_per_device.items():
    print(f"{device}: {value} µT")

# Обчислюємо, на скільки відсотків падає поле при віддаленні на 20 см
def calculate_drop(group):
    max_val = group['Magnetic_Field_uT'].values[0] # Значення на 0 см
    min_val = group['Magnetic_Field_uT'].values[-1] # Значення на 20 см
    drop_percent = ((max_val - min_val) / max_val) * 100
    return drop_percent

drop_stats = df.groupby('Device').apply(calculate_drop)
print("\n--- Падіння магнітного поля на відстані 20 см ---")
for device, drop in drop_stats.items():
    print(f"{device}: Зниження на {drop:.1f}%")

# ==========================================
# 3. ВІЗУАЛІЗАЦІЯ (ПОБУДОВА ГРАФІКА)
# ==========================================
plt.figure(figsize=(10, 6))

# Будуємо лінію для кожного пристрою
devices = df['Device'].unique()
colors = ['red', 'blue', 'green', 'purple', 'orange']

for i, device in enumerate(devices):
    device_data = df[df['Device'] == device]
    plt.plot(device_data['Distance_cm'], device_data['Magnetic_Field_uT'], 
             marker='o', linewidth=2, label=device, color=colors[i % len(colors)])

# Оформлення графіка
plt.title("Залежність рівня магнітного поля від відстані до пристрою", fontsize=14, fontweight='bold')
plt.xlabel("Відстань від джерела (см)", fontsize=12)
plt.ylabel("Рівень магнітного поля (µT)", fontsize=12)
plt.xticks(df['Distance_cm'].unique()) # Показуємо точні відмітки на осі Х
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Електронні пристрої", fontsize=11)
plt.tight_layout()

# Збереження графіка для звіту
plt.savefig("magnetic_field_analysis.png", dpi=300)
print("\n▶ Графік успішно побудовано та збережено як 'magnetic_field_analysis.png'")
plt.show()