import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.metrics import f1_score, precision_score, recall_score
import time

class PowerQualityAgent:
    def __init__(self):
        self.log = []
        self.report = {}
        self.decisions = []
        
        self.t = None
        self.v_clean = None
        self.v_dirty = None
        self.v_filtered = None
        self.v_rms = None
        
        
        self.true_anomaly_mask = None
        self.pred_anomaly_mask = None
        
        
        self.fs = 2000  
        self.f0 = 50.0 
        self.v_nom = 220.0 

    def _log(self, message, level="INFO"):
        timestamp = time.strftime('%H:%M:%S')
        entry = f"[{timestamp}][{level}] {message}"
        self.log.append(entry)
        print(entry)

    def _decide(self, reason, decision):
        entry = f"  >> РІШЕННЯ: {decision}  (причина: {reason})"
        self.decisions.append({'reason': reason, 'decision': decision})
        print(entry)

    def _separator(self, title=""):
        line = "=" * 65
        if title:
            print(f"\n{line}\n  {title}\n{line}")
        else:
            print(line)

    def cli_intro(self):
        self._separator("AI-АГЕНТ АНАЛІЗУ ЯКОСТІ ЕЛЕКТРОЕНЕРГІЇ v2.0 (Варіант 21)")
        print("Агент автоматично аналізує потоковий сигнал напруги,")
        print("приймає рішення про фільтрацію та виявляє просадки/перенапруги.")
        self._separator()

    def cli_ask_duration(self):
        default = 5
        answer = input(f"  Введіть тривалість симуляції мережі в секундах [{default}]: ").strip()
        try:
            return int(answer) if answer else default
        except ValueError:
            return default

    def cli_ask_noise(self):
        print("\n  Рівень електромагнітного шуму в мережі:")
        print("  [1] Низький   (ідеальні умови)")
        print("  [2] Середній  (типове промислове приміщення)")
        print("  [3] Високий   (сильні перешкоди від інверторів)")
        answer = input("  Ваш вибір [2]: ").strip()
        mapping = {'1': 5, '2': 15, '3': 40}
        return mapping.get(answer, 15)

    def generate_and_inject(self, duration, noise_std):
        self._separator("КРОК 1: СПРИЙНЯТТЯ ТА СИМУЛЯЦІЯ МЕРЕЖІ")
        self._log(f"Генерація ідеального сигналу {self.f0} Гц, {self.v_nom} В...")
        
        n_samples = duration * self.fs
        self.t = np.linspace(0, duration, n_samples)
        amplitude = self.v_nom * np.sqrt(2)
        self.v_clean = amplitude * np.sin(2 * np.pi * self.f0 * self.t)
        
        self._log(f"Додавання шуму (std={noise_std}) та вищих гармонік...")
    
        noise = np.random.normal(0, noise_std, n_samples)
        harmonics = (0.05 * amplitude * np.sin(2 * np.pi * 150 * self.t) + 
                     0.03 * amplitude * np.sin(2 * np.pi * 250 * self.t))
        
        self.v_dirty = self.v_clean + noise + harmonics
        
        self.true_anomaly_mask = np.zeros(n_samples, dtype=bool)
        
        sag_start, sag_end = int(1.2 * self.fs), int(1.6 * self.fs)
        self.v_dirty[sag_start:sag_end] *= 0.7
        self.true_anomaly_mask[sag_start:sag_end] = True
        
        swell_start, swell_end = int(3.1 * self.fs), int(3.4 * self.fs)
        if swell_end < n_samples:
            self.v_dirty[swell_start:swell_end] *= 1.25
            self.true_anomaly_mask[swell_start:swell_end] = True

        self._log(f"Згенеровано {n_samples} точок. Штучних аномалій: 2.")
        self.report['duration'] = duration
        self.report['noise_level'] = noise_std

    def diagnose(self):
        self._separator("КРОК 2: ДІАГНОСТИКА СИГНАЛУ ТА ПРИЙНЯТТЯ РІШЕНЬ")
        
        baseline = self.v_dirty[:self.fs]
        noise_variance = np.var(baseline - self.v_clean[:self.fs])
        signal_variance = np.var(self.v_clean[:self.fs])
        snr = 10 * np.log10(signal_variance / noise_variance)
        
        self._log(f"Оціночне відношення сигнал/шум (SNR): {snr:.1f} дБ")
        
        self._separator("АГЕНТ ПРИЙМАЄ РІШЕННЯ")
        
        if snr > 35:
            filter_type = 'none'
            self._decide(f"SNR={snr:.1f}дБ > 35", "Цифрова фільтрація не потрібна, сигнал чистий")
        elif snr > 20:
            filter_type = 'butter_low'
            self._decide(f"SNR={snr:.1f}дБ", "Застосувати ФНЧ Баттерворта 4-го порядку (Cutoff=100Hz)")
        else:
            filter_type = 'butter_high'
            self._decide(f"SNR={snr:.1f}дБ < 20 (високий шум)", "Застосувати агресивний ФНЧ (Cutoff=60Hz)")

        # Рішення 2: Допустимі межі (Thresholds)
        if self.report['noise_level'] > 20:
            tolerance = 0.15
            self._decide("Високий базовий шум може давати хибні спрацювання", "Розширити коридор допусків до ±15%")
        else:
            tolerance = 0.10
            self._decide("Нормальний рівень шуму", "Встановити стандартний коридор допусків ГОСТ ±10%")

        return filter_type, tolerance

    def apply_filter(self, filter_type):
        self._separator("КРОК 3: ЦИФРОВА ОБРОБКА СИГНАЛУ (DSP)")
        
        if filter_type == 'none':
            self.v_filtered = self.v_dirty.copy()
            self._log("Фільтрацію пропущено.")
        else:
            cutoff = 100 if filter_type == 'butter_low' else 60
            self._log(f"Розрахунок коефіцієнтів фільтра Баттерворта (Зріз {cutoff} Гц)...")
            nyq = 0.5 * self.fs
            normal_cutoff = cutoff / nyq
            b, a = butter(4, normal_cutoff, btype='low', analog=False)
            self.v_filtered = filtfilt(b, a, self.v_dirty)
            self._log("Фільтр успішно застосовано. Високочастотні гармоніки подавлено.")

    def calculate_rms_and_detect(self, tolerance):
        self._separator("КРОК 4: РОЗРАХУНОК RMS ТА ПОШУК АНОМАЛІЙ")
        
        window_size = int(self.fs / self.f0)  
        self._log(f"Розрахунок ковзного RMS. Розмір вікна: {window_size} точок (20 мс)")
        
        v_sq = self.v_filtered ** 2
        window = np.ones(window_size) / window_size
        self.v_rms = np.sqrt(np.convolve(v_sq, window, mode='same'))
        
        lower_bound = self.v_nom * (1 - tolerance)
        upper_bound = self.v_nom * (1 + tolerance)
        self.report['lower_bound'] = lower_bound
        self.report['upper_bound'] = upper_bound
        
        self._log(f"Межі норми RMS: {lower_bound:.1f} В ... {upper_bound:.1f} В")
        
        self.pred_anomaly_mask = (self.v_rms < lower_bound) | (self.v_rms > upper_bound)
        
        total_anomalies = np.sum(self.pred_anomaly_mask)
        self._log(f"Виявлено точок з відхиленням: {total_anomalies} ({(total_anomalies/len(self.t))*100:.1f}%)")

    def compute_metrics(self):
        self._separator("КРОК 5: РОЗРАХУНОК МЕТРИК ЯКОСТІ АГЕНТА")
        
        true_labels = self.true_anomaly_mask.astype(int)
        pred_labels = self.pred_anomaly_mask.astype(int)

        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall    = recall_score(true_labels, pred_labels, zero_division=0)
        f1        = f1_score(true_labels, pred_labels, zero_division=0)

        self._log(f"Precision (точність спрацювань): {precision:.4f}")
        self._log(f"Recall    (повнота виявлення):   {recall:.4f}")
        self._log(f"F1-score  (баланс):              {f1:.4f}")

        self.report.update({
            'precision': round(precision, 4),
            'recall':    round(recall, 4),
            'f1':        round(f1, 4)
        })

    def visualize(self):
        self._separator("КРОК 6: ВІЗУАЛІЗАЦІЯ ТА ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
        self._log("Побудова аналітичних звітів...")

        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
        fig.suptitle('AI-агент якості електроенергії (Full Pipeline)', fontsize=16, fontweight='bold')

        plot_configs = [
            {
                'title': '1. Ідеальна математична модель мережі (220В, 50Гц)',
                'ylabel': 'Напруга (В)',
                'filename': '1_clean_signal.png',
                'plot_func': lambda ax: ax.plot(self.t, self.v_clean, color='steelblue', lw=1)
            },
            {
                'title': '2. Забруднений сигнал (шум + гармоніки + аномалії)',
                'ylabel': 'Напруга (В)',
                'filename': '2_noisy_signal.png',
                'plot_func': lambda ax: (
                    ax.plot(self.t, self.v_dirty, color='gray', lw=0.7, alpha=0.8, label='Сирий сигнал'),
                    ax.scatter(self.t[np.where(self.true_anomaly_mask)[0]], 
                               self.v_dirty[np.where(self.true_anomaly_mask)[0]], 
                               color='red', s=2, label='Реальні аномалії'),
                    ax.legend(loc='upper right')
                )
            },
            {
                'title': '3. Сигнал після цифрової фільтрації (DSP)',
                'ylabel': 'Напруга (В)',
                'filename': '3_filtered_signal.png',
                'plot_func': lambda ax: ax.plot(self.t, self.v_filtered, color='seagreen', lw=1)
            },
            {
                'title': '4. Аналіз RMS та автоматичне виявлення відхилень',
                'ylabel': 'RMS Напруга (В)',
                'filename': '4_rms_detection.png',
                'plot_func': lambda ax: (
                    ax.plot(self.t, self.v_rms, color='darkblue', lw=1.5, label='Ковзне RMS'),
                    ax.axhline(self.report['lower_bound'], color='tomato', ls='--', label='Нижня межа'),
                    ax.axhline(self.report['upper_bound'], color='tomato', ls='--', label='Верхня межа'),
                    ax.fill_between(self.t, 0, 350, where=self.pred_anomaly_mask, color='red', alpha=0.2, label='Тригер Агента'),
                    ax.set_ylim(0, 350),
                    ax.legend(loc='upper right')
                )
            }
        ]

        for i, config in enumerate(plot_configs):
            # Додаємо на загальний фігуру
            config['plot_func'](axes[i])
            axes[i].set_title(config['title'])
            axes[i].set_ylabel(config['ylabel'])
            axes[i].grid(alpha=0.3)
            if i == 3: axes[i].set_xlabel('Час (с)')

            ind_fig, ind_ax = plt.subplots(figsize=(12, 5))
            config['plot_func'](ind_ax)
            ind_ax.set_title(config['title'], fontsize=12)
            ind_ax.set_ylabel(config['ylabel'])
            ind_ax.set_xlabel('Час (с)')
            ind_ax.grid(alpha=0.3)
            plt.tight_layout()
            ind_fig.savefig(config['filename'], dpi=120)
            plt.close(ind_fig) # Закриваємо, щоб не перевантажувати пам'ять
            self._log(f"Збережено окрему діаграму: {config['filename']}")

        plt.figure(fig.number)
        plt.tight_layout()
        plt.savefig('power_quality_full_report.png', dpi=150, bbox_inches='tight')
        plt.show()
        self._log("Загальний аналітичний звіт збережено як: power_quality_full_report.png")

    def print_final_report(self):
        self._separator("ПІДСУМКОВИЙ ЗВІТ АГЕНТА")
        print(f"  Тривалість аналізу:      {self.report.get('duration')} сек")
        print(f"  Рівень шуму:             {self.report.get('noise_level')}")
        print(f"  Коридор допусків RMS:    [{self.report.get('lower_bound'):.1f} В - {self.report.get('upper_bound'):.1f} В]")
        print()
        print(f"  МЕТРИКИ ВИЯВЛЕННЯ АНОМАЛІЙ:")
        print(f"  Precision (Точність):    {self.report.get('precision')}")
        print(f"  Recall (Повнота):        {self.report.get('recall')}")
        print(f"  F1-score:                {self.report.get('f1')}")
        print()
        print("  ПРИЙНЯТІ РІШЕННЯ АГЕНТА:")
        for i, d in enumerate(self.decisions, 1):
            print(f"  {i}. {d['decision']}")
            print(f"     Причина: {d['reason']}")
        self._separator()

    def run(self):
        self.cli_intro()
        
        duration = self.cli_ask_duration()
        noise = self.cli_ask_noise()
        
        self.generate_and_inject(duration, noise)
        filter_type, tolerance = self.diagnose()
        
        self.apply_filter(filter_type)
        self.calculate_rms_and_detect(tolerance)
        self.compute_metrics()
        self.visualize()
        self.print_final_report()


if __name__ == '__main__':
    agent = PowerQualityAgent()
    agent.run()