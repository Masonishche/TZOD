import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

M = 0.1  

def delta_P(t):
    P_gen = 1.0 
    P_load = 1.0 if t < 2 else 1.5
    noise = 0.1 * np.sin(2 * np.pi * t) 
    return P_gen - (P_load + noise)

def grid_dynamics(t, f_dev, M_param, D_param):
    return [(1 / M_param) * (delta_P(t) - D_param * f_dev[0])]

t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 500)

sol_stable = solve_ivp(grid_dynamics, t_span, [0], args=(M, 0.5), t_eval=t_eval)

sol_unstable = solve_ivp(grid_dynamics, t_span, [0], args=(M, 0.05), t_eval=t_eval)

mean_dP = np.mean([delta_P(t) for t in t_eval])
stat_f_dev = (mean_dP / 0.5) * np.ones_like(t_eval)

plt.figure(figsize=(12, 6))

plt.plot(sol_stable.t, sol_stable.y[0], label="Динамічна модель (Стабільна, D=0.5)", color='green', linewidth=2)
plt.plot(sol_unstable.t, sol_unstable.y[0], label="Динамічна модель (Нестабільна, D=0.05)", color='red', linestyle='--', alpha=0.8)
plt.plot(t_eval, stat_f_dev, label="Статистичне середнє (Ігнорує піки)", color='blue', linestyle=':')

plt.title("Моделювання енергетичних потоків: SciPy (Динаміка) vs Статистика")
plt.xlabel("Час (с)")
plt.ylabel("Відхилення в мережі (Δf)")
plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("energy_flow_simulation.png", dpi=300)
plt.show()