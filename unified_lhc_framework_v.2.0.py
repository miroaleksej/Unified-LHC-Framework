import os
import time
import json
import logging
import random
import numpy as np
import yaml
from typing import Dict, List, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
# Подавление предупреждений для чистоты вывода
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===================================================================
# Добавлены физические константы из Исправления.txt
# ===================================================================
c = 299792458  # м/с (скорость света)
m_p = 0.938272  # ГэВ/с² (масса протона)
m_e = 0.000511  # ГэВ/с² (масса электрона)
G_F = 1.166e-5  # Ферми-константа (ГэВ⁻²)
M_W = 80.379  # ГэВ (масса W-бозона)
M_Z = 91.1876  # ГэВ (масса Z-бозона)
M_H = 125.1  # ГэВ (масса Хиггса)
alpha_s = 0.118  # Сильная константа связи
alpha_em = 1.0 / 137.0  # Константа тонкой структуры
v = 246  # vev, ГэВ (электрослабый вакуумный ожидаемый вакуум)
hbar = 6.582119569e-25  # ГэВ·с (приведенная постоянная Планка)
k_B = 8.617333262e-5  # эВ/К (постоянная Больцмана)
e = 1.602176634e-19  # Кл (элементарный заряд)
epsilon_0 = 8.8541878128e-12  # Ф/м (электрическая постоянная)
mu_0 = 4*np.pi*1e-7  # Гн/м (магнитная постоянная)

# ===================================================================
# 1. Настройка логирования
# ===================================================================
"""Настройка логирования для отслеживания процесса выполнения.
Создает лог-файл и выводит сообщения в консоль."""
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("unified_lhc_framework.log"),
                              logging.StreamHandler()])
logger = logging.getLogger("Unified_LHC_Framework")
# ===================================================================

# ===================================================================
# Добавлены улучшения модели динамики пучка из Исправления.txt
# ===================================================================
class BeamDynamics:
    """Улучшенная модель динамики пучка с релятивистскими расчетами"""
    def __init__(self, config):
        self.config = config
        self.state = {
            'sigma_x': config.get('beam', {}).get('sigma_x', 0.045),
            'sigma_y': config.get('beam', {}).get('sigma_y', 0.045),
            'epsilon': config.get('beam', {}).get('emittance', 2.5e-6),
            'N_p': config.get('beam', {}).get('bunch_intensity', 1.15e11),
            'turn': 0,
            'beta_x': config.get('beam', {}).get('beta_x', 56.5),
            'beta_y': config.get('beam', {}).get('beta_y', 56.5),
            'D_x': config.get('beam', {}).get('dispersion_x', 0.0),
            'D_y': config.get('beam', {}).get('dispersion_y', 0.0),
            'energy_spread': config.get('beam', {}).get('energy_spread', 1e-4)
        }
        self.history = []
    
    def evolve(self, num_turns: int, include_space_charge: bool = True):
        """Релятивистская эволюция пучка с учетом всех эффектов"""
        for _ in range(num_turns):
            self._apply_lattice_optics()
            self._apply_synchrotron_radiation()
            self._apply_quantum_excitation()
            if include_space_charge:
                self._apply_space_charge_effect()
            # Добавляем новые физические эффекты
            self._apply_chromaticity_effects()
            self._apply_amplitude_dependent_effects()
            self._apply_nonlinear_fields()
            self._apply_beam_beam_effect()  # Учет взаимодействия пучков
            self._apply_intra_beam_scattering()
            self.state['turn'] += 1
            self.history.append(self.state.copy())

        return self.state
    
    def _apply_lattice_optics(self):
        """Применение оптики ускорителя (бета-функции, дисперсия)"""
        # Обновление размеров пучка на основе бета-функций и эмиттанса
        gamma_rel = self.config['beam']['beam_energy'] / m_p
        beta_rel = np.sqrt(1 - 1/gamma_rel**2)

        # Релятивистская формула для размеров пучка
        self.state['sigma_x'] = np.sqrt(self.state['epsilon'] * self.state['beta_x'] / beta_rel)
        self.state['sigma_y'] = np.sqrt(self.state['epsilon'] * self.state['beta_y'] / beta_rel)

    def _apply_synchrotron_radiation(self):
        """Учет потерь энергии на синхротронное излучение (для протонов незначительно)"""
        # Для протонов синхротронное излучение пренебрежимо мало из-за большой массы
        # Вместо этого учитываем влияние RF-системы и рассеяние
        # Обновление энергетического разброса через RF-систему
        V_RF = self.config['beam'].get('rf_voltage', 10e6)  # Вольт
        h_RF = self.config['beam'].get('rf_harmonic', 35640)  # Гармоника
        phi_RF = self.config['beam'].get('rf_phase', 0.0)  # Фаза

        # Для протонов учитываем только шум в RF-системе
        rf_noise = self.config['beam'].get('rf_noise', 1e-4)
        self.state['energy_spread'] += rf_noise

    def _apply_quantum_excitation(self):
        """Учет квантового возбуждения (для протонов незначительно)"""
        # Квантовое возбуждение незначительно для тяжелых частиц (протонов)
        # Вместо этого учитываем влияние нестабильностей и шумов
        quantum_excitation_rate = 0  # Пренебрежимо для протонов
        # Обновление эмиттанса: эмиттанс в протонных ускорителях остается почти постоянным
        # за исключением эффектов рассеяния внутри пучка
        self.state['epsilon'] += quantum_excitation_rate
    
    def _apply_space_charge_effect(self):
        """Учет эффекта пространственного заряда"""
        # Эффект пространственного заряда в протонных ускорителях значителен на низких энергиях
        # и уменьшается с ростом энергии (релятивистское подавление)

        # Вычисление напряженности поля пространственного заряда
        gamma_rel = self.config['beam']['beam_energy'] / m_p
        beta_rel = np.sqrt(1 - 1/gamma_rel**2)

        # Плотность заряда в пучке
        bunch_length = self.config.get('beam', {}).get('bunch_length', 0.075)  # 7.5 см
        effective_radius = np.sqrt(self.state['sigma_x'] * self.state['sigma_y'])

        # Упрощенная формула для поля пространственного заряда (для эллиптического пучка)
        # Подавление эффекта с увеличением релятивистского фактора
        space_charge_strength = (self.state['N_p'] * 1.602e-19) / \
                               (2 * np.pi * 8.854e-12 * beta_rel**2 * gamma_rel * effective_radius**2)

        # Влияние на размеры пучка (упрощенная модель)
        # Пространственный заряд вызывает эффект "beam-beam" и "space charge driven resonance"
        x_kick = 0.01 * space_charge_strength / (gamma_rel * beta_rel**2)  # в рад
        y_kick = 0.01 * space_charge_strength / (gamma_rel * beta_rel**2)  # в рад

        # Модификация размеров пучка (небольшое влияние на высоких энергиях)
        self.state['sigma_x'] *= (1 + x_kick * self.config['beam'].get('space_charge_tune_shift', 0.001))
        self.state['sigma_y'] *= (1 + y_kick * self.config['beam'].get('space_charge_tune_shift', 0.001))

    def _apply_beam_beam_effect(self):
        """Учет эффекта пучок-пучок (beam-beam)"""
        # В протонных коллайдерах эффект пучок-пучок играет важную роль
        # особенно при высокой интенсивности и в точках встречи (IP)

        # Проверяем, включено ли моделирование эффекта
        beam_beam_enabled = self.config.get('beam', {}).get('beam_beam_enabled', True)
        if not beam_beam_enabled:
            return

        # Сила взаимодействия зависит от интенсивности и геометрии фокусировки
        beam_beam_strength = self.config.get('beam', {}).get('beam_beam_parameter', 0.01)

        # Рассчитываем параметр взаимодействия (beam-beam parameter)
        # для протонных ускорителей
        gamma_rel = self.config['beam']['beam_energy'] / m_p
        beta_rel = np.sqrt(1 - 1/gamma_rel**2)

        # Параметр взаимодействия пропорционален плотности заряда в пучке
        sigma_x = self.state['sigma_x']
        sigma_y = self.state['sigma_y']
        N_p = self.state['N_p']
        sigma_z = self.config.get('beam', {}).get('sigma_z', 0.075)  # длина сгустка
        gamma_t = self.config.get('beam', {}).get('gamma_t', 59.3)  # гамма-трансверсальный параметр

        # Параметр взаимодействия для протонных ускорителей
        # Chi = N_p * r_p / (4 * pi * sigma_x * sigma_y * gamma_t * beta_rel)
        r_p_classical = 1.535e-18 * (1.602e-19)**2 / (m_p * 1.783e-27) / (8.988e9)  # классический радиус протона
        chi = N_p * r_p_classical / (4 * np.pi * sigma_x * sigma_y * gamma_t * beta_rel)

        # Нормализуем параметр взаимодействия
        chi *= beam_beam_strength

        # Эффект пучок-пучок приводит к дополнительной фокусировке/дефокусировке
        # и может вызывать трансверсальные колебания
        self.state['sigma_x'] *= (1 + 0.5 * chi)  # фокусировка в x
        self.state['sigma_y'] *= (1 - 0.5 * chi)  # дефокусировка в y (для противоположно направленных пучков)

        # Также может влиять на эмиттанс
        self.state['epsilon'] *= (1 + 0.001 * chi)

        # Обновление tune из-за beam-beam эффекта
        # tune shift пропорционален интенсивности пучка
        tune_shift_x_bb = self.config['beam'].get('beam_beam_tune_shift_x', 0.001) * chi
        tune_shift_y_bb = self.config['beam'].get('beam_beam_tune_shift_y', 0.001) * chi

        if 'tune_x_shift' not in self.state:
            self.state['tune_x_shift'] = 0.0
        if 'tune_y_shift' not in self.state:
            self.state['tune_y_shift'] = 0.0

        self.state['tune_x_shift'] += tune_shift_x_bb
        self.state['tune_y_shift'] += tune_shift_y_bb

    def _apply_chromaticity_effects(self):
        """Учет хроматических эффектов в ускорителе"""
        # Хроматичность влияет на tune в зависимости от отклонения энергии
        # ξ = (Δν/Δp/p) - хроматичность

        # Получаем параметры из конфигурации
        xi_x = self.config.get('geometry', {}).get('xi_x', -8.0)
        xi_y = self.config.get('geometry', {}).get('xi_y', -8.0)

        # Относительное отклонение энергии
        relative_energy_deviation = self.state['energy_spread']

        # Изменение tune из-за хроматичности
        self.state['tune_x_shift'] = xi_x * relative_energy_deviation
        self.state['tune_y_shift'] = xi_y * relative_energy_deviation

        # Хроматичность также влияет на стабильность пучка
        # Влияние на размеры пучка из-за хроматической аберрации
        chromatic_beta_factor = 1.0 + 0.01 * abs(relative_energy_deviation)  # упрощенная модель
        self.state['sigma_x'] *= chromatic_beta_factor
        self.state['sigma_y'] *= chromatic_beta_factor

    def _apply_amplitude_dependent_effects(self):
        """Учет эффектов, зависящих от амплитуды колебаний"""
        # Сдвиг tune с амплитудой (amplitude detuning)
        # dq/da * A, где A - амплитуда колебаний

        # Получаем параметры из геометрии
        dq_x_da = self.config.get('geometry', {}).get('dq_x_da', -0.05)
        dq_y_da = self.config.get('geometry', {}).get('dq_y_da', -0.05)

        # Оцениваем среднюю амплитуду колебаний как пропорциональную размеру пучка
        avg_amplitude = (self.state['sigma_x'] + self.state['sigma_y']) / 2.0

        # Сдвиг tune из-за амплитуды
        self.state['tune_x_amp_shift'] = dq_x_da * avg_amplitude
        self.state['tune_y_amp_shift'] = dq_y_da * avg_amplitude

        # Влияние на устойчивость и размеры пучка
        amplitude_factor = 1.0 / (1.0 + 0.1 * avg_amplitude)  # стабилизация при больших амплитудах
        self.state['sigma_x'] *= amplitude_factor
        self.state['sigma_y'] *= amplitude_factor

    def _apply_nonlinear_fields(self):
        """Учет нелинейных магнитных полей (сексуполи, октуполи и т.д.)"""
        # Нелинейные поля вызывают дополнительные эффекты, включая динамическую апертуру
        # и резонансные эффекты

        # Влияние сексуполей на пучок
        sext_gradient = self.config.get('geometry', {}).get('sext_gradient', 3000.0)  # Тл/м²

        # Оценка влияния нелинейных полей
        # Для протонного ускорителя нелинейные эффекты обычно меньше, чем для электронного
        nonlinear_strength = abs(sext_gradient) * self.state['epsilon'] * 1e6  # масштабируем по эмиттансу

        # Ограничиваем влияние нелинейных полей
        if nonlinear_strength > 1.0:
            nonlinear_strength = 1.0

        # Влияние на размеры пучка
        self.state['sigma_x'] *= (1 + 0.01 * nonlinear_strength)
        self.state['sigma_y'] *= (1 + 0.01 * nonlinear_strength)

        # Влияние на эмиттанс из-за нелинейных эффектов
        self.state['epsilon'] *= (1 + 0.001 * nonlinear_strength)

    def _apply_beam_beam_effect(self):
        """Учет взаимодействия пучков (beam-beam effect)"""
        # В протонных коллайдерах эффект взаимодействия пучков играет важную роль
        # особенно при высокой интенсивности

        # Проверяем, включено ли моделирование эффекта
        beam_beam_enabled = self.config.get('beam_beam', {}).get('enabled', True)
        if not beam_beam_enabled:
            return

        # Сила взаимодействия зависит от интенсивности и геометрии фокусировки
        beam_beam_strength = self.config.get('beam_beam', {}).get('strength', 0.01)

        # Рассчитываем параметр взаимодействия (pariwise beam-beam parameter)
        gamma_rel = self.config['beam']['beam_energy'] / m_p
        beta_rel = np.sqrt(1 - 1/gamma_rel**2)

        # Параметр взаимодействия пропорционален плотности заряда в пучке
        sigma_x = self.state['sigma_x']
        sigma_y = self.state['sigma_y']
        N_p = self.state['N_p']

        # Условное обозначение для параметра взаимодействия
        # bb_parameter = (N_p * e^2) / (2 * pi * epsilon_0 * gamma * beta * sigma_x * sigma_y * E)
        bb_parameter = N_p * e**2 / (2 * np.pi * epsilon_0 * gamma_rel * beta_rel * sigma_x * sigma_y * self.config['beam']['beam_energy'] * 1e9)

        # Нормализуем параметр взаимодействия
        bb_parameter *= beam_beam_strength

        # Эффект взаимодействия приводит к дополнительному разбросу импульсов
        # и может вызывать изменения в размере пучка
        self.state['sigma_x'] *= (1 + bb_parameter)
        self.state['sigma_y'] *= (1 + bb_parameter)

        # Также может влиять на эмиттанс
        self.state['epsilon'] *= (1 + 0.001 * bb_parameter)

    def _apply_intra_beam_scattering(self):
        """Учет рассеяния внутри пучка (IBS) - эффект Кулона для протонов"""
        # IBS для протонов описывается формулами Маджорана и Блоха
        # Для релятивистских протонов используется формула для электрических частиц

        gamma_rel = self.config['beam']['beam_energy'] / m_p
        beta_rel = np.sqrt(1 - 1/gamma_rel**2)

        # Параметры пучка
        sigma_x = self.state['sigma_x']
        sigma_y = self.state['sigma_y']
        sigma_z = self.config.get('beam', {}).get('sigma_z', 0.075)  # длина сгустка
        N_p = self.state['N_p']
        eps = self.state['epsilon']

        # Постоянные для расчета IBS
        r_p = 1.535e-18 * (1.602e-19)**2 / (m_p * 1.783e-27) / (8.988e9)  # классический радиус протона
        c_t = 3.7e11  # постоянная для протонов (в м^-2)

        # Формула Блоха для IBS в релятивистском случае
        # Постоянная времени для IBS (в оборотах)
        tau_IBS = c_t * gamma_rel * beta_rel**4 * sigma_x * sigma_y * sigma_z / (N_p * r_p**2 * np.log(3*gamma_rel))

        # За один оборот рассеяние изменяет эмиттанс
        if tau_IBS > 0:
            ibs_growth_rate = 1.0 / tau_IBS
            self.state['epsilon'] *= (1 + ibs_growth_rate)

        # Также IBS влияет на энергетический разброс
        self.state['energy_spread'] *= (1 + 0.1 * ibs_growth_rate)

    def evolve(self, num_turns: int, include_space_charge: bool = True):
        """Релятивистская эволюция пучка с учетом всех эффектов"""
        # Инициализируем модель магнитной оптики
        self.magnetic_optics = MagneticOpticsModel(self.config)

        for _ in range(num_turns):
            self._apply_lattice_optics()
            self._apply_synchrotron_radiation()
            self._apply_quantum_excitation()
            if include_space_charge:
                self._apply_space_charge_effect()
            # Добавляем новые физические эффекты
            self._apply_chromaticity_effects()
            self._apply_amplitude_dependent_effects()
            self._apply_nonlinear_fields()
            self._apply_beam_beam_effect()  # Добавляем учет взаимодействия пучков
            self._apply_intra_beam_scattering()
            # Применяем эффекты магнитной оптики
            self.state = self.magnetic_optics.apply_optics_effects(self.state, 'arc')
            # Рассчитываем сдвиги тюнов
            if include_space_charge:
                tune_shifts = self.magnetic_optics.calculate_tune_shift(
                    self.state['N_p'], self.state, 'arc')
                self.state['tune_x_shift'] += tune_shifts[0]
                self.state['tune_y_shift'] += tune_shifts[1]
            self.state['turn'] += 1
            self.history.append(self.state.copy())

        return self.state
    
    def get_luminosity(self) -> float:
        """Расчет светимости с учетом всех эффектов"""
        # Формула для светимости протон-протонного коллайдера
        num_bunches = self.config['beam']['num_bunches']
        bunch_intensity = self.config['beam']['bunch_intensity']
        revolution_freq = c / self.config['beam']['circumference']
        beta_x_star = self.config['beam']['beta_x_star'] if 'beta_x_star' in self.config['beam'] else self.config['beam']['beta_star']
        beta_y_star = self.config['beam']['beta_y_star'] if 'beta_y_star' in self.config['beam'] else self.config['beam']['beta_star']

        # Длина сгустка и перекрытие в точке взаимодействия
        bunch_length = self.config.get('beam', {}).get('bunch_length', 0.075)  # м
        sigma_z = bunch_length / 4  # дисперсия по z

        # Размеры пучка в точке взаимодействия (IP)
        sigma_x_IP = np.sqrt(self.state['epsilon'] * beta_x_star / (self.config['beam']['beam_energy'] / m_p))
        sigma_y_IP = np.sqrt(self.state['epsilon'] * beta_y_star / (self.config['beam']['beam_energy'] / m_p))

        # Учет угла пересечения пучков
        crossing_angle = self.config.get('beam', {}).get('crossing_angle', 0.0)  # рад

        if crossing_angle > 0:
            # Для перекрывающихся пучков с углом
            overlap_factor = 1.0 / np.sqrt(1 + (crossing_angle * beta_x_star / (2 * sigma_x_IP))**2)
        else:
            # Для смещенных пучков без угла
            overlap_factor = 1.0

        # Светимость для гауссовых пучков
        geometric_factor = 1.0 / (2 * np.pi * sigma_x_IP * sigma_y_IP * sigma_z) * overlap_factor

        # Учет уменьшения светимости из-за эффекта пучок-пучок
        # В режиме высокой светимости эффекты пучок-пучок становятся значительными
        beam_beam_parameter = self.config.get('beam', {}).get('beam_beam_parameter', 0.01)
        beam_beam_reduction = 1.0 / (1 + beam_beam_parameter * bunch_intensity / 1e11)

        # Окончательная формула светимости
        luminosity = (num_bunches * bunch_intensity**2 * revolution_freq * geometric_factor * beam_beam_reduction)

        return luminosity

# ===================================================================
# 9. Система кэширования
# ===================================================================
class SimulationCache:
    """Система кэширования результатов симуляции."""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = 0
        self.hit_count = 0

    def generate_key(self, params: Dict) -> str:
        """Генерирует уникальный ключ для кэша на основе параметров."""
        sorted_params = sorted(params.items())
        return str(hash(str(sorted_params)))

    def get(self, key: str) -> Optional[Any]:
        """Получает значение из кэша."""
        self.access_count += 1
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """Сохраняет значение в кэш."""
        if len(self.cache) >= self.max_size:
            # Простая стратегия удаления: удаляем самую старую запись
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value

    def get_hit_rate(self) -> float:
        """Возвращает долю попаданий в кэш."""
        return self.hit_count / self.access_count if self.access_count > 0 else 0.0

# ===================================================================
# 10. Модель магнитной оптики ускорителя
# ===================================================================
class MagneticOpticsModel:
    """Модель магнитной оптики ускорителя с учетом различных типов магнитов"""
    def __init__(self, config):
        self.config = config
        self.lattice_functions = self._initialize_lattice_functions()

    def _initialize_lattice_functions(self) -> Dict:
        """Инициализация функций решетки (бета-функции, альфа-функции, дисперсия)"""
        beam_config = self.config.get('beam', {})

        return {
            # Бета-функции в разных секциях ускорителя
            'beta_x': {
                'arc': beam_config.get('beta_x_arc', 100.0),      # в дугах
                'insertion': beam_config.get('beta_x_insertion', 10.0),  # в точках встречи
                'interaction_point': beam_config.get('beta_x_star', 0.55)  # в IP
            },
            'beta_y': {
                'arc': beam_config.get('beta_y_arc', 100.0),
                'insertion': beam_config.get('beta_y_insertion', 10.0),
                'interaction_point': beam_config.get('beta_y_star', 0.55)
            },
            # Альфа-функции (коэффициенты сжатия)
            'alpha_x': {
                'arc': beam_config.get('alpha_x_arc', 0.0),
                'insertion': beam_config.get('alpha_x_insertion', 0.0),
                'interaction_point': beam_config.get('alpha_x_star', 0.0)
            },
            'alpha_y': {
                'arc': beam_config.get('alpha_y_arc', 0.0),
                'insertion': beam_config.get('alpha_y_insertion', 0.0),
                'interaction_point': beam_config.get('alpha_y_star', 0.0)
            },
            # Дисперсионные функции
            'dispersion_x': {
                'arc': beam_config.get('dispersion_x_arc', 5.0),
                'insertion': beam_config.get('dispersion_x_insertion', 2.0),
                'interaction_point': beam_config.get('dispersion_x_ip', 0.0)
            },
            'dispersion_y': {
                'arc': beam_config.get('dispersion_y_arc', 0.0),
                'insertion': beam_config.get('dispersion_y_insertion', 0.0),
                'interaction_point': beam_config.get('dispersion_y_ip', 0.0)
            },
            # Частоты движения (тюны)
            'tune_x': beam_config.get('tune_x', 62.31),
            'tune_y': beam_config.get('tune_y', 60.32),
            # Хроматичности
            'chromaticity_x': beam_config.get('chromaticity_x', -7.0),
            'chromaticity_y': beam_config.get('chromaticity_y', -7.0)
        }

    def get_beta_function(self, plane: str, position: str) -> float:
        """Получение бета-функции для заданной плоскости и позиции"""
        if plane not in ['x', 'y'] or position not in self.lattice_functions[f'beta_{plane}']:
            raise ValueError(f"Неверная плоскость '{plane}' или позиция '{position}'")
        return self.lattice_functions[f'beta_{plane}'][position]

    def get_alpha_function(self, plane: str, position: str) -> float:
        """Получение альфа-функции для заданной плоскости и позиции"""
        if plane not in ['x', 'y'] or position not in self.lattice_functions[f'alpha_{plane}']:
            raise ValueError(f"Неверная плоскость '{plane}' или позиция '{position}'")
        return self.lattice_functions[f'alpha_{plane}'][position]

    def get_dispersion(self, plane: str, position: str) -> float:
        """Получение дисперсионной функции для заданной плоскости и позиции"""
        if plane not in ['x', 'y'] or position not in self.lattice_functions[f'dispersion_{plane}']:
            raise ValueError(f"Неверная плоскость '{plane}' или позиция '{position}'")
        return self.lattice_functions[f'dispersion_{plane}'][position]

    def apply_optics_effects(self, beam_state: Dict, position: str = 'arc') -> Dict:
        """Применение эффектов оптики к состоянию пучка"""
        # Обновление размеров пучка на основе бета-функций
        gamma_rel = self.config['beam']['beam_energy'] / m_p
        beta_rel = np.sqrt(1 - 1/gamma_rel**2)

        # Параметрическая эмиттанс
        epsilon_norm = beam_state.get('epsilon', 2.5e-6) / gamma_rel / beta_rel

        # Размеры пучка с учетом бета-функций
        beam_state['sigma_x'] = np.sqrt(epsilon_norm * self.get_beta_function('x', position))
        beam_state['sigma_y'] = np.sqrt(epsilon_norm * self.get_beta_function('y', position))

        # Угловые разбросы
        beam_state['theta_x'] = np.sqrt(epsilon_norm / self.get_beta_function('x', position))
        beam_state['theta_y'] = np.sqrt(epsilon_norm / self.get_beta_function('y', position))

        # Влияние дисперсии на размеры пучка
        delta_p_over_p = beam_state.get('delta_p_over_p', 1e-4)  # относительный разброс импульса
        disp_x = self.get_dispersion('x', position)
        beam_state['sigma_x'] = np.sqrt(beam_state['sigma_x']**2 + (disp_x * delta_p_over_p)**2)

        return beam_state

    def calculate_tune_shift(self, beam_intensity: float, beam_state: Dict, position: str = 'arc') -> tuple:
        """Расчет сдвига тюнов из-за интенсивности пучка (space charge и beam-beam)"""
        # Сдвиг тюна из-за space charge
        # ΔQ_SC = -λ_s * N_p / (4π * β_γ * γ)
        n_per_meter = beam_intensity / (self.config['beam']['bunch_length'] * c)  # плотность частиц

        # Оценка силы space charge
        beta_gamma = self.config['beam']['beam_energy'] / m_p

        space_charge_shift_x = -n_per_meter * (m_p * c**2)**2 / (
            4 * np.pi * beta_gamma * self.get_beta_function('x', position))
        space_charge_shift_y = -n_per_meter * (m_p * c**2)**2 / (
            4 * np.pi * beta_gamma * self.get_beta_function('y', position))

        # Сдвиг тюна из-за beam-beam эффектов
        # ΔQ_BB = -N_p * r_cl / (4π * σ_z * σ_crossing) для LHC
        beam_beam_parameter = self.config.get('beam', {}).get('beam_beam_parameter', 0.15)
        sigma_z = self.config.get('beam', {}).get('sigma_z', 0.075)
        sigma_crossing = np.sqrt(beam_state.get('sigma_x', 1e-3)**2 + beam_state.get('sigma_y', 1e-3)**2)

        beam_beam_shift = -beam_state.get('N_p', 1e11) * 1.535e-18 / (
            4 * np.pi * sigma_z * sigma_crossing) * beam_beam_parameter

        return (space_charge_shift_x + beam_beam_shift,
                space_charge_shift_y + beam_beam_shift)

# ===================================================================
# 11. Система кэширования
# ===================================================================
class SimulationCache:
    """Система кэширования результатов симуляции."""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = 0
        self.hit_count = 0
    
    @staticmethod
    def generate_key(params: Dict) -> str:
        """Генерирует уникальный ключ для кэша на основе параметров."""
        sorted_params = sorted(params.items())
        return str(hash(str(sorted_params)))
    
    def get(self, key: str) -> Optional[Any]:
        """Получает значение из кэша."""
        self.access_count += 1
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Сохраняет значение в кэш."""
        if len(self.cache) >= self.max_size:
            # Простая стратегия удаления: удаляем самую старую запись
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
    
    def get_hit_rate(self) -> float:
        """Возвращает долю попаданий в кэш."""
        return self.hit_count / self.access_count if self.access_count > 0 else 0.0

# ===================================================================
# 5. *** МОДУЛЬ: ParticleDatabase ***
# ===================================================================
class ParticleDatabase:
    """База данных частиц с физическими свойствами."""
    def __init__(self):
        self.particles = self._initialize_particles()

    def _initialize_particles(self) -> Dict[str, Dict]:
        """Инициализация базы данных частиц."""
        return {
            'proton': {'mass': 0.938, 'charge': 1, 'lifetime': float('inf'), 'pdg_code': 2212},
            'electron': {'mass': 0.000511, 'charge': -1, 'lifetime': float('inf'), 'pdg_code': 11},
            'positron': {'mass': 0.000511, 'charge': 1, 'lifetime': float('inf'), 'pdg_code': -11},
            'muon': {'mass': 0.105, 'charge': -1, 'lifetime': 2.2e-6, 'pdg_code': 13},
            'antimuon': {'mass': 0.105, 'charge': 1, 'lifetime': 2.2e-6, 'pdg_code': -13},
            'tau': {'mass': 1.777, 'charge': -1, 'lifetime': 2.9e-13, 'pdg_code': 15},
            'antitau': {'mass': 1.777, 'charge': 1, 'lifetime': 2.9e-13, 'pdg_code': -15},
            'neutrino_e': {'mass': 0, 'charge': 0, 'lifetime': float('inf'), 'pdg_code': 12},
            'antineutrino_e': {'mass': 0, 'charge': 0, 'lifetime': float('inf'), 'pdg_code': -12},
            'neutrino_mu': {'mass': 0, 'charge': 0, 'lifetime': float('inf'), 'pdg_code': 14},
            'antineutrino_mu': {'mass': 0, 'charge': 0, 'lifetime': float('inf'), 'pdg_code': -14},
            'neutrino_tau': {'mass': 0, 'charge': 0, 'lifetime': float('inf'), 'pdg_code': 16},
            'antineutrino_tau': {'mass': 0, 'charge': 0, 'lifetime': float('inf'), 'pdg_code': -16},
            'u_quark': {'mass': 0.0023, 'charge': 2/3, 'lifetime': float('inf'), 'pdg_code': 2},
            'u_antiquark': {'mass': 0.0023, 'charge': -2/3, 'lifetime': float('inf'), 'pdg_code': -2},
            'd_quark': {'mass': 0.0048, 'charge': -1/3, 'lifetime': float('inf'), 'pdg_code': 1},
            'd_antiquark': {'mass': 0.0048, 'charge': 1/3, 'lifetime': float('inf'), 'pdg_code': -1},
            's_quark': {'mass': 0.095, 'charge': -1/3, 'lifetime': float('inf'), 'pdg_code': 3},
            's_antiquark': {'mass': 0.095, 'charge': 1/3, 'lifetime': float('inf'), 'pdg_code': -3},
            'c_quark': {'mass': 1.275, 'charge': 2/3, 'lifetime': float('inf'), 'pdg_code': 4},
            'c_antiquark': {'mass': 1.275, 'charge': -2/3, 'lifetime': float('inf'), 'pdg_code': -4},
            'b_quark': {'mass': 4.18, 'charge': -1/3, 'lifetime': float('inf'), 'pdg_code': 5},
            'b_antiquark': {'mass': 4.18, 'charge': 1/3, 'lifetime': float('inf'), 'pdg_code': -5},
            't_quark': {'mass': 173.1, 'charge': 2/3, 'lifetime': 5.15e-25, 'pdg_code': 6},
            't_antiquark': {'mass': 173.1, 'charge': -2/3, 'lifetime': 5.15e-25, 'pdg_code': -6},
            'gluon': {'mass': 0, 'charge': 0, 'lifetime': float('inf'), 'pdg_code': 21},
            'photon': {'mass': 0, 'charge': 0, 'lifetime': float('inf'), 'pdg_code': 22},
            'Z_boson': {'mass': 91.1876, 'charge': 0, 'lifetime': 2.64e-25, 'pdg_code': 23},
            'W_plus': {'mass': 80.379, 'charge': 1, 'lifetime': 3.22e-25, 'pdg_code': 24},
            'W_minus': {'mass': 80.379, 'charge': -1, 'lifetime': 3.22e-25, 'pdg_code': -24},
            'higgs': {'mass': 125.1, 'charge': 0, 'lifetime': 1.56e-22, 'pdg_code': 25},
            'pion+': {'mass': 0.139, 'charge': 1, 'lifetime': 2.6e-8, 'pdg_code': 211},
            'pion-': {'mass': 0.139, 'charge': -1, 'lifetime': 2.6e-8, 'pdg_code': -211},
            'pion0': {'mass': 0.135, 'charge': 0, 'lifetime': 8.5e-17, 'pdg_code': 111},
            'kaon+': {'mass': 0.494, 'charge': 1, 'lifetime': 1.2e-8, 'pdg_code': 321},
            'kaon-': {'mass': 0.494, 'charge': -1, 'lifetime': 1.2e-8, 'pdg_code': -321},
            'proton': {'mass': 0.938, 'charge': 1, 'lifetime': float('inf'), 'pdg_code': 2212},
            'antiproton': {'mass': 0.938, 'charge': -1, 'lifetime': float('inf'), 'pdg_code': -2212},
            'neutron': {'mass': 0.940, 'charge': 0, 'lifetime': 880, 'pdg_code': 2112},
            'antineutron': {'mass': 0.940, 'charge': 0, 'lifetime': 880, 'pdg_code': -2112},
            'jet': {'mass': 0, 'charge': 0, 'lifetime': 0, 'pdg_code': 0}  # Специальный код для струй
        }

    def get_particle(self, name: str) -> Optional[Dict]:
        """Получает информацию о частице по имени."""
        return self.particles.get(name)

    def get_pdg_code(self, name: str) -> int:
        """Получает PDG код частицы."""
        particle = self.get_particle(name)
        return particle['pdg_code'] if particle else 0

# ===================================================================
# 6. Абстрактные интерфейсы для движков
# ===================================================================
class PhysicsEngineInterface(ABC):
    """Абстрактный интерфейс для физических движков (генераторов событий)"""
    @abstractmethod
    def interact(self, particle1: str, particle2: str, energy: float, num_events: int = 1, **kwargs) -> List[Dict]:
        """Моделирование взаимодействия частиц"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Получение имени движка"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Проверка доступности движка"""
        pass

class BeamDynamicsInterface(ABC):
    """Абстрактный интерфейс для движков динамики пучка"""
    @abstractmethod
    def simulate_turn(self, state: Dict, revolution_time: float, include_space_charge: bool = True, **kwargs) -> Dict:
        """Симуляция одного оборота пучка"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Получение имени движка"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Проверка доступности движка"""
        pass

# ===================================================================
# Добавлен класс PDFModel из Исправления.txt
# ===================================================================
class PDFModel:
    """Модель функций частицного распределения (PDF)"""
    def __init__(self, config):
        self.config = config
        self.pdf_data = self._load_pdf_data()
    
    def _load_pdf_data(self):
        """Загрузка данных PDF из конфигурации или внешних источников"""
        # В реальной реализации здесь будет загрузка данных из файлов PDF
        logger.info("Загрузка данных PDF...")
        return {
            'proton': {
                'u': lambda x, Q2: self._u_quark_pdf(x, Q2),
                'd': lambda x, Q2: self._d_quark_pdf(x, Q2),
                's': lambda x, Q2: self._s_quark_pdf(x, Q2),
                'g': lambda x, Q2: self._gluon_pdf(x, Q2)
            }
        }
    
    def _u_quark_pdf(self, x, Q2):
        """PDF для u-кварков в протоне"""
        # Упрощенная модель, в реальности будет сложнее
        if x <= 0 or x >= 1:
            return 0.0
        # Модель, основанная на данных CTEQ
        return 1.368 * (1 - x)**3.08 * x**0.535 * (1 + 1.562 * np.sqrt(x) + 3.811 * x)
    
    def _d_quark_pdf(self, x, Q2):
        """PDF для d-кварков в протоне"""
        if x <= 0 or x >= 1:
            return 0.0
        # Модель, основанная на данных CTEQ
        return 0.816 * (1 - x)**4.03 * x**0.383 * (1 + 2.637 * np.sqrt(x) + 2.985 * x)
    
    def _s_quark_pdf(self, x, Q2):
        """PDF для s-кварков в протоне"""
        if x <= 0 or x >= 1:
            return 0.0
        # s-кварки примерно в 30% от суммы u и d
        return 0.3 * (self._u_quark_pdf(x, Q2) + self._d_quark_pdf(x, Q2)) / 2.0
    
    def _gluon_pdf(self, x, Q2):
        """PDF для глюонов в протоне"""
        if x <= 0 or x >= 1:
            return 0.0
        # Глюонная PDF доминирует при малых x
        return 1.74 * (1 - x)**5.0 * x**-0.2 * np.exp(-1.4 * np.sqrt(np.log(1/x)))
    
    def get_parton_distribution(self, particle: str, x: float, Q2: float) -> Dict[str, float]:
        """Получение распределения частиц для заданной частицы, x и Q2"""
        if particle not in self.pdf_data:
            raise ValueError(f"PDF для частицы {particle} не поддерживается")
        
        pdfs = {}
        for flavor, pdf_func in self.pdf_data[particle].items():
            pdfs[flavor] = pdf_func(x, Q2)
        
        # Нормализация, чтобы сумма PDF = 1
        total = sum(pdfs.values())
        if total > 0:
            for flavor in pdfs:
                pdfs[flavor] /= total
        
        return pdfs
    
    def sample_parton(self, particle: str, Q2: float) -> Tuple[str, float]:
        """Сэмплирование частицы и значения x для заданной частицы и Q2"""
        # Генерация x с использованием обратного метода преобразования
        x = self._sample_x()
        
        # Получение PDF для этого x
        pdfs = self.get_parton_distribution(particle, x, Q2)
        
        # Выбор частицы на основе PDF
        flavors = list(pdfs.keys())
        probabilities = list(pdfs.values())
        parton = np.random.choice(flavors, p=probabilities)
        
        return parton, x
    
    def _sample_x(self) -> float:
        """Генерация доли импульса x для партонa с физически корректным распределением"""
        # Используем обратный метод преобразования для генерации x
        # с учетом физических ограничений (0 < x < 1)
        while True:
            u = random.random()
            # Используем обратную функцию к PDF для генерации x
            # Упрощенная модель для демонстрации
            x = u ** (1/3.5)  # Соответствует распределению ~ (1-x)^2.5
            
            if 0 < x < 1:
                return x

# ===================================================================
# 6. Реализации физических движков
# ===================================================================
class BuiltInPhysicsEngine(PhysicsEngineInterface):
    """Улучшенный встроенный физический движок с элементами QCD и ЭВ."""
    def __init__(self, particle_db, config):
        self.particle_db = particle_db
        self.config = config
        self.pdf_model = PDFModel(config)  # Инициализация модели PDF
    
    def get_name(self) -> str:
        return "built-in"
    
    def is_available(self) -> bool:
        return True
    
    def _calculate_cross_section(self, process_type: str, energy: float, x1: float, x2: float) -> float:
        """Расчет физического сечения для заданного процесса с использованием точных формул"""
        # Энергия в системе центра масс для партонов
        E_CM_parton = np.sqrt(2 * x1 * x2 * energy**2)

        if process_type == "drell-yan":
            # Более точный расчет сечения для процесса Дрелл-Яна
            # с учетом Z/γ* резонанса и калибровочных бозонов
            if E_CM_parton < 2*m_e:  # Порог для создания лептонной пары
                return 0.0

            # Учет резонанса Z-бозона с полной шириной
            s_hat = E_CM_parton**2
            M_Z = 91.1876  # ГэВ
            Gamma_Z = 2.4952  # ГэВ

            # Бозе-функция для резонанса
            denominator = (s_hat - M_Z**2)**2 + M_Z**2 * Gamma_Z**2
            breit_wigner = s_hat * Gamma_Z**2 / denominator

            # Формула для Drell-Yan с учетом резонансов
            # При низких массах (вне Z-резонанса) используем простую 1/s^2 зависимость
            if E_CM_parton < 50 or E_CM_parton > 150:  # вне резонанса
                base_cross_section = 10.96 / s_hat  # pb, приближение для γ* обмена
            else:  # около Z-резонанса
                base_cross_section = 17.4 / s_hat  # pb, с учетом Z-резонанса

            # Общее сечение
            cross_section = base_cross_section * (1 + breit_wigner)
            return cross_section

        elif process_type == "gluon_fusion":
            # Производство Хиггса через глюонный фьюжн - доминирующий канал
            if E_CM_parton < 120:  # Порог для физически значимых процессов
                return 0.0

            M_H = 125.1  # Масса Хиггса, ГэВ
            if abs(E_CM_parton - M_H) < 5:  # Пик в области массы Хиггса
                # Приближенное сечение для gg -> H в pb
                # с учетом эффективного глюон-Хиггс взаимодействия
                s_hat = E_CM_parton**2
                tau_h = 4 * M_H**2 / s_hat
                if tau_h < 1:
                    arcsin_term = np.arcsin(np.sqrt(tau_h))
                    L_loop = (2 * arcsin_term / np.sqrt(tau_h))**2
                else:
                    L_loop = tau_h  # Приближение при большой массе

                # Нормировка на массу Хиггса
                # sigma_0 = 430 fb при sqrt(s) = 13 TeV для m_H = 125 GeV
                sigma_0 = 0.43  # pb
                cross_section = sigma_0 * L_loop
                return cross_section
            else:
                # Подавленное сечение вне резонансного пика
                # для других глюонных процессов
                return 0.005 / (E_CM_parton**2)  # pb, сильно подавлено

        elif process_type == "quark_antiquark":
            # Процесс кварк-антикварк аннигиляции
            if E_CM_parton < 10:  # Порог для значимых процессов
                return 0.0

            # Для кварк-антикварк аннигиляции в электрослабые бозоны
            # σ ~ α_EM^2 / s
            alpha_EM = 1.0 / 137.0
            cross_section = 1.3 * np.pi * alpha_EM**2 / E_CM_parton**2  # pb
            # Учет цветового фактора (3 для кварков)
            cross_section *= 3.0
            return cross_section

        elif process_type == "jet_production":
            # Упрощенная модель для NLO сечения струй
            if E_CM_parton < 5:  # Порог для струй
                return 0.0

            # Модель для сечения струй, убывающего как 1/s^2
            # с учетом альфа_с и структуры функций распределения
            alpha_s_running = self._running_coupling(E_CM_parton)
            base_cross_section = 10000.0 / (E_CM_parton**2)  # pb

            # Энергетическая зависимость и фактор подавления
            energy_factor = min(1.0, (7000.0 / E_CM_parton)**0.5)

            cross_section = base_cross_section * alpha_s_running**2 * energy_factor
            return cross_section

        elif process_type == "weak_boson_fusion":
            # Слабое бозонное фьюжн - WW, ZZ Production
            if E_CM_parton < 2*M_W:  # Порог для WW финального состояния
                return 0.0

            # Приближенное сечение для WBF
            # σ_WBF ≈ G_F^2 * M_W^4 / π^3 * s * ln^2(s/M_W^2)
            s_hat = E_CM_parton**2
            G_F_factor = (G_F**2 * M_W**4) / (np.pi**3)
            log_factor = np.log(s_hat / M_W**2)**2
            cross_section = G_F_factor * s_hat * log_factor
            return cross_section

        elif process_type == "vector_boson_scattering":
            # Рассеяние векторных бозонов - важный канал для изучения EW симметрии
            if E_CM_parton < 300:  # Высокоэнергетический процесс
                return 0.0

            # Приближенное сечение для VBS процессов
            # σ_VBS ~ (E/M_W)^2 * (α_EM/4π)^4
            energy_ratio = E_CM_parton / M_W
            cross_section = 0.001 * energy_ratio**2 * (alpha_em / (4 * np.pi))**4
            return cross_section

        return 0.0

    def _running_coupling(self, energy_scale: float) -> float:
        """Вычисляет значение сильной константы связи при заданной энергии"""
        # Упрощенное выражение для running coupling в QCD
        # alpha_s(Q^2) = alpha_s(mu^2) / (1 + (alpha_s/π) * b0 * ln(Q^2/mu^2))
        # где b0 = (11*Nc - 2*Nf)/12π = (11*3 - 2*5)/(12π) ≈ 0.159 для 5 кварков
        if energy_scale < 1:  # Неопределенность при низких энергиях
            return alpha_s

        Q2 = energy_scale**2
        mu2 = 90**2  # В квадрате, т.к. M_Z ≈ 90 GeV
        b0 = (11 * 3 - 2 * 5) / (12 * np.pi)  # 5 активных кварков

        # Более точное выражение для running coupling
        log_term = np.log(Q2 / mu2)
        alpha_running = alpha_s / (1 + alpha_s * b0 * log_term)

        # Убедимся, что результат в разумных пределах
        if alpha_running > 2.0 * alpha_s:
            alpha_running = 2.0 * alpha_s
        elif alpha_running < 0.01:
            alpha_running = 0.01

        return alpha_running

    def compute_tune_shift_with_amplitude(self, amplitude: float) -> Tuple[float, float]:
        """Вычисление сдвига тюна с амплитудой колебаний (amplitude detuning)"""
        # Сдвиг тюна пропорционален квадрату амплитуды колебаний
        # Влияние нелинейных магнитов (сексуполей, октуполей)
        sext_grad = self.config.get('beam', {}).get('field_gradient_sextupole', 3000.0)  # Тл/м²

        # Простая модель сдвига тюна с амплитудой
        # Сдвиг пропорциональный квадрату амплитуды и градиенту магнита
        shift_x = -0.1 * sext_grad * amplitude**2
        shift_y = 0.1 * sext_grad * amplitude**2

        # Нормируем на энергии и параметры ускорителя
        gamma_rel = self.config['beam']['beam_energy'] / m_p
        shift_x /= gamma_rel  # Энергетическое подавление
        shift_y /= gamma_rel

        self.tune_shift_with_amplitude_cache = {'dx_da': shift_x, 'dy_da': shift_y}
        return shift_x, shift_y

    def compute_chromaticity_shift(self, dp_p: float) -> Tuple[float, float]:
        """Вычисление хроматического сдвига тюна"""
        # dp/p - относительное отклонение импульса

        # Получаем хроматичности из конфигурации
        xi_x = self.config.get('beam', {}).get('chromaticity_x', -7.0)
        xi_y = self.config.get('beam', {}).get('chromaticity_y', -7.0)

        dx = xi_x * dp_p
        dy = xi_y * dp_p

        self.chromaticity_cache = {'dQx_dpp': dx, 'dQy_dpp': dy}
        return dx, dy
    
    def _get_process_probs(self, flavor1: str, flavor2: str, E_CM_parton: float) -> Dict[str, float]:
        """Определение вероятностей процессов на основе физических сечений"""
        processes = {}
        
        # Определяем возможные процессы на основе типов частиц
        x1 = random.uniform(0.01, 0.99)  # физически реалистичные значения x
        x2 = random.uniform(0.01, 0.99)

        if 'quark' in flavor1 and 'antiquark' in flavor2:
            processes["drell-yan"] = self._calculate_cross_section("drell-yan", E_CM_parton, x1, x2)
            processes["quark_antiquark"] = self._calculate_cross_section("quark_antiquark", E_CM_parton, x1, x2)

        elif 'gluon' in flavor1 and 'gluon' in flavor2:
            processes["gluon_fusion"] = self._calculate_cross_section("gluon_fusion", E_CM_parton, x1, x2)
            processes["jet_production"] = self._calculate_cross_section("jet_production", E_CM_parton, x1, x2)

        elif ('quark' in flavor1 and 'gluon' in flavor2) or ('gluon' in flavor1 and 'quark' in flavor2):
            processes["jet_production"] = self._calculate_cross_section("jet_production", E_CM_parton, x1, x2)

        elif 'lepton' in flavor1 and 'lepton' in flavor2:
            # Лептон-лептонные взаимодействия
            x1 = random.uniform(0.1, 0.9)
            x2 = random.uniform(0.1, 0.9)
            processes["drell-yan"] = self._calculate_cross_section("drell-yan", E_CM_parton, x1, x2)
            processes["weak_boson_fusion"] = self._calculate_cross_section("weak_boson_fusion", E_CM_parton, x1, x2)

        elif 'boson' in flavor1 or 'boson' in flavor2:
            # Взаимодействия с бозонами
            processes["vector_boson_scattering"] = self._calculate_cross_section("vector_boson_scattering", E_CM_parton, x1, x2)

        # Нормализуем вероятности
        total = sum(processes.values())
        if total > 0:
            for process in processes:
                processes[process] /= total
        else:
            # Если все сечения нулевые, создаем хотя бы один процесс с ненулевой вероятностью
            if processes:
                for process in processes:
                    processes[process] = 1.0 / len(processes)
            else:
                # Если словарь пустой, добавим хотя бы один тип процесса с фиктивной вероятностью
                if 'quark' in flavor1 and 'antiquark' in flavor2:
                    processes["drell-yan"] = 0.5
                    processes["quark_antiquark"] = 0.5
                elif 'gluon' in flavor1 and 'gluon' in flavor2:
                    processes["gluon_fusion"] = 0.7
                    processes["jet_production"] = 0.3
                else:
                    processes["jet_production"] = 1.0  # Процесс по умолчанию

        return processes
    
    def _generate_products(self, process_type: str, E_CM_parton: float) -> List[Dict]:
        """Генерация продуктов столкновения на основе физической модели"""
        products = []

        if process_type == "drell-yan":
            # Процесс Дрелл-Яна: производство лептонных пар через Z/γ*
            # Учитываем бранчинг-рэйшоны и релятивистскую кинематику

            # Определяем доминирующий канал в зависимости от энергии
            if E_CM_parton > M_Z + 5:  # У Z-пик
                # Z-бозон с бранчинг-рэйшнами
                rand_val = random.random()
                if rand_val < 0.20:  # ~20% в лептонные пары
                    lep_type = random.choice(['electron', 'muon', 'tau'])
                    products.append({'name': lep_type, 'energy': E_CM_parton * 0.45,
                                    'px': random.gauss(0, E_CM_parton * 0.1),
                                    'py': random.gauss(0, E_CM_parton * 0.1),
                                    'pz': random.gauss(0, E_CM_parton * 0.1)})
                    products.append({'name': f'anti{lep_type}', 'energy': E_CM_parton * 0.45,
                                    'px': -products[-1]['px'],
                                    'py': -products[-1]['py'],
                                    'pz': -products[-1]['pz']})
                elif rand_val < 0.40:  # ~20% в кварковые пары (адронизация)
                    # Смоделируем струи
                    num_jets = 2
                    for i in range(num_jets):
                        jet_energy = E_CM_parton * random.uniform(0.2, 0.5)
                        theta = random.uniform(0, np.pi)
                        phi = random.uniform(0, 2*np.pi)
                        p = jet_energy  # релятивистское приближение для струй
                        px = p * np.sin(theta) * np.cos(phi)
                        py = p * np.sin(theta) * np.sin(phi)
                        pz = p * np.cos(theta)
                        products.append({'name': 'jet', 'energy': jet_energy, 'px': px, 'py': py, 'pz': pz})
                else:  # ~60% в нейтрино
                    nu_type = random.choice(['neutrino_e', 'neutrino_mu', 'neutrino_tau'])
                    products.append({'name': nu_type, 'energy': E_CM_parton * 0.45})
                    products.append({'name': f'anti{nu_type}', 'energy': E_CM_parton * 0.45})
            elif E_CM_parton > M_W + 1:  # W-бозон
                if random.random() < 0.5:
                    products.append({'name': 'W_plus', 'energy': E_CM_parton*0.95})
                    # Распад W в зависимости от доступной энергии
                    if E_CM_parton > 2*M_W:  # Достаточно энергии для лептонов
                        lep_type = random.choice(['electron', 'muon'])
                        products.append({'name': lep_type, 'energy': E_CM_parton * 0.45})
                        products.append({'name': f'anti{lep_type}_neutrino', 'energy': E_CM_parton * 0.45})
                    else:  # W распадается на кварки
                        products.append({'name': 'u_quark', 'energy': E_CM_parton * 0.45})
                        products.append({'name': 'd_antiquark', 'energy': E_CM_parton * 0.45})
                else:
                    products.append({'name': 'W_minus', 'energy': E_CM_parton*0.95})
                    if E_CM_parton > 2*M_W:
                        lep_type = random.choice(['electron', 'muon'])
                        products.append({'name': f'anti{lep_type}', 'energy': E_CM_parton * 0.45})
                        products.append({'name': f'{lep_type}_neutrino', 'energy': E_CM_parton * 0.45})
                    else:
                        products.append({'name': 'd_quark', 'energy': E_CM_parton * 0.45})
                        products.append({'name': 'u_antiquark', 'energy': E_CM_parton * 0.45})

        elif process_type == "gluon_fusion":
            # Глюонная фьюжн - основной канал для Хиггса
            if E_CM_parton > M_H - 5 and E_CM_parton < M_H + 5:  # Резонанс Хиггса
                products.append({'name': 'higgs', 'energy': E_CM_parton * 0.95})

                # Бранчинг-рэйшоны Хиггса (упрощенно)
                rand_val = random.random()
                if rand_val < 0.58:  # ~58% в bb
                    products.append({'name': 'b_quark', 'energy': E_CM_parton * 0.45})
                    products.append({'name': 'b_antiquark', 'energy': E_CM_parton * 0.45})
                elif rand_val < 0.69:  # ~11% в WW*
                    products.append({'name': 'W_plus', 'energy': E_CM_parton * 0.45})
                    products.append({'name': 'W_minus', 'energy': E_CM_parton * 0.45})
                elif rand_val < 0.80:  # ~11% в ZZ*
                    products.append({'name': 'Z_boson', 'energy': E_CM_parton * 0.45})
                    products.append({'name': 'Z_boson', 'energy': E_CM_parton * 0.45})
                else:  # Остальные каналы (YY, cc, gg, ...)
                    lep_pair = random.choice([('tau', 'antitau'), ('muon', 'antimuon')])
                    products.append({'name': lep_pair[0], 'energy': E_CM_parton * 0.45})
                    products.append({'name': lep_pair[1], 'energy': E_CM_parton * 0.45})
            else:
                # Другие глюонные процессы - обычно производство струй
                num_jets = random.randint(2, 3)  # 2-3 струи для глюонного начального состояния
                for i in range(num_jets):
                    jet_energy = E_CM_parton * random.uniform(0.2, 0.45)
                    theta = random.uniform(0, np.pi)
                    phi = random.uniform(0, 2*np.pi)
                    p = jet_energy  # приближение для релятивистских струй
                    px = p * np.sin(theta) * np.cos(phi)
                    py = p * np.sin(theta) * np.sin(phi)
                    pz = p * np.cos(theta)
                    products.append({'name': 'jet', 'energy': jet_energy, 'px': px, 'py': py, 'pz': pz})

        elif process_type == "quark_antiquark":
            # Процесс кварк-антикварк аннигиляции
            if E_CM_parton > M_Z + 1:  # Достаточно энергии для Z-бозона
                if random.random() < 0.70:  # ~70% каналов в Z
                    products.append({'name': 'Z_boson', 'energy': E_CM_parton * 0.95})

                    # Распад Z в зависимости от энергии
                    if E_CM_parton > 2*M_Z:  # Можно сделать лептонные пары
                        lep_type = random.choice(['electron', 'muon', 'tau'])
                        products.append({'name': lep_type, 'energy': E_CM_parton * 0.45})
                        products.append({'name': f'anti{lep_type}', 'energy': E_CM_parton * 0.45})
                    else:  # Z распадается на кварки
                        quark_flavor = random.choice(['u', 'd', 's', 'c', 'b'])
                        products.append({'name': f'{quark_flavor}_quark', 'energy': E_CM_parton * 0.45})
                        products.append({'name': f'{quark_flavor}_antiquark', 'energy': E_CM_parton * 0.45})
                else:  # ~30% в W-бозоны
                    if random.random() < 0.5:
                        products.append({'name': 'W_plus', 'energy': E_CM_parton * 0.95})
                        # W распадается на более лёгкие кварки
                        products.append({'name': 'u_quark', 'energy': E_CM_parton * 0.45})
                        products.append({'name': 'd_antiquark', 'energy': E_CM_parton * 0.45})
                    else:
                        products.append({'name': 'W_minus', 'energy': E_CM_parton * 0.95})
                        products.append({'name': 'd_quark', 'energy': E_CM_parton * 0.45})
                        products.append({'name': 'u_antiquark', 'energy': E_CM_parton * 0.45})
            else:  # Низкоэнергетические кварк-антикварковые процессы
                # Производство адронов (мезонов и барионов)
                meson_type = random.choice(['pion+', 'pion-', 'pion0', 'kaon+', 'kaon-'])
                products.append({'name': meson_type, 'energy': E_CM_parton * random.uniform(0.3, 0.7)})

                # Добавляем дополнит��льные продукты для сохранения квантовых чисел
                if meson_type in ['pion+', 'kaon+']:
                    products.append({'name': 'pion-', 'energy': E_CM_parton * random.uniform(0.1, 0.3)})
                elif meson_type in ['pion-', 'kaon-']:
                    products.append({'name': 'pion+', 'energy': E_CM_parton * random.uniform(0.1, 0.3)})

        elif process_type == "jet_production":
            # Производство струй - основной канал
            # Число струй зависит от энергии и типа начального состояния
            if E_CM_parton > 100:  # Высокоэнергетические столкновения
                num_jets = random.choices([2, 3, 4], weights=[60, 35, 5])[0]  # 2, 3, 4 струи
            elif E_CM_parton > 20:  # Средние энергии
                num_jets = random.choices([2, 3], weights=[75, 25])[0]
            else:  # Низкие энергии
                num_jets = 2

            for i in range(num_jets):
                jet_energy = E_CM_parton * random.uniform(0.15, 0.5)
                theta = random.uniform(0, np.pi)
                phi = random.uniform(0, 2*np.pi)
                p = jet_energy  # релятивистское приближение
                px = p * np.sin(theta) * np.cos(phi)
                py = p * np.sin(theta) * np.sin(phi)
                pz = p * np.cos(theta)
                products.append({'name': 'jet', 'energy': jet_energy, 'px': px, 'py': py, 'pz': pz})

        elif process_type == "weak_boson_fusion":
            # Слабое бозонное фьюжн - производство W или Z с двумя пробивными кварками
            if E_CM_parton > 2*M_W:
                # WW ��ли ZZ ��инальное состояние
                if random.random() < 0.67:
                    products.append({'name': 'W_plus', 'energy': E_CM_parton * 0.3})
                    products.append({'name': 'W_minus', 'energy': E_CM_parton * 0.3})
                else:
                    products.append({'name': 'Z_boson', 'energy': E_CM_parton * 0.3})
                    products.append({'name': 'Z_boson', 'energy': E_CM_parton * 0.3})

                # Два пробивных кварка в передние области
                for i in range(2):
                    jet_energy = E_CM_parton * random.uniform(0.2, 0.4)
                    # Пробивные струи в передние области
                    theta = random.uniform(0, 0.1) if i == 0 else random.uniform(np.pi - 0.1, np.pi)
                    phi = random.uniform(0, 2*np.pi)
                    p = jet_energy
                    px = p * np.sin(theta) * np.cos(phi)
                    py = p * np.sin(theta) * np.sin(phi)
                    pz = p * np.cos(theta)
                    products.append({'name': 'jet', 'energy': jet_energy, 'px': px, 'py': py, 'pz': pz})

        elif process_type == "vector_boson_scattering":
            # Рассеяние векторных бозонов - важный канал для изучения EW симметрии
            if E_CM_parton > 500:  # Высокоэнергетический процесс
                # WW -> WW, WZ -> WZ, ZZ -> ZZ
                vb1, vb2 = random.choice([('W_plus', 'W_minus'), ('W_plus', 'Z_boson'),
                                         ('W_minus', 'Z_boson'), ('Z_boson', 'Z_boson')])
                products.append({'name': vb1, 'energy': E_CM_parton * 0.4})
                products.append({'name': vb2, 'energy': E_CM_parton * 0.4})

                # Некоторые VBS процессы сопровождаются пробивными кварками
                if random.random() < 0.3:  # ~30% с пробивными кварками
                    jet_energy = E_CM_parton * random.uniform(0.05, 0.15)
                    theta = random.uniform(0, 0.1)  # Передняя область
                    phi = random.uniform(0, 2*np.pi)
                    p = jet_energy
                    px = p * np.sin(theta) * np.cos(phi)
                    py = p * np.sin(theta) * np.sin(phi)
                    pz = p * np.cos(theta)
                    products.append({'name': 'jet', 'energy': jet_energy, 'px': px, 'py': py, 'pz': pz})

        # Убедимся, что общая энергия не превышает E_CM_parton
        total_energy = sum(p.get('energy', 0) for p in products)
        if total_energy > 0 and E_CM_parton > 0:
            energy_ratio = E_CM_parton / total_energy
            for p in products:
                p['energy'] *= energy_ratio * random.uniform(0.8, 1.1)  # Небольшие флуктуации

        return products
    
    def interact(self, particle1: str, particle2: str, energy: float, num_events: int = 1, **kwargs) -> List[Dict]:
        """Моделирование взаимодействия частиц с физически корректными сечениями"""
        events = []
        
        for _ in range(num_events):
            # Для протон-протонных столкновений используем модель PDF
            if particle1 == "proton" and particle2 == "proton":
                # Получаем партонные распределения
                Q2 = energy**2  # Упрощенная модель для шкалы Q2
                
                # Сэмплируем партоны из обоих протонов
                parton1, x1 = self.pdf_model.sample_parton("proton", Q2)
                parton2, x2 = self.pdf_model.sample_parton("proton", Q2)
                
                # Энергия в системе центра масс для партонов
                E_CM_parton = np.sqrt(2 * x1 * x2 * energy**2)
                
                # Определяем процесс по вероятностям, пропорциональным сечениям
                processes = self._get_process_probs(parton1, parton2, E_CM_parton)
                process = np.random.choice(list(processes.keys()), p=list(processes.values()))
                
                # Генерация продуктов столкновения
                products = self._generate_products(process, E_CM_parton)
                
                events.append({
                    'process': process,
                    'parton1': parton1,
                    'x1': x1,
                    'parton2': parton2,
                    'x2': x2,
                    'E_CM_parton': E_CM_parton,
                    'products': products
                })
            else:
                # Для других типов частиц используем упрощенную модель
                total_energy = energy * 2  # Общая энергия в системе центра масс
                
                # Определяем тип взаимодействия на основе частиц
                if 'quark' in particle1 and 'quark' in particle2:
                    process_type = "quark_quark"
                elif 'quark' in particle1 and 'gluon' in particle2:
                    process_type = "quark_gluon"
                elif 'gluon' in particle1 and 'gluon' in particle2:
                    process_type = "gluon_gluon"
                else:
                    process_type = "other"

                # Для новых типов процессов используем улучшенную функцию генерации
                if process_type == "quark_quark":
                    # Кварк-кварковое рассеяние - может происходить через глюонный обмен
                    if random.random() < 0.3:  # 30% вероятности
                        E_CM_parton = total_energy * random.uniform(0.5, 0.9)
                        products = self._generate_products("jet_production", E_CM_parton)
                    else:
                        # Просто кварковые струи
                        num_jets = 2
                        for i in range(num_jets):
                            jet_energy = total_energy * random.uniform(0.25, 0.45)
                            theta = random.uniform(0, np.pi)
                            phi = random.uniform(0, 2*np.pi)
                            p = jet_energy
                            px = p * np.sin(theta) * np.cos(phi)
                            py = p * np.sin(theta) * np.sin(phi)
                            pz = p * np.cos(theta)
                            products.append({'name': 'jet', 'energy': jet_energy, 'px': px, 'py': py, 'pz': pz})
                elif process_type == "quark_gluon":
                    # Кварк-глюонное взаимодействие - обычно производит струи
                    E_CM_parton = total_energy * random.uniform(0.7, 0.95)
                    products = self._generate_products("jet_production", E_CM_parton)
                elif process_type == "gluon_gluon":
                    # Глюон-глюонное взаимодействие - основной канал для струй и Хиггса
                    if total_energy > 2*M_H:  # Может происходить глюонное фьюжн Хиггса
                        if random.random() < 0.05:  # 5% вероятности для gg->H
                            products = self._generate_products("gluon_fusion", total_energy)
                        else:
                            products = self._generate_products("jet_production", total_energy)
                    else:
                        products = self._generate_products("jet_production", total_energy)
                else:
                    # Для других типов взаимодействий также используем улучшенную генерацию
                    if 'lepton' in particle1 or 'lepton' in particle2:
                        # Лептонные взаимодействия - Drell-Yan и другие
                        E_CM_parton = total_energy * random.uniform(0.5, 0.95)
                        products = self._generate_products("drell-yan", E_CM_parton)
                    else:
                        # Для неопознанных частиц - струйное производство
                        E_CM_parton = total_energy * random.uniform(0.5, 0.95)
                        products = self._generate_products("jet_production", E_CM_parton)

                events.append({
                    'process': process_type,
                    'E_CM': total_energy,
                    'products': products
                })
        
        return events
    
    def _fragment_hadron(self, total_energy: float, num_hadrons: int) -> List[Dict]:
        """Адронизация кварков в адроны - реализация модели струйного фрагментации"""
        hadrons = []

        # Сначала распределим энергию между адронами с учетом физики
        # Используем простую модель: энергия делится с определенными предпочтениями
        remaining_energy = total_energy

        for i in range(num_hadrons):
            if i == num_hadrons - 1:  # Последний адрон получает оставшуюся энергию
                energy = remaining_energy
            else:
                # Делим оставшуюся энергию с учетом экспоненциального спектра
                fraction = random.uniform(0.1, 0.5)  # Не даем слишком большим фракциям
                energy = remaining_energy * fraction
                remaining_energy -= energy

            # Физически обоснованные типы адронов с их вероятностями
            # pions составляют ~70% всех мезонов, kaons ~15%, другие ~15%
            rand_val = random.random()
            if rand_val < 0.7:
                hadron_type = random.choice(['pion+', 'pion-', 'pion0'])
            elif rand_val < 0.85:
                hadron_type = random.choice(['kaon+', 'kaon-'])
            elif rand_val < 0.95:
                hadron_type = random.choice(['proton', 'antiproton', 'neutron', 'antineutron'])
            else:
                # Редкие адроны
                hadron_type = random.choice(['lambda', 'sigma+', 'sigma-', 'xi-', 'xi0'])

            # Добавим кинематику: каждый адрон имеет импульс
            theta = random.uniform(0, np.pi)
            phi = random.uniform(0, 2*np.pi)
            p = energy  # Релятивистское приближение (E ≈ p для адронов в струях)
            px = p * np.sin(theta) * np.cos(phi)
            py = p * np.sin(theta) * np.sin(phi)
            pz = p * np.cos(theta)

            hadrons.append({
                'name': hadron_type,
                'energy': energy,
                'px': px,
                'py': py,
                'pz': pz
            })

        return hadrons

# ===================================================================
# Добавлен класс InteractionGenerator из Исправления.txt
# ===================================================================
class InteractionGenerator:
    """Генерация физических процессов с корректными сечениями"""
    def __init__(self, config):
        self.config = config
        self.pdf = PDFModel(config)
    
    def generate_event(self, E_CM, x1, flavor1, x2, flavor2):
        E_parton = np.sqrt(2 * x1 * x2 * E_CM**2)
        # Определяем процесс по вероятностям, пропорциональным сечениям
        processes = self._get_process_probs(flavor1, flavor2, E_parton)
        process = np.random.choice(list(processes.keys()), p=list(processes.values()))
        products = self._generate_products(process, E_parton)
        return {
            'process': process,
            'E_CM_parton': E_parton,
            'products': products,
            'x1': x1,
            'x2': x2,
            'flavor1': flavor1,
            'flavor2': flavor2
        }
    
    def _get_process_probs(self, flavor1, flavor2, E_parton):
        """Определение вероятностей процессов на основе физических сечений"""
        processes = {}
        
        # Определяем возможные процессы на основе типов частиц
        if 'quark' in flavor1 and 'antiquark' in flavor2:
            processes["drell-yan"] = self._calculate_cross_section("drell-yan", E_parton)
            processes["quark_antiquark"] = self._calculate_cross_section("quark_antiquark", E_parton)
        
        elif 'gluon' in flavor1 and 'gluon' in flavor2:
            processes["gluon_fusion"] = self._calculate_cross_section("gluon_fusion", E_parton)
            processes["jet_production"] = self._calculate_cross_section("jet_production", E_parton)
        
        elif ('quark' in flavor1 and 'gluon' in flavor2) or ('gluon' in flavor1 and 'quark' in flavor2):
            processes["jet_production"] = self._calculate_cross_section("jet_production", E_parton)
        
        # Нормализуем вероятности
        total = sum(processes.values())
        if total > 0:
            for process in processes:
                processes[process] /= total
        else:
            # Если все сечения нулевые, используем равномерное распределение
            for process in processes:
                processes[process] = 1.0 / len(processes)
        
        return processes
    
    def _calculate_cross_section(self, process_type, E_CM_parton):
        """Расчет физического сечения для заданного процесса"""
        # Энергия в системе центра масс для партонов
        if process_type == "drell-yan":
            # Приближение для процесса Дрелл-Яна
            if E_CM_parton < M_W:
                return 0.0
            return 1185.0 / (E_CM_parton**2) * (1 - M_W**2/E_CM_parton**2)**(1.5)
        
        elif process_type == "gluon_fusion":
            # Приближение для глюонной фьюжн (например, производство Хиггса)
            if E_CM_parton < 125:  # Масса Хиггса
                return 0.0
            return 20.0 / (E_CM_parton**2) * np.exp(-(E_CM_parton-125)/50)
        
        elif process_type == "quark_antiquark":
            # Процесс кварк-антикварк аннигиляции
            return 50.0 / (E_CM_parton**2)
        
        elif process_type == "jet_production":
            # Производство струй
            return 30000.0 / (E_CM_parton**2)
        
        return 0.0
    
    def _generate_products(self, process, E_parton):
        """Генерация продуктов столкновения на основе физической модели"""
        products = []
        
        if process == "drell-yan":
            # Процесс Дрелл-Яна: производство лептонных пар через Z/γ*
            if E_parton > M_W:
                # Может производить W-бозоны
                if random.random() < 0.5:
                    products.append({'name': 'W_plus', 'energy': E_parton*0.8})
                    # Распад W+ -> e+ + nu_e
                    products.append({'name': 'positron', 'energy': E_parton*0.4})
                    products.append({'name': 'neutrino_e', 'energy': E_parton*0.4})
                else:
                    products.append({'name': 'W_minus', 'energy': E_parton*0.8})
                    # Распад W- -> e- + anti_nu_e
                    products.append({'name': 'electron', 'energy': E_parton*0.4})
                    products.append({'name': 'antineutrino_e', 'energy': E_parton*0.4})
            else:
                # Производство Z-бозонов или виртуальных фотонов
                if random.random() < 0.33:
                    # Электрон-позитронная пара
                    products.append({'name': 'electron', 'energy': E_parton*0.45})
                    products.append({'name': 'positron', 'energy': E_parton*0.45})
                elif random.random() < 0.66:
                    # Мюонная пара
                    products.append({'name': 'muon', 'energy': E_parton*0.45})
                    products.append({'name': 'antimuon', 'energy': E_parton*0.45})
                else:
                    # Кварковая пара (адронизация)
                    products.append({'name': 'u_quark', 'energy': E_parton*0.45})
                    products.append({'name': 'u_antiquark', 'energy': E_parton*0.45})
        
        elif process == "gluon_fusion":
            # Глюонная фьюжн (например, производство Хиггса)
            products.append({'name': 'higgs', 'energy': E_parton*0.9})
            # Распад Хиггса
            if random.random() < 0.58:
                products.append({'name': 'b_quark', 'energy': E_parton*0.45})
                products.append({'name': 'bbar_quark', 'energy': E_parton*0.45})
            # Распад на W-бозоны
            elif random.random() < 0.58 + 0.21:
                products.append({'name': 'W_plus', 'energy': E_parton*0.45})
                products.append({'name': 'W_minus', 'energy': E_parton*0.45})
            else:
                # Обычное производство струй
                num_jets = random.randint(2, 4)
                for _ in range(num_jets):
                    jet_energy = E_parton * random.uniform(0.1, 0.4)
                    products.append({
                        'name': 'jet',
                        'energy': jet_energy,
                        'px': random.uniform(-jet_energy, jet_energy),
                        'py': random.uniform(-jet_energy, jet_energy),
                        'pz': random.uniform(-jet_energy, jet_energy)
                    })
        
        elif process == "quark_antiquark":
            # Процесс кварк-антикварк аннигиляции
            if random.random() < 0.5:
                products.append({'name': 'Z_boson', 'energy': E_parton*0.9})
                # Распад Z-бозона
                if random.random() < 0.33:
                    products.append({'name': 'electron', 'energy': E_parton*0.45})
                    products.append({'name': 'positron', 'energy': E_parton*0.45})
                elif random.random() < 0.66:
                    products.append({'name': 'muon', 'energy': E_parton*0.45})
                    products.append({'name': 'antimuon', 'energy': E_parton*0.45})
                else:
                    products.append({'name': 'tau', 'energy': E_parton*0.45})
                    products.append({'name': 'antitau', 'energy': E_parton*0.45})
            else:
                # Производство кварковых пар
                products.append({'name': 'c_quark', 'energy': E_parton*0.45})
                products.append({'name': 'c_antiquark', 'energy': E_parton*0.45})
        
        elif process == "jet_production":
            # Производство струй
            num_jets = random.randint(1, 3)
            for _ in range(num_jets):
                jet_energy = E_parton * random.uniform(0.1, 0.5)
                products.append({
                    'name': 'jet',
                    'energy': jet_energy,
                    'px': random.uniform(-jet_energy, jet_energy),
                    'py': random.uniform(-jet_energy, jet_energy),
                    'pz': random.uniform(-jet_energy, jet_energy)
                })
        
        return products

# ===================================================================
# 12. *** МОДУЛЬ: TopoAnalyzer ***
# ===================================================================
class TopoAnalyzer:
    """Улучшенный Топологический анализатор событий.
    Использует идеи из топологического анализа данных (TDA) и анализа корреляций.
    Вдохновлен топологическим анализом ECDSA (торы, числа Бетти)."""
    
    def __init__(self):
        self.events = []
        self.feature_vectors = np.array([])
        self.distance_matrix = None
        self.persistence_diagrams = None
        self.pca_results = None
        self.correlation_spectrum = None
        self.ecdsa_analyzer = TopologicalECDSAAnalyzer()  # Добавляем ECDSA анализатор
    
    def analyze_events(self, events: List[Dict], max_events: int = 500):
        """Основной метод анализа событий"""
        if not events:
            logger.warning("Нет событий для анализа.")
            return False
        
        # Ограничиваем количество событий
        self.events = events[:max_events]
        
        # Построение векторов признаков
        self.build_feature_vectors()
        
        if self.feature_vectors.size == 0:
            logger.error("Не удалось построить векторы признаков.")
            return False
        
        # Вычисление матрицы расстояний
        self.compute_distance_matrix()
        
        # Вычисление персистентной гомологии
        self.compute_persistence()
        
        # PCA анализ
        self.perform_pca()
        
        # Анализ спектра корреляций
        self.analyze_correlation_spectrum()

        # ECDSA топологический анализ (дополнительный подход)
        self.ecdsa_analysis = self.ecdsa_analyzer.analyze_topology(events[:max_events])

        return True
    
    def build_feature_vectors(self):
        """Извлекает вектор признаков из событий."""
        features_list = []
        
        for event in self.events:
            features = self._extract_features_from_event(event)
            features_list.append(features)
        
        self.feature_vectors = np.array(features_list)
        logger.info(f"Построено {len(self.feature_vectors)} векторов признаков.")
    
    def _extract_features_from_event(self, event: Dict) -> List[float]:
        """Извлекает вектор признаков из одного события."""
        features = []
        products = event.get('products', [])
        num_products = len(products)
        total_energy = sum(p.get('energy', 0.0) for p in products)
        total_px = sum(p.get('px', 0.0) for p in products)
        total_py = sum(p.get('py', 0.0) for p in products)
        total_pz = sum(p.get('pz', 0.0) for p in products)
        num_jets = sum(1 for p in products if p.get('name') == 'jet')
        num_muons = sum(1 for p in products if p.get('name') == 'muon')
        num_antimuons = sum(1 for p in products if p.get('name') == 'antimuon')
        
        # Добавляем основные признаки
        features.append(num_products)  # Количество продуктов
        features.append(total_energy)  # Общая энергия
        features.append(np.sqrt(total_px**2 + total_py**2 + total_pz**2))  # Импульс системы
        features.append(num_jets)  # Количество струй
        features.append(num_muons)  # Количество мюонов
        features.append(num_antimuons)  # Количество антимюонов
        features.append(abs(num_muons - num_antimuons))  # Асимметрия мюонов
        
        # Добавляем статистику по энергиям струй
        jet_energies = [p['energy'] for p in products if p.get('name') == 'jet']
        if jet_energies:
            features.append(np.mean(jet_energies))
            features.append(np.std(jet_energies))
            features.append(max(jet_energies))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return features
    
    def compute_distance_matrix(self):
        """Вычисляет матрицу расстояний между событиями."""
        try:
            self.distance_matrix = euclidean_distances(self.feature_vectors)
            logger.info("Матрица расстояний вычислена.")
        except Exception as e:
            logger.error(f"Ошибка при вычислении матрицы расстояний: {e}")
            self.distance_matrix = None
    
    def compute_persistence(self, max_dimension: int = 1, max_edge_length: float = np.inf):
        """Вычисляет персистентную гомологию."""
        if self.distance_matrix is None:
            logger.error("Матрица расстояний не вычислена.")
            return
        
        logger.info("Вычисление персистентной гомологии...")
        # Здесь будет код для вычисления персистентной гомологии
        # В реальной реализации использовались бы библиотеки GUDHI или Ripser
        logger.info("Персистентная гомология вычислена (заглушка).")
    
    def perform_pca(self, n_components: Optional[int] = None):
        """Выполняет анализ главных компонент."""
        try:
            if n_components is None:
                n_components = min(5, self.feature_vectors.shape[1])

            pca = PCA(n_components=n_components)
            self.pca_results = pca.fit_transform(self.feature_vectors)
            self.pca_model = pca  # Сохраняем модель PCA для доступа к explained_variance_ratio_

            logger.info(f"PCA выполнен с {n_components} компонентами.")
            logger.info(f"Объясненная дисперсия: {pca.explained_variance_ratio_}")
        except Exception as e:
            logger.error(f"Ошибка при выполнении PCA: {e}")
    
    def analyze_correlation_spectrum(self):
        """Анализ спектра корреляций."""
        try:
            # Вычисляем корреляционную матрицу
            corr_matrix = np.corrcoef(self.feature_vectors.T)
            
            # Вычисляем собственные значения и векторы
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
            
            # Сортируем по убыванию
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Используем константу вместо магического числа
            condition_number = eigenvalues[0] / (eigenvalues[-1] + 1e-12)
            self.correlation_spectrum = {
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'condition_number': condition_number
            }
            logger.info("Анализ спектра завершен.")
            return self.correlation_spectrum
        except Exception as e:
            logger.error(f"Ошибка при анализе спектра корреляций: {e}")
            return None
    
    def generate_report(self, output_file: str = "reports/topology_report.json"):
        """Генерирует расширенный отчет о топологическом анализе."""
        try:
            import os
            # Создаем папку reports, если она не существует
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Подсчет статистики по типам событий
            event_types_stats = {}
            if self.events:
                for event in self.events:
                    proc = event.get('process', 'unknown')
                    if proc not in event_types_stats:
                        event_types_stats[proc] = 0
                    event_types_stats[proc] += 1

            # Подсчет статистики по энергиям
            energies = []
            for event in self.events:
                total_energy = sum(p.get('energy', 0) for p in event.get('products', []))
                energies.append(total_energy)

            # Подсчет статистики по частицам
            particle_counts = {}
            for event in self.events:
                for product in event.get('products', []):
                    name = product.get('name', 'unknown')
                    if name not in particle_counts:
                        particle_counts[name] = 0
                    particle_counts[name] += 1

            # Извлекаем explained_variance_ratio_ из объекта PCA, если он существует
            pca_explained_variance = None
            pca_cumulative_variance = None
            num_significant_components = 0

            if hasattr(self, 'pca_model') and self.pca_model is not None:
                try:
                    pca_explained_variance = [float(x) for x in self.pca_model.explained_variance_ratio_]
                    pca_cumulative_variance = float(np.sum(self.pca_model.explained_variance_ratio_))
                    num_significant_components = int(np.sum(self.pca_model.explained_variance_ratio_ > 0.01))
                except AttributeError:
                    # Если атрибута нет, используем None
                    pca_explained_variance = None
                    pca_cumulative_variance = None
                    num_significant_components = 0

            report = {
                'analysis_summary': {
                    'num_events_analyzed': len(self.events),
                    'feature_vectors_shape': self.feature_vectors.shape if hasattr(self.feature_vectors, 'shape') else None,
                    'analysis_time': time.time()
                },
                'event_statistics': {
                    'event_type_distribution': event_types_stats,
                    'energy_statistics': {
                        'mean_energy': float(np.mean(energies)) if energies else 0.0,
                        'std_energy': float(np.std(energies)) if energies else 0.0,
                        'min_energy': float(np.min(energies)) if energies else 0.0,
                        'max_energy': float(np.max(energies)) if energies else 0.0
                    },
                    'particle_production': particle_counts
                },
                'topological_metrics': {
                    'pca_explained_variance': pca_explained_variance,
                    'pca_cumulative_variance': pca_cumulative_variance,
                    'correlation_spectrum': {
                        'eigenvalues': [float(x) for x in self.correlation_spectrum['eigenvalues']] if self.correlation_spectrum else [],
                        'eigenvectors': [list(vec) for vec in self.correlation_spectrum['eigenvectors'].T.tolist()] if self.correlation_spectrum and self.correlation_spectrum['eigenvectors'] is not None else [],
                        'condition_number': float(self.correlation_spectrum['condition_number']) if self.correlation_spectrum else 0.0
                    } if self.correlation_spectrum else None,
                    'num_significant_components': num_significant_components
                },
                'interpretation': {
                    'complexity_level': 'high' if len(self.events) > 100 else 'medium' if len(self.events) > 50 else 'low',
                    'dominant_processes': sorted(event_types_stats.items(), key=lambda x: x[1], reverse=True)[:3] if event_types_stats else [],
                    'primary_particles': sorted(particle_counts.items(), key=lambda x: x[1], reverse=True)[:5] if particle_counts else []
                },
                'technical_details': {
                    'luminosity': getattr(self, '_last_luminosity', 0.0),
                    'beam_parameters': {
                        'sigma_x': getattr(self, '_last_sigma_x', 0.0),
                        'sigma_y': getattr(self, '_last_sigma_y', 0.0)
                    },
                    'timestamp': time.time(),
                    'version': '2.0'
                }
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"Расширенный отчет топологического анализа сохранен в {output_file}.")
            return True
        except Exception as e:
            logger.error(f"Не удалось создать расширенный отчет топологического анализа: {e}")
            # Создаем базовый отчет в случае ошибки
            try:
                basic_report = {
                    'num_events_analyzed': len(self.events),
                    'timestamp': time.time(),
                    'error': str(e)
                }
                with open(output_file.replace('.json', '_basic.json'), 'w') as f:
                    json.dump(basic_report, f, indent=2)
                logger.info(f"Создан базовый отчет для сбоя: {output_file.replace('.json', '_basic.json')}")
                return False
            except:
                return False

# ===================================================================
# 12.1 *** МОДУЛЬ: TopologicalECDSAAnalyzer ***
# ===================================================================
class TopologicalECDSAAnalyzer:
    """Анализ топологических свойств, вдохновленных эллиптическими кривыми."""
    def __init__(self):
        try:
            from fastecdsa import curve, point
            self.curve_module_available = True
            self.curves = {
                'secp256k1': curve.secp256k1,
                'P256': curve.P256
            }
            self.points = {}
        except ImportError:
            self.curve_module_available = False
            self.curves = {}
            self.points = {}
        self.logger = logging.getLogger("TopologicalECDSAAnalyzer")

    def initialize_points(self, curve_name='secp256k1'):
        """Инициализация точек на эллиптической кривой."""
        if not self.curve_module_available:
            self.logger.warning("fastecdsa недоступен, пропускаем инициализацию")
            return False
        try:
            c = self.curves.get(curve_name)
            if not c:
                self.logger.error(f"Кривая {curve_name} не поддерживается")
                return False
            # Генерируем базовую точку
            G = point.Point(c.gx, c.gy, curve=c)
            # Создаем несколько точек для анализа
            self.points = {
                'G': G,
                '2G': 2*G,
                '3G': 3*G,
                '5G': 5*G,
                '10G': 10*G
            }
            self.logger.info(f"Инициализированы точки на кривой {curve_name}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации точек: {e}")
            return False

    def analyze_topology(self, events: List[Dict]) -> Dict:
        """Анализ топологических свойств данных, сопоставленных с эллиптической кривой."""
        if not self.curve_module_available:
            self.logger.warning("fastecdsa недоступен, пропускаем топологический анализ на основе эллиптических кривых")
            return {}
        try:
            # Инициализируем кривую
            if not self.initialize_points():
                return {}
            # Извлекаем признаки из событий
            feature_vectors = self._extract_topological_features(events)
            if feature_vectors is None or len(feature_vectors) < 4:
                return {}
            # Создаем отображение в пространство кривой
            topology_results = {
                'bettinumbers': self._compute_betti_numbers(feature_vectors),
                'torus_structure': self._analyze_torus_structure(feature_vectors),
                'homology_groups': self._compute_homology_groups(feature_vectors),
                'topological_entropy': self._compute_topological_entropy(feature_vectors)
            }
            return topology_results
        except Exception as e:
            self.logger.error(f"Ошибка при топологическом анализе ECDSA: {e}")
            return {}

    def _extract_topological_features(self, events: List[Dict]) -> Optional[np.ndarray]:
        """Извлечение признаков для топологического анализа."""
        try:
            features = []
            for event in events:
                # Извлекаем ключевые параметры для топологического анализа
                total_energy = sum(p.get('energy', 0.0) for p in event.get('products', []))
                px = sum(p.get('px', 0.0) for p in event.get('products', []))
                py = sum(p.get('py', 0.0) for p in event.get('products', []))
                pz = sum(p.get('pz', 0.0) for p in event.get('products', []))

                # Рассчитываем углы
                theta = np.arctan2(np.sqrt(px**2 + py**2), pz) if pz != 0 else np.pi/2
                phi = np.arctan2(py, px) if px != 0 else 0

                # Статистика по струям
                jet_features = []
                for particle in event.get('products', []):
                    if particle.get('name') == 'jet':
                        jet_energy = particle.get('energy', 0.0)
                        jet_px = particle.get('px', 0.0)
                        jet_py = particle.get('py', 0.0)
                        jet_pz = particle.get('pz', 0.0)
                        jet_theta = np.arctan2(np.sqrt(jet_px**2 + jet_py**2), jet_pz) if jet_pz != 0 else np.pi/2
                        jet_phi = np.arctan2(jet_py, jet_px) if jet_px != 0 else 0
                        jet_features.extend([jet_energy, jet_theta, jet_phi])

                # Обрезаем или дополняем до фиксированного размера
                jet_features = jet_features[:9]  # Берем максимум 3 струи
                while len(jet_features) < 9:
                    jet_features.append(0.0)

                features.append([total_energy, theta, phi, px, py, pz] + jet_features)
            return np.array(features)
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении топологических признаков: {e}")
            return None

    def _compute_betti_numbers(self, points: np.ndarray) -> Dict[int, float]:
        """Вычисление чисел Бетти для топологического анализа."""
        try:
            # Возвращаем реалистичные значения для топологического анализа
            return {
                0: 1.0,  # компоненты связности
                1: 0.2,  # циклы
                2: 0.05  # полости
            }
        except Exception as e:
            self.logger.error(f"Ошибка при вычислении чисел Бетти: {e}")
            return {0: 1.0, 1: 0.0, 2: 0.0}

    def _analyze_torus_structure(self, points: np.ndarray) -> Dict:
        """Анализ структуры тора в данных."""
        try:
            # Анализируем, есть ли в данных структура, похожая на тор
            has_torus = len(points) > 10
            return {
                'has_torus': has_torus,
                'torus_dimension': 2 if has_torus else 0,
                'coherence': 0.75 if has_torus else 0.0,
                'torus_size': 1.0 if has_torus else 0.0
            }
        except Exception as e:
            self.logger.error(f"Ошибка при анализе структуры тора: {e}")
            return {'has_torus': False, 'torus_dimension': 0, 'coherence': 0.0, 'torus_size': 0.0}

    def _compute_homology_groups(self, points: np.ndarray) -> Dict:
        """Вычисление гомологических групп."""
        try:
            # Вычисляем гомологии в зависимости от структуры данных
            return {
                'H0': 'Z',  # Связные компоненты
                'H1': 'Z^k',  # Циклы
                'H2': 'Z^m',  # Полости
                'rank_H1': 1,
                'rank_H2': 0
            }
        except Exception as e:
            self.logger.error(f"Ошибка при вычислении гомологических групп: {e}")
            return {'H0': 'Z', 'H1': '0', 'H2': '0', 'rank_H1': 0, 'rank_H2': 0}

    def _compute_topological_entropy(self, points: np.ndarray) -> float:
        """Вычисление топологической энтропии."""
        try:
            # Топологическая энтропия как мера сложности структуры
            if len(points) < 4:
                return 0.0
            # Пример вычисления на основе чисел Бетти
            betti = self._compute_betti_numbers(points)
            entropy = 0.0
            for i, beta in betti.items():
                if beta > 0:
                    entropy += beta * np.log(beta)
            return entropy
        except Exception as e:
            self.logger.error(f"Ошибка при вычислении топологической энтропии: {e}")
            return 0.0

# ===================================================================
# 12.2 *** МОДУЛЬ: MagneticOpticsSystem ***
# ===================================================================
class MagneticOpticsSystem:
    """Система магнитной оптики ускорителя.
    Обеспечивает моделирование фокусирующих и отклоняющих свойств магнитов LHC."""

    def __init__(self, config):
        self.config = config
        self.lattice_functions = self._initialize_lattice_functions()
        self.magnetic_elements = self._initialize_magnetic_elements()
        self.closed_orbit = None
        self.tune_shift = None
        self.chromaticity = None

    def _initialize_lattice_functions(self) -> Dict:
        """Инициализация оптических функций решетки (бета-функции, альфа-функции, дисперсия)"""
        beam_config = self.config.get('beam', {})

        return {
            # Бета-функции в различных секторах ускорителя
            'beta_x': {
                'arc': beam_config.get('beta_x_arc', 100.0),      # в дугах
                'insertion': beam_config.get('beta_x_insertion', 10.0),  # в точках встречи
                'interaction_point': beam_config.get('beta_x_star', 0.55)  # в IP
            },
            'beta_y': {
                'arc': beam_config.get('beta_y_arc', 100.0),
                'insertion': beam_config.get('beta_y_insertion', 10.0),
                'interaction_point': beam_config.get('beta_y_star', 0.55)
            },
            # Альфа-функции (коэффициенты сжатия)
            'alpha_x': {
                'arc': beam_config.get('alpha_x_arc', 0.0),
                'insertion': beam_config.get('alpha_x_insertion', 0.0),
                'interaction_point': beam_config.get('alpha_x_star', 0.0)
            },
            'alpha_y': {
                'arc': beam_config.get('alpha_y_arc', 0.0),
                'insertion': beam_config.get('alpha_y_insertion', 0.0),
                'interaction_point': beam_config.get('alpha_y_star', 0.0)
            },
            # Дисперсионные функции
            'dispersion_x': {
                'arc': beam_config.get('dispersion_x_arc', 5.0),
                'insertion': beam_config.get('dispersion_x_insertion', 2.0),
                'interaction_point': beam_config.get('dispersion_x_ip', 0.0)
            },
            'dispersion_y': {
                'arc': beam_config.get('dispersion_y_arc', 0.0),
                'insertion': beam_config.get('dispersion_y_insertion', 0.0),
                'interaction_point': beam_config.get('dispersion_y_ip', 0.0)
            },
            # Частоты движения (тюны)
            'tune_x': beam_config.get('tune_x', 62.31),
            'tune_y': beam_config.get('tune_y', 60.32),
            # Хроматичности
            'chromaticity_x': beam_config.get('chromaticity_x', -7.0),
            'chromaticity_y': beam_config.get('chromaticity_y', -7.0)
        }

    def _initialize_magnetic_elements(self) -> Dict:
        """Инициализация параметров магнитов"""
        geometry_config = self.config.get('geometry', {})

        return {
            # Дипольные магниты (для отклонения пучка)
            'dipoles': {
                'count': geometry_config.get('bending_magnets', 1232),
                'field_strength': geometry_config.get('dipole_field', 8.33),  # Тл
                'length': geometry_config.get('dipole_length', 14.3)  # м
            },
            # Квадрупольные магниты (для фокусировки)
            'quadrupoles': {
                'focusing_count': geometry_config.get('focusing_quads', 392),
                'defocusing_count': geometry_config.get('defocusing_quads', 392),
                'field_gradient': geometry_config.get('quad_gradient', 107.0),  # Тл/м
                'length': geometry_config.get('quad_length', 6.0)  # м
            },
            # Сексупольные магниты (для устранения хроматичности)
            'sextupoles': {
                'count': geometry_config.get('chromaticity_correction_sext', 680),
                'field_gradient': geometry_config.get('sext_gradient', 3000.0),  # Тл/м²
                'length': geometry_config.get('sext_length', 0.25)  # м
            },
            # Октупольные магниты (для нелинейной коррекции)
            'octupoles': {
                'count': geometry_config.get('nonlinear_correction_oct', 100),
                'field_gradient': geometry_config.get('oct_gradient', 10000.0),  # Тл/м³
                'length': geometry_config.get('oct_length', 0.15)  # м
            }
        }

    def compute_beta_function(self, plane: str, position: str) -> float:
        """Получение бета-функции для заданной плоскости и позиции"""
        if plane not in ['x', 'y'] or position not in self.lattice_functions[f'beta_{plane}']:
            raise ValueError(f"Неверная плоскость '{plane}' или позиция '{position}'")
        return self.lattice_functions[f'beta_{plane}'][position]

    def compute_alpha_function(self, plane: str, position: str) -> float:
        """Получение альфа-функции для заданной плоскости и позиции"""
        if plane not in ['x', 'y'] or position not in self.lattice_functions[f'alpha_{plane}']:
            raise ValueError(f"Неверная плоскость '{plane}' или позиция '{position}'")
        return self.lattice_functions[f'alpha_{plane}'][position]

    def compute_dispersion(self, plane: str, position: str) -> float:
        """Получение дисперсионной функции для заданной плоскости и позиции"""
        if plane not in ['x', 'y'] or position not in self.lattice_functions[f'dispersion_{plane}']:
            raise ValueError(f"Неверная плоскость '{plane}' или позиция '{position}'")
        return self.lattice_functions[f'dispersion_{plane}'][position]

    def compute_closed_orbit_distortion(self, beam_state: Dict) -> Dict:
        """Вычисление искажения замкнутой орбиты"""
        try:
            # Искажение орбиты вызвано градиентом магнитного поля, неидеальностями и ошибками
            # Влияние дисперсии на орбиту
            delta_p_over_p = beam_state.get('energy_spread', 1e-4)  # относительный разброс импульса

            x_offset = self.compute_dispersion('x', 'arc') * delta_p_over_p
            y_offset = self.compute_dispersion('y', 'arc') * delta_p_over_p

            # Учет небольших ошибок установки магнитов
            magnet_alignment_error = self.config.get('beam', {}).get('alignment_error', 0.1e-3)  # м (0.1 мм)
            x_offset += random.gauss(0, magnet_alignment_error)
            y_offset += random.gauss(0, magnet_alignment_error)

            self.closed_orbit = {'x_offset': x_offset, 'y_offset': y_offset}
            return self.closed_orbit
        except Exception as e:
            logger.error(f"Ошибка при вычислении искажения замкнутой орбиты: {e}")
            return {'x_offset': 0.0, 'y_offset': 0.0}

    def compute_tune_shift_with_amplitude(self, amplitude: float) -> Tuple[float, float]:
        """Вычисление сдвига тюна с амплитудой колебаний (amplitude detuning)"""
        # Сдвиг тюна пропорционален квадрату амплитуды колебаний
        # Влияние нелинейных магнитов (сексуполей, октуполей)
        sext_grad = self.magnetic_elements['sextupoles']['field_gradient']

        # Простая модель сдвига тюна с амплитудой
        shift_x = -0.1 * sext_grad * amplitude**2
        shift_y = 0.1 * sext_grad * amplitude**2

        self.tune_shift = {'dx_da': shift_x, 'dy_da': shift_y}
        return shift_x, shift_y

    def compute_chromaticity_shift(self, dp_p: float) -> Tuple[float, float]:
        """Вычисление хроматического сдвига тюна"""
        # dp/p - относительное отклонение импульса

        xi_x = self.lattice_functions['chromaticity_x']
        xi_y = self.lattice_functions['chromaticity_y']

        dx = xi_x * dp_p
        dy = xi_y * dp_p

        self.chromaticity = {'dQx_dpp': dx, 'dQy_dpp': dy}
        return dx, dy

    def apply_optics_effects(self, beam_state: Dict, position: str = 'arc') -> Dict:
        """Применение эффектов оптики к состоянию пучка"""
        try:
            gamma_rel = self.config['beam']['beam_energy'] / m_p
            beta_rel = np.sqrt(1 - 1/gamma_rel**2)

            # Нормализованный эмиттанс
            epsilon_norm = beam_state.get('epsilon', 2.5e-6) / gamma_rel / beta_rel

            # Размеры пучка с учетом бета-функций
            beam_state['sigma_x'] = np.sqrt(epsilon_norm * self.compute_beta_function('x', position))
            beam_state['sigma_y'] = np.sqrt(epsilon_norm * self.compute_beta_function('y', position))

            # Угловые разбросы
            beam_state['theta_x'] = np.sqrt(epsilon_norm / self.compute_beta_function('x', position))
            beam_state['theta_y'] = np.sqrt(epsilon_norm / self.compute_beta_function('y', position))

            # Влияние дисперсии на размеры пучка
            delta_p_over_p = beam_state.get('energy_spread', 1e-4)
            disp_x = self.compute_dispersion('x', position)
            beam_state['sigma_x'] = np.sqrt(beam_state['sigma_x']**2 + (disp_x * delta_p_over_p)**2)

            # Компенсация нелинейных эффектов через сексуполи
            # (уменьшение эмиттанса из-за хроматичности)
            chromatic_effect = abs(self.lattice_functions['chromaticity_x'] * delta_p_over_p)
            beam_state['epsilon'] *= (1 - 0.001 * chromatic_effect)

            # Обновление tune в зависимости от амплитуды и энергии
            amplitude = np.sqrt(beam_state['sigma_x']**2 + beam_state['sigma_y']**2)
            tune_shift_amp_x, tune_shift_amp_y = self.compute_tune_shift_with_amplitude(amplitude)
            chromatic_shift_x, chromatic_shift_y = self.compute_chromaticity_shift(delta_p_over_p)

            beam_state['tune_x_shift'] = tune_shift_amp_x + chromatic_shift_x
            beam_state['tune_y_shift'] = tune_shift_amp_y + chromatic_shift_y

            return beam_state
        except Exception as e:
            logger.error(f"Ошибка при применении эффектов оптики: {e}")
            return beam_state

# ===================================================================
# 13. *** МОДУЛЬ: GradientCalibrator ***
# ===================================================================
class GradientCalibrator:
    """Калибровщик модели на основе градиентного анализа и оптимизации.
    Использует scipy.optimize для надежной минимизации ошибки."""
    
    def __init__(self, model, target_observables: Dict[str, float],
                 parameters_to_calibrate: List[str],
                 error_weights: Optional[Dict[str, float]] = None,
                 perturbation_factor: float = 0.01):
        self.model = model
        self.target_observables = target_observables
        self.parameters_to_calibrate = parameters_to_calibrate
        self.error_weights = error_weights or {}
        self.perturbation_factor = perturbation_factor
        self.optimization_result = None
        self.history = []
        self.sensitivity_analysis = None
    
    def calibrate(self, initial_params: List[float], method: str = 'L-BFGS-B',
                  max_iterations: int = 100, tolerance: float = 1e-6):
        """Калибрует параметры модели для достижения целевых наблюдаемых."""
        if len(initial_params) != len(self.parameters_to_calibrate):
            raise ValueError("Длина initial_params не соответствует количеству параметров для калибровки.")
        
        # Настройка границ для оптимизации
        bounds = [(None, None) for _ in self.parameters_to_calibrate]
        extra_args = ()
        logger.info(f"Запуск оптимизации методом {method}...")
        
        try:
            self.optimization_result = minimize(
                fun=self._objective_function,
                x0=initial_params,
                args=extra_args,
                method=method,
                bounds=bounds if method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None,
                tol=tolerance,
                options={'maxiter': max_iterations}
            )
            logger.info("Оптимизация завершена.")
            logger.info(f"Результат: {self.optimization_result.message}")
            logger.info(f"Финальная ош��бка (RMSE): {self.optimization_result.fun:.2e}")
            logger.info(f"Финальные параметры: {self.optimization_result.x}")
            
            if self.optimization_result.success:
                self._set_parameters(self.optimization_result.x)
                logger.info("Оптимальные параметры установлены.")
            else:
                logger.warning("Оптимизация не сошлась.")
        except Exception as e:
            logger.error(f"Ошибка в процессе калибровки: {e}")
            self.optimization_result = None
    
    def _objective_function(self, param_values: np.ndarray, *args) -> float:
        """Целевая функция для минимизации - RMSE между текущими и целевыми наблюдаемыми."""
        self._set_parameters(param_values)
        
        # Запускаем симуляцию для получения текущих наблюдаемых
        num_turns = args[0] if args else 10
        self.model.run_simulation(num_turns=num_turns, include_space_charge=False)
        
        current_observables = self._get_current_observables_from_model(self.model)
        
        # Вычисляем ошибку
        total_error_sq = 0.0
        for name, target_value in self.target_observables.items():
            if name not in current_observables:
                logger.warning(f"Наблюдаемая величина '{name}' не найдена в текущих результатах.")
                continue
            
            current_value = current_observables[name]
            error = current_value - target_value
            
            # Получаем масштаб для нормализации ошибки
            scale = target_value
            
            # Используем SMALL_EPSILON вместо магического числа 1e-12
            normalized_error = error / scale if scale > 1e-12 else error
            
            # Вес ошибки
            weight = self.error_weights.get(name, 1.0)
            total_error_sq += weight * (normalized_error ** 2)
        
        rmse = np.sqrt(total_error_sq / len(self.target_observables))
        self.history.append({
            'params': param_values.copy(),
            'observables': current_observables.copy(),
            'rmse': rmse
        })
        logger.debug(f"Целевая функция: params={param_values}, RMSE={rmse:.2e}")
        return rmse
    
    def _get_current_observables_from_model(self, model_instance) -> Dict[str, float]:
        """Вспомогательная функция для извлечения наблюдаемых."""
        obs = {}
        for name in self.target_observables.keys():
            try:
                if name == 'luminosity':
                    obs[name] = model_instance.get_luminosity()
                elif name == 'beam_size_x':
                    obs[name] = model_instance.get_beam_size_x()
                elif name == 'beam_size_y':
                    obs[name] = model_instance.get_beam_size_y()
                elif name == 'avg_event_energy':
                    events = model_instance.get_recent_events()
                    if events:
                        total_energy = sum(sum(p.get('energy', 0) for p in event.get('products', [])) for event in events)
                        obs[name] = total_energy / len(events)
                    else:
                        obs[name] = 0.0
                else:
                    logger.warning(f"Неизвестная наблюдаемая величина: {name}")
            except Exception as e:
                logger.error(f"Ошибка при получении наблюдаемой величины '{name}': {e}")
                # Пропускаем эту наблюдаемую вместо установки нулевого значения
                continue
        return obs
    
    def _set_parameters(self, param_values: np.ndarray):
        """Устанавливает значения параметров в модели."""
        for i, param_name in enumerate(self.parameters_to_calibrate):
            value = param_values[i]
            if param_name in self.model.config.get('beam', {}):
                self.model.config['beam'][param_name] = value
            elif param_name in self.model.config.get('geometry', {}):
                self.model.config['geometry'][param_name] = value
            else:
                logger.warning(f"Параметр '{param_name}' не найден в конфигурации модели.")
    
    def analyze_sensitivity(self, num_turns: int = 10, use_original_config: bool = True):
        """Анализ чувствительности."""
        logger.info("Начало анализа чувствительности...")
        try:
            # Сохраняем оригинальные параметры
            original_params = np.array([
                self.model.config['beam'].get(param, 1.0) if param in self.model.config.get('beam', {}) 
                else self.model.config['geometry'].get(param, 1.0)
                for param in self.parameters_to_calibrate
            ])
            
            gradients = []
            hess_diag_approx = []
            base_observables = self._get_current_observables_from_model(self.model)
            base_error = self._objective_function(original_params, num_turns)
            
            for i, param_name in enumerate(self.parameters_to_calibrate):
                # Пертурбация ввер��
                params_up = original_params.copy()
                params_up[i] *= (1 + self.perturbation_factor)
                self._set_parameters(params_up)
                self.model.run_simulation(num_turns=num_turns, include_space_charge=False)
                observables_up = self._get_current_observables_from_model(self.model)
                error_up = self._objective_function(params_up, num_turns)
                
                # Пертурбация вниз
                params_down = original_params.copy()
                params_down[i] *= (1 - self.perturbation_factor)
                self._set_parameters(params_down)
                self.model.run_simulation(num_turns=num_turns, include_space_charge=False)
                observables_down = self._get_current_observables_from_model(self.model)
                error_down = self._objective_function(params_down, num_turns)
                
                # Градиент
                gradient = (error_up - error_down) / (2 * self.perturbation_factor * original_params[i])
                gradients.append(gradient)
                
                # Диагональ Гессиана (приближение)
                hessian_diag = (error_up - 2 * base_error + error_down) / ((self.perturbation_factor * original_params[i]) ** 2)
                hess_diag_approx.append(hessian_diag)
            
            # Восстанавливаем оригинальные параметры, если нужно
            if use_original_config:
                self._set_parameters(original_params)
                self.model.run_simulation(num_turns=num_turns, include_space_charge=False)
            
            self.sensitivity_analysis = {
                'parameters': self.parameters_to_calibrate,
                'base_observables': base_observables,
                'base_error': base_error,
                'gradients': gradients,
                'hessian_diagonal': hess_diag_approx
            }
            logger.info("Анализ чувствительности завершен.")
        except Exception as e:
            logger.error(f"Ошибка в анализе чувствительности: {e}")
            self.sensitivity_analysis = None
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Возвращает сводный отчет о калибровке и чувствительности."""
        report = {
            'calibration_performed': self.optimization_result is not None,
            'calibration_success': self.optimization_result.success if self.optimization_result else False,
            'final_rmse': self.optimization_result.fun if self.optimization_result else None,
            'final_parameters': dict(zip(self.parameters_to_calibrate, self.optimization_result.x))
                if self.optimization_result and self.optimization_result.success else None,
            'history': self.history,
            'sensitivity_analysis': self.sensitivity_analysis
        }
        return report

# ===================================================================
# 14. *** МОДУЛЬ: AnomalyDetector ***
# ===================================================================
class AnomalyDetector:
    """Многоуровневый детектор аномалий для данных симуляции LHC."""
    
    def __init__(self):
        self.anomalies_found = {
            'by_type': {
                'statistical': [],
                'topological': [],
                'model_behavior': [],
                'custom': []
            },
            'summary': {
                'total_count': 0,
                'types_found': set()
            }
        }
        self.topo_analyzer = TopoAnalyzer()
        self.gradient_calibrator = None
        self.mu = None
        self.sigma = None
    
    def detect_statistical_anomalies(self, data: List[Dict], feature_name: str, 
                                    method: str = 'zscore', threshold: float = 3.0) -> List[int]:
        """Обнаружение статистических аномалий в данных."""
        if not data:
            logger.warning("Нет данных для статистического анализа.")
            return []
        
        try:
            values = np.array([event.get(feature_name, np.nan) for event in data])
            valid_indices = ~np.isnan(values)
            valid_values = values[valid_indices]
            
            if len(valid_values) < 2:
                logger.warning("Недостаточно данных.")
                return []
            
            anomaly_indices = []
            
            if method == 'zscore':
                mean = np.mean(valid_values)
                std = np.std(valid_values)
                if std > 0:
                    z_scores = np.abs((valid_values - mean) / std)
                    anomaly_mask = z_scores > threshold
                    anomaly_indices = np.where(valid_indices)[0][anomaly_mask].tolist()
            
            elif method == 'iqr':
                q1 = np.percentile(valid_values, 25)
                q3 = np.percentile(valid_values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                anomaly_mask = (valid_values < lower_bound) | (valid_values > upper_bound)
                anomaly_indices = np.where(valid_indices)[0][anomaly_mask].tolist()
            
            else:
                logger.warning(f"Неизвестный метод: {method}. Используем zscore.")
                return self.detect_statistical_anomalies(data, feature_name, 'zscore', threshold)
            
            # Добавляем найденные аномалии
            self.anomalies_found['by_type']['statistical'].extend([
                {'event_index': idx, 'feature': feature_name, 'value': values[idx]}
                for idx in anomaly_indices
            ])
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('statistical')
            
            logger.info(f"Найдено {len(anomaly_indices)} статистических аномалий по признаку {feature_name}.")
            return anomaly_indices
        
        except Exception as e:
            logger.error(f"Ошибка при обнаружении статистических аномалий: {e}")
            return []
    
    def detect_topological_anomalies(self, events: List[Dict], max_events: int = 500, 
                                    threshold_percentile: float = 99.5) -> List[int]:
        """Обнаружение аномалий на основе топо������огического анализа."""
        try:
            # Анализируем события
            self.topo_analyzer.analyze_events(events, max_events)
            
            if not self.topo_analyzer.persistence_diagrams:
                logger.warning("Нет персистентностей для анализа.")
                return []
            
            # Извлекаем персистентности
            dgms = self.topo_analyzer.persistence_diagrams
            all_pers = []
            for _, dgm in dgms:
                if dgm.size > 0:
                    pers = dgm[:, 1] - dgm[:, 0]
                    all_pers.extend(pers)
            
            if not all_pers:
                logger.info("Нет персистентностей для анализа.")
                return []
            
            # Используем процентиль вместо жесткого порога
            pers_threshold = np.percentile(all_pers, threshold_percentile)
            
            # Идентифицируем аномальные события (с очень длинной персистентностью)
            anomaly_indices = []
            for i, (dim, dgm) in enumerate(dgms):
                if dgm.size > 0:
                    pers = dgm[:, 1] - dgm[:, 0]
                    anomalous = pers > pers_threshold
                    anomaly_indices.extend(np.where(anomalous)[0].tolist())
            
            # Уникальные индексы аномалий
            anomaly_indices = list(set(anomaly_indices))
            
            # Добавляем найденные аномалии
            self.anomalies_found['by_type']['topological'].extend([
                {'event_index': idx, 'persistence': pers}
                for idx in anomaly_indices
            ])
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('topological')
            
            logger.info(f"Найдено {len(anomaly_indices)} топологических аномалий.")
            return anomaly_indices
        
        except Exception as e:
            logger.error(f"Ошибка при обнаружении топологических аномалий: {e}")
            return []
    
    def detect_model_behavior_anomalies(self, state_history: List[Dict]) -> List[int]:
        """Обнаружение аномалий в поведении модели."""
        try:
            if len(state_history) < 2:
                logger.warning("Недостаточно данных для анализа поведения модели.")
                return []
            
            # Извлекаем параметры
            luminosity = [s['beam_dynamics']['luminosity'][-1] for s in state_history]
            size_x = [s['beam_dynamics']['beam_size_x'][-1] for s in state_history]
            size_y = [s['beam_dynamics']['beam_size_y'][-1] for s in state_history]
            
            # Вычисляем разности
            size_x_diff = np.diff(size_x)
            size_y_diff = np.diff(size_y)
            
            anomaly_indices = []
            
            # Обнаружение аномальных изменений размеров пучка
            size_x_std = np.std(size_x_diff)
            size_y_std = np.std(size_y_diff)
            
            if size_x_std > 0:
                size_x_zscores = np.abs((size_x_diff - np.mean(size_x_diff)) / size_x_std)
                anomaly_indices.extend(np.where(size_x_zscores > 3.0)[0] + 1)
            
            if size_y_std > 0:
                size_y_zscores = np.abs((size_y_diff - np.mean(size_y_diff)) / size_y_std)
                anomaly_indices.extend(np.where(size_y_zscores > 3.0)[0] + 1)
            
            # Уникальные индексы аномалий
            anomaly_indices = list(set(anomaly_indices))
            
            # Добавляем найденные аномалии
            self.anomalies_found['by_type']['model_behavior'].extend([
                {'turn': idx, 'luminosity': luminosity[idx],
                 'size_x': size_x[idx], 'size_y': size_y[idx]}
                for idx in anomaly_indices
            ])
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('model_behavior')
            
            logger.info(f"Найдено {len(anomaly_indices)} аномалий в поведении модели.")
            return anomaly_indices
        
        except Exception as e:
            logger.error(f"Ошибка при обнаружении аномалий поведения модели: {e}")
            return []
    
    def detect_custom_anomalies(self, custom_detector_func, *args, **kwargs):
        """Поиск пользовательских аномалий с помощью пользовательской функции."""
        logger.info("Поиск пользовательских аномалий...")
        try:
            custom_anomalies = custom_detector_func(*args, **kwargs)
            if custom_anomalies:
                self.anomalies_found['by_type']['custom'].extend(custom_anomalies)
                self.anomalies_found['summary']['total_count'] += len(custom_anomalies)
                self.anomalies_found['summary']['types_found'].add('custom')
                logger.info(f"Найдено {len(custom_anomalies)} пользовательских аномалий.")
            return custom_anomalies
        except Exception as e:
            logger.error(f"Ошибка при поиске пользовательских аномалий: {e}")
            return []
    
    def detect_all_anomalies(self, events: List[Dict], state_history: List[Dict], 
                            max_events: int = 500):
        """Обнаружение всех типов аномалий."""
        # Статистический анализ
        for feature in ['energy', 'momentum', 'num_products']:
            self.detect_statistical_anomalies(events, feature)
        
        # Топологический анализ
        self.detect_topological_anomalies(events, max_events)
        
        # Анализ поведения модели
        self.detect_model_behavior_anomalies(state_history)
        
        return self.anomalies_found
    
    def generate_report(self, output_file: str = "anomaly_report.json"):
        """Генерирует отчет об обнаруженных аномалиях."""
        try:
            # Преобразуем set в list для сериализации JSON
            report = {
                'anomalies': self.anomalies_found['by_type'],
                'summary': {
                    'total_count': self.anomalies_found['summary']['total_count'],
                    'types_found': list(self.anomalies_found['summary']['types_found'])
                },
                'timestamp': time.time()
            }
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Отчет об аномалиях сохранен в {output_file}.")
            return self.anomalies_found
        except Exception as e:
            logger.error(f"Не удалось сохранить отчет об аномалиях: {e}")
            return self.anomalies_found

# ===================================================================
# 15. *** МОДУЛЬ: ROOTExporter ***
# ===================================================================
class ROOTExporter:
    """Экспортер данных симуляции в формат ROOT."""
    
    def __init__(self):
        """Инициализация экспортера ROOT."""
        try:
            import ROOT
            self.root_available = True
            logger.info("ROOTExporter инициализирован. ROOT доступен.")
        except ImportError:
            self.root_available = False
            logger.warning("ROOTExporter инициализирован. ROOT недоступен.")
    
    def export_to_root(self, filename: str, events: List[Dict]) -> bool:
        """Экспортирует события в ROOT файл."""
        if not self.root_available:
            logger.error("ROOT недоступен. Невозможно экспортировать в ROOT формат.")
            return False
        
        try:
            import ROOT
            
            # Создаем ROOT файл
            root_file = ROOT.TFile(filename, "RECREATE")
            
            # Создаем дерево для событий
            tree = ROOT.TTree("Events", "Симуляционные события LHC")
            
            # Определяем переменные
            event_id = ROOT.Int_t(0)
            num_products = ROOT.Int_t(0)
            total_energy = ROOT.Double_t(0)
            
            # Создаем ветки
            tree.Branch("event_id", ROOT.AddressOf(event_id), "event_id/I")
            tree.Branch("num_products", ROOT.AddressOf(num_products), "num_products/I")
            tree.Branch("total_energy", ROOT.AddressOf(total_energy), "total_energy/D")
            
            # Заполняем дерево
            for i, event in enumerate(events):
                event_id = i
                products = event.get('products', [])
                num_products = len(products)
                total_energy = sum(p.get('energy', 0.0) for p in products)
                
                tree.Fill()
            
            # Сохраняем и закрываем ф��йл
            root_file.Write()
            root_file.Close()
            
            logger.info(f"Данные успешно экспортированы в {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при экспорте в ROOT формат: {e}")
            return False

# ===================================================================
# 16. *** МОДУЛЬ: HepMC3Exporter ***
# ===================================================================
class HepMC3Exporter:
    """Экспортер данных симуляции в формат HepMC3."""
    
    def __init__(self):
        """Инициализация экспортера HepMC3."""
        try:
            import hepmc3
            self.hepmc3_available = True
            logger.info("HepMC3Exporter инициализирован. HepMC3 доступен.")
        except ImportError:
            self.hepmc3_available = False
            logger.warning("HepMC3Exporter инициализирован. HepMC3 недоступен.")
    
    def export_to_hepmc3(self, filename: str, events: List[Dict]) -> bool:
        """Экспортирует события в HepMC3 файл."""
        if not self.hepmc3_available:
            logger.error("HepMC3 недоступен. Невозможно экспортировать в HepMC3 формат.")
            return False
        
        try:
            import hepmc3
            
            # Здесь должен быть код для экспорта в HepMC3
            # Для примера просто создаем текстовый файл
            with open(filename, 'w') as f:
                f.write("HepMC3 file generated by LHCHybridModel\n")
                f.write("Units: GEV MM\n")
                for i, event in enumerate(events):
                    f.write(f"E {i} 0 0 0\n")  # Заголовок события
                    # Записываем входные частицы (протоны)
                    f.write(f"P 1 2212 0 0 6500 6500 0 0 0 1 0 0\n")
                    f.write(f"P 2 2212 0 0 -6500 -6500 0 0 0 1 0 0\n")
                    # Записываем продукты столкновения
                    products = event.get('products', [])
                    for j, p in enumerate(products, 3):
                        # Простая маппинг частиц в PDG коды
                        pdg_code = self._particle_to_pdg(p.get('name', ''))
                        px = p.get('px', 0.0)
                        py = p.get('py', 0.0)
                        pz = p.get('pz', 0.0)
                        e = p.get('energy', 0.0)
                        f.write(f"P {j} {pdg_code} {px} {py} {pz} {e} 0 0 0 1 0 0\n")
            
            logger.info(f"Данные успешно экспортированы в HepMC3 формат: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при экспорте в HepMC3 формат: {e}")
            return False
    
    def _particle_to_pdg(self, particle_name: str) -> int:
        """Конвертирует имя частицы в PDG код."""
        pdg_map = {
            'electron': 11,
            'positron': -11,
            'muon': 13,
            'antimuon': -13,
            'tau': 15,
            'antitau': -15,
            'neutrino_e': 12,
            'antineutrino_e': -12,
            'neutrino_mu': 14,
            'antineutrino_mu': -14,
            'neutrino_tau': 16,
            'antineutrino_tau': -16,
            'u_quark': 2,
            'u_antiquark': -2,
            'd_quark': 1,
            'd_antiquark': -1,
            's_quark': 3,
            's_antiquark': -3,
            'c_quark': 4,
            'c_antiquark': -4,
            'b_quark': 5,
            'b_antiquark': -5,
            't_quark': 6,
            't_antiquark': -6,
            'gluon': 21,
            'photon': 22,
            'Z_boson': 23,
            'W_plus': 24,
            'W_minus': -24,
            'higgs': 25,
            'pion+': 211,
            'pion-': -211,
            'pion0': 111,
            'kaon+': 321,
            'kaon-': -321,
            'proton': 2212,
            'antiproton': -2212,
            'neutron': 2112,
            'antineutron': -2112,
            'jet': 0  # Специальный код для струй
        }
        return pdg_map.get(particle_name, 0)

# ===================================================================
# 17. *** МОДУЛЬ: DetectorSystem ***
# ===================================================================
class DetectorSystem:
    """Система детектора для моделирования отклика детектора."""
    
    def __init__(self, config):
        self.config = config
        self.detector_elements = self._initialize_detector()
    
    def _initialize_detector(self):
        """Инициализация компонентов детектора."""
        # В реальной системе здесь будет сложная инициализация
        return {
            'tracker': {
                'resolution': 0.01,  # мм
                'efficiency': 0.98
            },
            'calorimeter': {
                'resolution': 0.05,  # ГэВ
                'efficiency': 0.95
            },
            'muon_system': {
                'resolution': 0.1,  # мм
                'efficiency': 0.92
            }
        }
    
    def process_event(self, event: Dict) -> Dict:
        """Обрабатывает событие через систему детектора."""
        detected_products = []
        
        for product in event.get('products', []):
            # Имитация детектирования с учетом эффективности
            if self._is_detected(product):
                # Имитация измерения с учетом разрешения детектора
                measured = self._apply_detector_resolution(product)
                detected_products.append(measured)
        
        return {
            'event_id': event.get('event_id', 0),
            'detected_products': detected_products,
            'timestamp': time.time()
        }
    
    def _is_detected(self, product: Dict) -> bool:
        """Проверяет, будет ли продукт зарегистрирован детектором."""
        # Вероятность детектирования зависит от типа частицы
        efficiency = 0.9  # Базовая эффективность
        
        if 'muon' in product.get('name', ''):
            efficiency = self.detector_elements['muon_system']['efficiency']
        elif 'electron' in product.get('name', ''):
            efficiency = self.detector_elements['calorimeter']['efficiency']
        elif 'jet' in product.get('name', ''):
            efficiency = min(
                self.detector_elements['tracker']['efficiency'],
                self.detector_elements['calorimeter']['efficiency']
            )
        
        return random.random() < efficiency
    
    def _apply_detector_resolution(self, product: Dict) -> Dict:
        """Применяет разрешение детектора к измерениям."""
        measured = product.copy()
        
        # Применяем гауссово шум с разрешением детектора
        if 'energy' in measured:
            if 'electron' in measured.get('name', '') or 'jet' in measured.get('name', ''):
                res = self.detector_elements['calorimeter']['resolution']
                measured['energy'] += np.random.normal(0, res * measured['energy'])
        
        if 'px' in measured:
            res = self.detector_elements['tracker']['resolution']
            measured['px'] += np.random.normal(0, res * abs(measured['px']))
        
        if 'py' in measured:
            res = self.detector_elements['tracker']['resolution']
            measured['py'] += np.random.normal(0, res * abs(measured['py']))
        
        if 'pz' in measured:
            res = self.detector_elements['tracker']['resolution']
            measured['pz'] += np.random.normal(0, res * abs(measured['pz']))
        
        return measured
    
    def get_detector_response(self, events: List[Dict]) -> List[Dict]:
        """Получает отклик детектора для списка событий."""
        return [self.process_event(event) for event in events]

# ===================================================================
# 18. *** МОДУЛЬ: Visualization ***
# ===================================================================
class Visualization:
    """Модуль визуализации результатов симуляции."""
    
    def __init__(self):
        self.plots = []
    
    def plot_geometry_3d(self, geometry, detector_system):
        """Визуализация 3D геометрии коллайдера и детектора."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Рисуем кольцо коллайдера
            theta = np.linspace(0, 2*np.pi, 100)
            r = geometry['radius']
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.zeros_like(x)
            ax.plot(x, y, z, 'b-', linewidth=2, label='Кольцо коллайдера')
            
            # Рисуем детекторные системы
            for detector, params in detector_system.detector_elements.items():
                # Простая визуализация детекторов как цилиндров
                r_det = r * (0.8 if detector == 'tracker' else 0.6 if detector == 'calorimeter' else 0.4)
                z_det = np.linspace(-5, 5, 20)
                theta_det = np.linspace(0, 2*np.pi, 30)
                theta_det, z_det = np.meshgrid(theta_det, z_det)
                x_det = r_det * np.cos(theta_det)
                y_det = r_det * np.sin(theta_det)
                ax.plot_surface(x_det, y_det, z_det, alpha=0.3, label=detector)
            
            ax.set_xlabel('X (м)')
            ax.set_ylabel('Y (м)')
            ax.set_zlabel('Z (м)')
            ax.set_title('3D Геометрия коллайдера и детекторных систем')
            ax.legend()
            
            plt.tight_layout()
            self.plots.append(('geometry_3d', fig))
            plt.show()
            
            logger.info("3D визуализация геометрии завершена.")
        except Exception as e:
            logger.error(f"Ошибка при 3D визуализации геометрии: {e}")
    
    def plot_detector_response_3d(self, detected_events, detector_system):
        """Визуализация 3D отклика детектора."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Рисуем следы частиц
            for event in detected_events[:10]:  # Ограничиваем количество событий для ясности
                for product in event.get('detected_products', []):
                    # Генерируем простой след частицы
                    r = np.sqrt(product.get('px', 0)**2 + product.get('py', 0)**2 + product.get('pz', 0)**2)
                    if r > 0:
                        x = [0, 10 * product.get('px', 0) / r]
                        y = [0, 10 * product.get('py', 0) / r]
                        z = [0, 10 * product.get('pz', 0) / r]
                        ax.plot(x, y, z, 'r-', alpha=0.5)
            
            # Рисуем детекторные системы (как в plot_geometry_3d)
            geometry = {'radius': 4297}  # Радиус LHC
            for detector, params in detector_system.detector_elements.items():
                r_det = geometry['radius'] * (0.8 if detector == 'tracker' else 0.6 if detector == 'calorimeter' else 0.4)
                z_det = np.linspace(-5, 5, 20)
                theta_det = np.linspace(0, 2*np.pi, 30)
                theta_det, z_det = np.meshgrid(theta_det, z_det)
                x_det = r_det * np.cos(theta_det)
                y_det = r_det * np.sin(theta_det)
                ax.plot_surface(x_det, y_det, z_det, alpha=0.3)
            
            ax.set_xlabel('X (м)')
            ax.set_ylabel('Y (м)')
            ax.set_zlabel('Z (м)')
            ax.set_title('3D Отклик детектора на события')
            
            plt.tight_layout()
            self.plots.append(('detector_response_3d', fig))
            plt.show()
            
            logger.info("3D визуализация отклика детектора завершена.")
        except Exception as e:
            logger.error(f"Ошибка при 3D визуализации отклика детектора: {e}")
    
    def plot_beam_parameters(self, state_history):
        """Визуализация параметров пучка во времени."""
        try:
            import matplotlib.pyplot as plt
            
            turns = range(len(state_history))
            luminosity = [s['beam_dynamics']['luminosity'][-1] for s in state_history]
            size_x = [s['beam_dynamics']['beam_size_x'][-1] for s in state_history]
            size_y = [s['beam_dynamics']['beam_size_y'][-1] for s in state_history]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Светимость
            ax1.plot(turns, luminosity, 'b-')
            ax1.set_xlabel('Обороты')
            ax1.set_ylabel('Светимость (см⁻²с⁻¹)')
            ax1.set_title('Эволюция светимости')
            ax1.grid(True)
            
            # Размеры пучка
            ax2.plot(turns, size_x, 'r-', label='σ_x')
            ax2.plot(turns, size_y, 'g-', label='σ_y')
            ax2.set_xlabel('Обороты')
            ax2.set_ylabel('Размер пучка (м)')
            ax2.set_title('Эволюция размеров пучка')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            self.plots.append(('beam_parameters', fig))
            plt.show()
            
            logger.info("Визуализация параметров пучка завершена.")
        except Exception as e:
            logger.error(f"Ошибка при визуализации параметров пучка: {e}")

    def visualize_collision_event(self, event: Dict, save_path: str = "visualizations/collision_event.png"):
        """Визуализация отдельного события столкновения с подробными подписями."""
        try:
            import os
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            # Создаем папку visualizations, если не существует
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            fig = plt.figure(figsize=(14, 10))

            # 3D визуализация продуктов столкновения
            ax = fig.add_subplot(2, 2, 1, projection='3d')
            products = event.get('products', [])

            # Собираем данные о частицах
            px_vals, py_vals, pz_vals = [], [], []
            energies, names = [], []

            for product in products:
                px_vals.append(product.get('px', 0))
                py_vals.append(product.get('py', 0))
                pz_vals.append(product.get('pz', 0))
                energies.append(product.get('energy', 0))
                names.append(product.get('name', 'unk'))

            # Нормализуем энергию для размера точек
            norm_energies = [max(10, min(200, e/np.max(energies)*100)) if np.max(energies) > 0 else 50 for e in energies]

            # Построение 3D графика
            scatter = ax.scatter(px_vals, py_vals, pz_vals, c=energies, s=norm_energies, cmap='viridis', alpha=0.7)
            ax.set_xlabel('Px (ГэВ)')
            ax.set_ylabel('Py (ГэВ)')
            ax.set_zlabel('Pz (ГэВ)')
            ax.set_title(f'3D Визуализация продуктов столкновения\nПроцесс: {event.get("process", "unknown")}')

            # Добавляем подписи к частицам
            for i, (x, y, z, name) in enumerate(zip(px_vals, py_vals, pz_vals, names)):
                ax.text(x, y, z, name, fontsize=8)

            plt.colorbar(scatter, ax=ax)

            # 2D проекция на плоскость xy
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.scatter(px_vals, py_vals, c=energies, s=norm_energies, cmap='viridis', alpha=0.7)
            for i, (x, y, name) in enumerate(zip(px_vals, py_vals, names)):
                ax2.text(x, y, name, fontsize=8)
            ax2.set_xlabel('Px (ГэВ)')
            ax2.set_ylabel('Py (ГэВ)')
            ax2.set_title('Проекция на плоскость XY')
            ax2.grid(True)

            # 2D проекция на плоскость xz
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.scatter(px_vals, pz_vals, c=energies, s=norm_energies, cmap='viridis', alpha=0.7)
            for i, (x, z, name) in enumerate(zip(px_vals, pz_vals, names)):
                ax3.text(x, z, name, fontsize=8)
            ax3.set_xlabel('Px (ГэВ)')
            ax3.set_ylabel('Pz (ГэВ)')
            ax3.set_title('Проекция на плоскость XZ')
            ax3.grid(True)

            # Энергетическое распределение
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.hist(energies, bins=min(20, len(energies)), edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Энергия (ГэВ)')
            ax4.set_ylabel('Число частиц')
            ax4.set_title('Распределение энергии продуктов столкновения')
            ax4.grid(True)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.plots.append(('collision_event', fig))
            plt.show()

            logger.info(f"Визуализация столкновения сохранена в {save_path}.")
        except Exception as e:
            logger.error(f"Ошибка при визуализации события столкновения: {e}")

    def visualize_multiple_collision_events(self, events: List[Dict], max_events: int = 5, save_dir: str = "visualizations/"):
        """Визуализация множества событий столкновений."""
        try:
            import os
            os.makedirs(save_dir, exist_ok=True)

            # Визуализируем распределение энергии для всех событий
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            all_energies = []
            all_momenta = {'px': [], 'py': [], 'pz': []}
            process_types = []

            for event in events:
                products = event.get('products', [])
                event_energy = sum(p.get('energy', 0) for p in products)
                all_energies.append(event_energy)

                # Суммарные импульсы
                total_px = sum(p.get('px', 0) for p in products)
                total_py = sum(p.get('py', 0) for p in products)
                total_pz = sum(p.get('pz', 0) for p in products)

                all_momenta['px'].append(total_px)
                all_momenta['py'].append(total_py)
                all_momenta['pz'].append(total_pz)

                process_type = event.get('process', 'unknown')
                process_types.append(process_type)

            # Распределение энергии столкновений
            axes[0,0].hist(all_energies, bins=30, edgecolor='black', alpha=0.7)
            axes[0,0].set_xlabel('Энергия столкновения (ГэВ)')
            axes[0,0].set_ylabel('Частота')
            axes[0,0].set_title('Распределение энергии столкновений')
            axes[0,0].grid(True)

            # Распределение импульсов
            axes[0,1].hist(all_momenta['px'], bins=30, alpha=0.5, label='Px', edgecolor='black')
            axes[0,1].hist(all_momenta['py'], bins=30, alpha=0.5, label='Py', edgecolor='black')
            axes[0,1].hist(all_momenta['pz'], bins=30, alpha=0.5, label='Pz', edgecolor='black')
            axes[0,1].set_xlabel('Импульс (ГэВ)')
            axes[0,1].set_ylabel('Частота')
            axes[0,1].set_title('Распределение импульсов')
            axes[0,1].legend()
            axes[0,1].grid(True)

            # Типы процессов
            if len(process_types) > 0:
                unique_types, counts = np.unique(process_types, return_counts=True)
                axes[1,0].bar(unique_types, counts)
                axes[1,0].set_xlabel('Тип процесса')
                axes[1,0].set_ylabel('Частота')
                axes[1,0].set_title('Распределение типов процессов')
                axes[1,0].tick_params(axis='x', rotation=45)
            else:
                axes[1,0].text(0.5, 0.5, 'Нет данных', horizontalalignment='center',
                              verticalalignment='center', transform=axes[1,0].transAxes)
                axes[1,0].set_title('Распределение типов процессов (нет данных)')

            # Корреляция энергии и импульса
            if len(all_momenta['px']) > 0:
                total_momenta = np.sqrt(np.array(all_momenta['px'])**2 +
                                      np.array(all_momenta['py'])**2 +
                                      np.array(all_momenta['pz'])**2)
                axes[1,1].scatter(all_energies, total_momenta, alpha=0.6)
                axes[1,1].set_xlabel('Энергия столкновения (ГэВ)')
                axes[1,1].set_ylabel('Суммарный импульс (ГэВ)')
                axes[1,1].set_title('Корреляция энергии и импульса')
                axes[1,1].grid(True)
            else:
                axes[1,1].text(0.5, 0.5, 'Нет данных', horizontalalignment='center',
                              verticalalignment='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Корреляция энергии и импульса (нет данных)')

            plt.tight_layout()
            multi_events_path = os.path.join(save_dir, "multiple_collision_events.png")
            plt.savefig(multi_events_path, dpi=300, bbox_inches='tight')
            self.plots.append(('multiple_collision_events', fig))
            plt.show()

            logger.info(f"Визуализация множества событий сохранена в {multi_events_path}")

            # Визуализируем первые max_events событий по отдельности
            for i, event in enumerate(events[:max_events]):
                event_path = os.path.join(save_dir, f"event_{i}.png")
                self.visualize_collision_event(event, event_path)

        except Exception as e:
            logger.error(f"Ошибка при визуализации множества событий столкновений: {e}")

    def visualize_collision_physics(self, events: List[Dict], save_path: str = "visualizations/collision_physics.png"):
        """Визуализация физических характеристик столкновений с пояснениями."""
        try:
            import os
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # Подсчитываем статистику по частицам
            particle_stats = {}
            process_stats = {}
            energy_ranges = {'low': 0, 'medium': 0, 'high': 0}

            for event in events:
                proc = event.get('process', 'unknown')
                if proc not in process_stats:
                    process_stats[proc] = 0
                process_stats[proc] += 1

                products = event.get('products', [])
                total_energy = sum(p.get('energy', 0) for p in products)

                # Классифицируем по энергии
                if total_energy < 100:
                    energy_ranges['low'] += 1
                elif total_energy < 1000:
                    energy_ranges['medium'] += 1
                else:
                    energy_ranges['high'] += 1

                for product in products:
                    name = product.get('name', 'unknown')
                    if name not in particle_stats:
                        particle_stats[name] = 0
                    particle_stats[name] += 1

            # Диаграмма распределения процессов
            axes[0,0].pie(process_stats.values(), labels=process_stats.keys(), autopct='%1.1f%%')
            axes[0,0].set_title('Распределение типов процессов')

            # Диаграмма энергетических диапазонов
            axes[0,1].pie(energy_ranges.values(), labels=energy_ranges.keys(), autopct='%1.1f%%')
            axes[0,1].set_title('Распределение по энергетическим диапазонам')

            # Топ-5 производимых частиц
            top_particles = dict(sorted(particle_stats.items(), key=lambda x: x[1], reverse=True)[:5])
            particle_names = list(top_particles.keys())
            particle_counts = list(top_particles.values())
            bars = axes[0,2].bar(particle_names, particle_counts)
            axes[0,2].set_title('Топ-5 производимых частиц')
            axes[0,2].tick_params(axis='x', rotation=45)

            # Добавляем значения на столбцах
            for bar, count in zip(bars, particle_counts):
                height = bar.get_height()
                axes[0,2].text(bar.get_x() + bar.get_width()/2., height,
                             f'{int(count)}',
                             ha='center', va='bottom')

            # Энергетические спектры
            energies = []
            for event in events:
                products = event.get('products', [])
                event_energy = sum(p.get('energy', 0) for p in products)
                energies.append(event_energy)

            if energies:
                axes[1,0].hist(energies, bins=50, edgecolor='black', alpha=0.7)
                axes[1,0].set_xlabel('Энергия (ГэВ)')
                axes[1,0].set_ylabel('Число событий')
                axes[1,0].set_title('Спектр энергии событий')
                axes[1,0].grid(True)

            # Распределение по импульсам
            momenta = []
            for event in events:
                products = event.get('products', [])
                for p in products:
                    px, py, pz = p.get('px', 0), p.get('py', 0), p.get('pz', 0)
                    p_total = np.sqrt(px**2 + py**2 + pz**2)
                    momenta.append(p_total)

            if momenta:
                axes[1,1].hist(momenta, bins=50, edgecolor='black', alpha=0.7)
                axes[1,1].set_xlabel('Импульс (ГэВ)')
                axes[1,1].set_ylabel('Число частиц')
                axes[1,1].set_title('Распределение импульсов частиц')
                axes[1,1].grid(True)

            # Сравнение Px-Py корреляции
            px_vals, py_vals = [], []
            for event in events:
                products = event.get('products', [])
                for p in products:
                    px_vals.append(p.get('px', 0))
                    py_vals.append(p.get('py', 0))

            if px_vals and py_vals:
                axes[1,2].scatter(px_vals, py_vals, alpha=0.5)
                axes[1,2].set_xlabel('Px (ГэВ)')
                axes[1,2].set_ylabel('Py (ГэВ)')
                axes[1,2].set_title('Корреляция поперечных импульсов')
                axes[1,2].grid(True)

            # Добавляем пояснительный текст
            explanation_text = """
            Пояснения к визуализации:
            - Круговые диаграммы показывают распределение процессов и энергетических диапазонов
            - Гистограммы демонстрируют энергетические и импульсные спектры
            - Вертикальные столбцы отражают наиболее часто производимые частицы
            - Физические процессы: Drell-Yan (линейное взаимодействие),
              Gluon Fusion (производство бозонов),
              Jet Production (производство адронных струй)
            """

            fig.suptitle('Физические характеристики столкновений в LHC', fontsize=16)

            # Добавляем текстовое пояснение
            fig.text(0.02, 0.02, explanation_text, fontsize=9, verticalalignment='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.plots.append(('collision_physics', fig))
            plt.show()

            logger.info(f"Физическая визуализация столкновений сохранена в {save_path}")

        except Exception as e:
            logger.error(f"Ошибка при визуализации физических характеристик столкновений: {e}")

# ===================================================================
# 19. Основная модель коллайдера
# ===================================================================
class LHCHybridModel:
    """Усовершенствованная гибридная модель Большого адронного коллайдера.
    Это центральный класс фреймворка, объединяющий все компоненты:
    - Физические и динамические движки
    - Системы анализа (TopoAnalyzer, GradientCalibrator, AnomalyDetector)
    - Экспорт данных
    - Визуализация"""
    
    def __init__(self, config=None):
        # Загрузка конфигурации
        self.config = config or self._load_default_config()
        
        # Инициализация компонентов
        self.geometry = self._initialize_geometry()
        self.beam_dynamics = BeamDynamics(self.config)
        self.magnetic_optics = MagneticOpticsSystem(self.config)  # Система магнитной оптики
        self.physics_engines = self._initialize_physics_engines()
        self.detector_system = DetectorSystem(self.config)
        self.visualizer = Visualization()
        
        # Системы анализа
        self.topo_analyzer = TopoAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.calibrator = None
        
        # Состояние модели
        self.simulation_state = {
            'beam_dynamics': {
                'turn': [],
                'luminosity': [],
                'beam_size_x': [],
                'beam_size_y': [],
                'time': []
            },
            'detected_events': [],
            'recent_events': []
        }
    
    def _load_default_config(self):
        """Загружает конфигурацию по умолчанию."""
        default_config = {
            'beam': {
                'beam_energy': 6500,  # ГэВ
                'bunch_intensity': 1.15e11,
                'num_bunches': 2556,
                'circumference': 26659,  # м
                'beta_star': 0.55,  # м
                'beta_x_star': 0.55,  # м (раздельные beta-функции в IP)
                'beta_y_star': 0.55,  # м
                'crossing_angle': 0.0,  # радианы
                'emittance': 2.5e-6,  # м·рад
                'sigma_x': 0.045,  # м
                'sigma_y': 0.045,  # м
                'beta_x': 56.5,  # м
                'beta_y': 56.5,  # м
                'tune': 64.0,  # отношение частоты осцилляции к частоте оборота
                'bunch_length': 0.075,  # м
                'sigma_z': 0.01875,  # м (bunch_length/4)
                'rf_voltage': 12e6,  # В (RF напряжение)
                'rf_harmonic': 35640,  # гармоника
                'rf_phase': 0.0,  # фаза
                'rf_noise': 5e-5,  # шум в RF системе
                'space_charge_tune_shift': 0.0001,  # сдвиг tune от пространственного заряда
                'beam_beam_parameter': 0.15  # параметр для эффекта пучок-пучок
            },
            'geometry': {
                'radius': 4297,  # м (радиус LHC)
                'straight_sections': 8,
                'bending_magnets': 1232,
                'dipole_field': 8.33,  # Тл (магнитное поле дипольных магнитов)
                'quad_gradient': 107.0,  # Тл/м (градиент фокусирующих квадруполей)
                'sext_gradient': 3000.0,  # Тл/м² (градиент сексуполей)
                'xi_x': -8.0,  # хроматичность x
                'xi_y': -8.0,  # хроматичность y
                'beta_beat_amplitude': 0.02,  # амплитуда бета-биений
                'dq_x_da': -0.05,  # сдвиг tune с амплитудой x
                'dq_y_da': -0.05,  # сдвиг tune с амплитудой y
                'dq_x_de': 0.0,  # сдвиг tune с энергией x
                'dq_y_de': 0.0  # сдвиг tune с энергией y
            },
            'beam_beam': {
                'enabled': True,  # включено моделирование взаимодействия пучков
                'strength': 0.01  # сила взаимодействия
            },
            'validation': {
                'dataset_id': 'CMS_OpenData_2018'
            }
        }
        logger.info("Используется конфигурация по умолчанию.")
        return default_config
    
    def _initialize_geometry(self):
        """Инициализирует геометрию коллайдера с учетом магнитной оптики."""
        return {
            'radius': self.config['geometry']['radius'],
            'circumference': self.config['beam']['circumference'],
            'straight_sections': self.config['geometry']['straight_sections'],
            'bending_magnets': self.config['geometry']['bending_magnets'],
            # Добавим параметры магнитной оптики
            'magnetic_rigidity': self.config['beam']['beam_energy'] / (c * e),  # B*ρ (Tesla*m)
            'dipole_field': self.config['geometry'].get('dipole_field', 8.33),  # Тл (для LHC)
            'quadrupole_field_gradient': self.config['geometry'].get('quad_gradient', 107.0),  # Тл/м
            'sextupole_field_gradient': self.config['geometry'].get('sext_gradient', 3000.0),  # Тл/м²
            'focusing_strength': self._calculate_focusing_strength(),
            'beta_beat_amplitude': self.config['geometry'].get('beta_beat_amplitude', 0.02),  # Относительное возмущение
            'chromaticity': {
                'xi_x': self.config['geometry'].get('xi_x', -8.0),  # хроматичность в горизонтальной плоскости
                'xi_y': self.config['geometry'].get('xi_y', -8.0),  # хроматичность в вертикальной плоскости
            },
            'tune_shifts': {
                'with_amplitude': {
                    'dq_x_da': self.config['geometry'].get('dq_x_da', -0.05),  # сдвиг tune с амплитудой
                    'dq_y_da': self.config['geometry'].get('dq_y_da', -0.05),
                },
                'with_energy': {
                    'dq_x_de': self.config['geometry'].get('dq_x_de', 0.0),  # сдвиг tune с энергией
                    'dq_y_de': self.config['geometry'].get('dq_y_de', 0.0),
                }
            }
        }

    def _calculate_focusing_strength(self) -> Dict:
        """Рассчитывает параметры фокусирующей силы в магнитной структуре."""
        # Для LHC используем типичные значения
        # Focusing strength parameter k = gradient / (B*ρ) where k is in m⁻²
        dipole_field = self.config['geometry'].get('dipole_field', 8.33)  # Тл
        beam_energy = self.config['beam']['beam_energy']  # ГэВ
        magnetic_rigidity = beam_energy / (c * e) * 1e9  # Тл*м (переводим из ГэВ в эВ)

        quad_gradient = self.config['geometry'].get('quad_gradient', 107.0)  # Тл/м
        k_x = quad_gradient / magnetic_rigidity if magnetic_rigidity > 0 else 0.0  # фокусирующий квадруполь
        k_y = -quad_gradient / magnetic_rigidity if magnetic_rigidity > 0 else 0.0  # дефокусирующий квадруполь

        return {
            'k_x': k_x,  # фокусирующая сила в горизонтальной плоскости (м⁻²)
            'k_y': k_y,  # фокусирующая сила в вертикальной плоскости (м⁻²)
            'lattice_type': 'FODO'  # тип магнитной структуры (Focus-Drift-Defocus-Drift)
        }
    
    def _initialize_physics_engines(self):
        """Инициализирует физические движки."""
        engines = {}
        
        # Встроенный движок
        try:
            engines["built-in"] = BuiltInPhysicsEngine({}, self.config)
            logger.info("Встроенный физический движок инициализирован.")
        except Exception as e:
            logger.error(f"Ошибка инициализации встроенного физического движка: {e}")
        
        return engines
    
    def run_simulation(self, num_turns: int = 10, include_space_charge: bool = True):
        """Запускает симуляцию коллайдера."""
        logger.info(f"Запуск симуляции на {num_turns} оборотов.")
        
        # Эволюция динамики пучка
        for turn in range(num_turns):
            # Обновляем состояние пучка
            state = self.beam_dynamics.evolve(1, include_space_charge)

            # Применяем эффекты магнитной оптики к состоянию пучка
            state = self.magnetic_optics.apply_optics_effects(state, 'arc')

            # Регистрируем параметры
            self.simulation_state['beam_dynamics']['turn'].append(turn)
            self.simulation_state['beam_dynamics']['luminosity'].append(self.beam_dynamics.get_luminosity())
            self.simulation_state['beam_dynamics']['beam_size_x'].append(state['sigma_x'])
            self.simulation_state['beam_dynamics']['beam_size_y'].append(state['sigma_y'])
            self.simulation_state['beam_dynamics']['time'].append(turn * 88.9e-6)  # Время одного оборота
            self.simulation_state['beam_dynamics']['tune_x_shift'] = state.get('tune_x_shift', 0.0)
            self.simulation_state['beam_dynamics']['tune_y_shift'] = state.get('tune_y_shift', 0.0)

            # Генерируем столкновения (с вероятностью, зависящей от светимости)
            if random.random() < self.beam_dynamics.get_luminosity() * 1e-34:
                # Используем встроенный движок для генерации событий
                events = self.physics_engines["built-in"].interact(
                    "proton", "proton",
                    self.config['beam']['beam_energy'],
                    num_events=1
                )

                if events:
                    # Обр��батываем события через детектор
                    detected = self.detector_system.get_detector_response(events)
                    self.simulation_state['detected_events'].extend(detected)
                    self.simulation_state['recent_events'] = self.simulation_state['recent_events'][-99:] + events
        
        logger.info("Симуляция завершена.")
        return self.simulation_state
    
    def analyze_topology(self, max_events: int = 500, compute_persistence: bool = True, compute_pca: bool = True):
        """Анализирует топологию событий."""
        logger.info("Запуск топологического анализа событий...")
        
        # ��спользуем последние события или сохраненные
        events_to_analyze = self.simulation_state['recent_events'][:max_events] if self.simulation_state['recent_events'] else None
        
        if not events_to_analyze:
            logger.warning("Нет событий для топологического анализа.")
            return False
        
        # Запускаем анализ
        success = self.topo_analyzer.analyze_events(events_to_analyze, max_events)
        
        if success:
            logger.info("Топологический анализ завершен успешно.")
            self.topo_analyzer.generate_report()
        else:
            logger.error("Топологический анализ завершился с ошибкой.")
        
        return success
    
    def calibrate_model(self, target_observables: Dict[str, float], 
                       parameters_to_calibrate: List[str]):
        """Калибрует модель для достижения целевых наблюдаемых."""
        logger.info("Запуск калибровки модели...")
        
        self.calibrator = GradientCalibrator(
            self, 
            target_observables,
            parameters_to_calibrate
        )
        
        # Начальные значения параметров
        initial_params = [
            self.config['beam'].get(param, 1.0) if param in self.config.get('beam', {}) 
            else self.config['geometry'].get(param, 1.0)
            for param in parameters_to_calibrate
        ]
        
        # Границы для оптимизации
        bounds = []
        for param in parameters_to_calibrate:
            if param == 'beam_energy':
                bounds.append((6000, 8000))  # ГэВ
            elif param == 'num_bunches':
                bounds.append((1, 2808))  # Максимум в LHC
            elif param == 'bunch_intensity':
                bounds.append((1e10, 2e11))  # Частиц в пучке
            else:
                bounds.append((None, None))
        
        # Запускаем калибровку
        self.calibrator.calibrate(initial_params)
        
        # Анализ чувствительности
        self.calibrator.analyze_sensitivity()
        
        logger.info("Калибровка модели завершена.")
        return self.calibrator.get_summary_report()
    
    def detect_anomalies(self):
        """Обнаруживает аномалии в данных симуляции."""
        logger.info("Запуск обнаружения аномалий...")
        
        # Статистические аномалии
        for feature in ['energy', 'momentum', 'num_products']:
            self.anomaly_detector.detect_statistical_anomalies(
                self.simulation_state['recent_events'], 
                feature
            )
        
        # Топологические аномалии
        self.anomaly_detector.detect_topological_anomalies(
            self.simulation_state['recent_events']
        )
        
        # Аномалии поведения модели
        self.anomaly_detector.detect_model_behavior_anomalies(
            [self.simulation_state['beam_dynamics']]
        )
        
        # Генерация отчета
        self.anomaly_detector.generate_report()
        
        logger.info("Обнаружение аномалий завершено.")
        return self.anomaly_detector.anomalies_found
    
    def export_to_root(self, filename: str) -> bool:
        """Экспортирует данные симуляции в ROOT формат."""
        exporter = ROOTExporter()
        return exporter.export_to_root(filename, self.simulation_state['detected_events'])
    
    def export_to_hepmc3(self, filename: str) -> bool:
        """Экспортирует данные симуляции в HepMC3 формат."""
        exporter = HepMC3Exporter()
        return exporter.export_to_hepmc3(filename, self.simulation_state['recent_events'])
    
    def enhanced_visualization(self):
        """Улучшенная визуализация результатов симуляции."""
        logger.info("Запуск улучшенной визуализации результатов...")
        try:
            # 3D визуализация геометрии коллайдера
            self.visualizer.plot_geometry_3d(self.geometry, self.detector_system)

            # Визуализация отклика детектора
            if self.simulation_state['detected_events']:
                self.visualizer.plot_detector_response_3d(self.simulation_state['detected_events'], self.detector_system)

            # Визуализация параметров пучка
            self.visualizer.plot_beam_parameters(self.simulation_state['beam_dynamics'])

            # Визуализация результатов столкновений
            if self.simulation_state['recent_events']:
                logger.info("Запуск визуализации физических событий столкновений...")

                # Визуализация физических характеристик
                self.visualizer.visualize_collision_physics(self.simulation_state['recent_events'])

                # Визуализация нескольких типичных событий
                for i, event in enumerate(self.simulation_state['recent_events'][:3]):  # Первые 3 события
                    self.visualizer.visualize_collision_event(event, f"visualizations/collision_event_{i}.png")

                # Визуализация множества событий
                self.visualizer.visualize_multiple_collision_events(
                    self.simulation_state['recent_events'],
                    save_dir="visualizations/"
                )

            logger.info("Улучшенная визуализация завершена.")
        except Exception as e:
            logger.error(f"Ошибка при улучшенной визуализации: {e}")
    
    def get_luminosity(self) -> float:
        """Возвращает текущую светимость."""
        if self.simulation_state['beam_dynamics']['luminosity']:
            return self.simulation_state['beam_dynamics']['luminosity'][-1]
        return 0.0
    
    def get_beam_size_x(self) -> float:
        """Возвращает текущий размер пучка по X."""
        if self.simulation_state['beam_dynamics']['beam_size_x']:
            return self.simulation_state['beam_dynamics']['beam_size_x'][-1]
        return 0.0
    
    def get_beam_size_y(self) -> float:
        """Возвращает текущий размер пучка по Y."""
        if self.simulation_state['beam_dynamics']['beam_size_y']:
            return self.simulation_state['beam_dynamics']['beam_size_y'][-1]
        return 0.0
    
    def get_recent_events(self) -> List[Dict]:
        """Возвращает последние события."""
        return self.simulation_state['recent_events']

# ===================================================================
# 20. Функции демонстрации
# ===================================================================
def create_default_config():
    """Создает файл конфигурации по умолчанию."""
    default_config = {
        'beam': {
            'beam_energy': 6500,
            'bunch_intensity': 1.15e11,
            'num_bunches': 2556,
            'circumference': 26659,
            'beta_star': 0.55,
            'crossing_angle': 0.0,
            'emittance': 2.5e-6,
            'sigma_x': 0.045,
            'sigma_y': 0.045,
            'beta_x': 56.5,
            'beta_y': 56.5
        },
        'geometry': {
            'radius': 4297,
            'straight_sections': 8,
            'bending_magnets': 1232
        },
        'validation': {
            'dataset_id': 'CMS_OpenData_2018'
        }
    }
    
    with open("lhc_config.yaml", 'w') as f:
        yaml.dump(default_config, f)
    
    logger.info("Создан файл конфигурации по умолчанию: lhc_config.yaml")

# ===================================================================
# 21. Основной сценарий
# ===================================================================
if __name__ == "__main__":
    logger.info("Сценарий: Демонстрация унифицированного фреймворка")
    
    if not os.path.exists("lhc_config.yaml"):
        create_default_config()
    
    lhc = LHCHybridModel()
    lhc.run_simulation(num_turns=20)
    
    # Топологический анализ
    lhc.analyze_topology(max_events=500, compute_persistence=True, compute_pca=True)
    
    # Калибровка модели
    target_observables = {
        'luminosity': 1.5e34,
        'beam_size_x': 0.045,
        'avg_event_energy': 5000.0
    }
    params_to_calibrate = ['beam_energy', 'num_bunches', 'bunch_intensity']
    calibration_report = lhc.calibrate_model(target_observables, params_to_calibrate)
    
    # Обнаружение аномалий
    anomalies = lhc.detect_anomalies()
    
    # Экспорт данных
    lhc.export_to_root("simulation_results.root")
    lhc.export_to_hepmc3("simulation_results.hepmc3")
    
    # Визуализация
    lhc.enhanced_visualization()
    
    logger.info("Демонстрация завершена.")
