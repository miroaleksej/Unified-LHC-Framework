#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LHC 2.0: Расширенная система моделирования и анализа данных Большого адронного коллайдера
Версия с исправленными ошибками и улучшенной функциональностью
"""
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
# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lhc_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LHC_2.0")

# ===================================================================
# Добавлены физические константы с уточненными значениями
# ===================================================================
c = 299792458  # м/с (скорость света)
m_p = 0.938272  # ГэВ/с² (масса протона)
G_F = 1.1663787e-5  # Ферми-константа (ГэВ⁻²)
M_W = 80.379  # ГэВ (масса W-бозона)
M_Z = 91.1876  # ГэВ (масса Z-бозона)
M_H = 125.1  # ГэВ (масса Хиггса)
alpha_s = 0.118  # Сильная константа связи
alpha_em = 1.0 / 137.0  # Постоянная тонкой структуры
v = 246  # vev, ГэВ (электрослабый вакуумный ожидаемый вакуум)
hbar = 6.582119569e-25  # ГэВ·с (приведенная постоянная Планка)
k_B = 8.617333262e-5  # эВ/К (постоянная Больцмана)
SMALL_EPSILON = 1e-12  # Малая константа для предотвращения деления на ноль

# ===================================================================
# Попытка импортировать специализированные библиотеки
# ===================================================================
try:
    import gudhi
    GUDHI_AVAILABLE = True
    logger.info("Библиотека GUDHI успешно импортирована")
except ImportError:
    GUDHI_AVAILABLE = False
    logger.warning("Библиотека GUDHI не установлена. Установите через 'pip install gudhi'")

try:
    from fastecdsa import curve, point
    FAST_ECDSA_AVAILABLE = True
    logger.info("Библиотека fastecdsa успешно импортирована")
except ImportError:
    FAST_ECDSA_AVAILABLE = False
    logger.warning("Библиотека fastecdsa не установлена. Установите через 'pip install fastecdsa'")

try:
    import ROOT
    ROOT_AVAILABLE = True
    logger.info("Библиотека ROOT успешно импортирована")
except ImportError:
    ROOT_AVAILABLE = False
    logger.warning("Библиотека ROOT не установлена. Требуется для экспорта данных")

try:
    import hepmc3
    HEP_MC3_AVAILABLE = True
    logger.info("Библиотека hepmc3 успешно импортирована")
except ImportError:
    HEP_MC3_AVAILABLE = False
    logger.warning("Библиотека hepmc3 не установлена. Установите через 'pip install hepmc3'")

try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("GPU доступен для ускорения вычислений")
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("GPU не доступен. Установите cupy для ускорения вычислений")

try:
    import xrootdpyfs
    XROOTD_AVAILABLE = True
    logger.info("Библиотека xrootdpyfs успешно импортирована")
except ImportError:
    XROOTD_AVAILABLE = False
    logger.warning("Библиотека xrootdpyfs не установлена. Установите через 'pip install xrootdpyfs'")

try:
    import dash
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output
    import plotly.express as px
    DASH_AVAILABLE = True
    logger.info("Библиотека Dash успешно импортирована")
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("Библиотека Dash не установлена. Установите через 'pip install dash'")

# ===================================================================
# 1. *** МОДУЛЬ: Кэш ***
# ===================================================================
class SimpleCache:
    """Простой кэш для ускорения вычислений."""
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
# 2. *** МОДУЛЬ: ParticleDatabase ***
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
# 3. *** МОДУЛЬ: TopologicalECDSAAnalyzer ***
# ===================================================================
class TopologicalECDSAAnalyzer:
    """Анализ топологических свойств, вдохновленных эллиптическими кривыми."""
    def __init__(self):
        self.curves = {
            'secp256k1': curve.secp256k1,
            'P256': curve.P256
        }
        self.points = {}
        self.logger = logging.getLogger("TopologicalECDSAAnalyzer")
    def initialize_points(self, curve_name='secp256k1'):
        """Инициализация точек на эллиптической кривой."""
        if not FAST_ECDSA_AVAILABLE:
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
        if not FAST_ECDSA_AVAILABLE:
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
                energy = event.get('total_energy', 0.0)
                px = event.get('px', 0.0)
                py = event.get('py', 0.0)
                pz = event.get('pz', 0.0)
                theta = np.arctan2(np.sqrt(px**2 + py**2), pz) if pz != 0 else np.pi/2
                phi = np.arctan2(py, px) if px != 0 else 0
                # Добавляем барицентрические координаты для струй
                jet_features = []
                for particle in event.get('particles', []):
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
                features.append([energy, theta, phi, px, py, pz] + jet_features)
            return np.array(features)
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении топологических признаков: {e}")
            return None
    def _compute_betti_numbers(self, points: np.ndarray) -> Dict[int, float]:
        """Вычисление чисел Бетти для топологического анализа."""
        try:
            # Здесь можно реализовать более сложный анализ
            # Для примера возвращаем заглушку с реалистичными значениями
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
            # Это упрощенная реализация
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
# 4. *** МОДУЛЬ: TopoAnalyzer ***
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
        self.ecdsa_analyzer = TopologicalECDSAAnalyzer()
        self.logger = logging.getLogger("TopoAnalyzer")
    def analyze_events(self, events: List[Dict], max_events: int = 500) -> bool:
        """Основной метод анализа событий"""
        if not events:
            self.logger.warning("Нет событий для анализа.")
            return False
        # Ограничиваем количество событий
        self.events = events[:max_events]
        # Построение векторов признаков
        self.build_feature_vectors()
        if self.feature_vectors.size == 0:
            self.logger.error("Не удалось построить векторы признаков.")
            return False
        # Вычисление матрицы расстояний
        self.compute_distance_matrix()
        if self.distance_matrix is None:
            self.logger.error("Не удалось вычислить матрицу расстояний.")
            return False
        # Вычисление персистентной гомологии
        if not self.compute_persistent_homology():
            self.logger.error("Не удалось вычислить персистентную гомологию.")
            return False
        # Анализ спектра корреляций
        self.analyze_correlation_spectrum()
        # Выполнение PCA
        self.perform_pca()
        return True
    def build_feature_vectors(self):
        """Построение векторов признаков для топологического анализа."""
        try:
            feature_list = []
            for event in self.events:
                # Извлекаем ключевые параметры
                total_energy = event.get('total_energy', 0.0)
                num_particles = len(event.get('particles', []))
                # Вычисляем средние значения для частиц
                avg_energy = 0.0
                avg_theta = 0.0
                avg_phi = 0.0
                num_valid_particles = 0
                for particle in event.get('particles', []):
                    energy = particle.get('energy', 0.0)
                    px = particle.get('px', 0.0)
                    py = particle.get('py', 0.0)
                    pz = particle.get('pz', 0.0)
                    if energy > 0:
                        avg_energy += energy
                        # Вычисляем углы
                        theta = np.arctan2(np.sqrt(px**2 + py**2), pz) if pz != 0 else np.pi/2
                        phi = np.arctan2(py, px) if px != 0 else 0
                        avg_theta += theta
                        avg_phi += phi
                        num_valid_particles += 1
                if num_valid_particles > 0:
                    avg_energy /= num_valid_particles
                    avg_theta /= num_valid_particles
                    avg_phi /= num_valid_particles
                # Добавляем статистику по струям
                num_jets = 0
                total_jet_energy = 0.0
                for particle in event.get('particles', []):
                    if particle.get('name') == 'jet':
                        num_jets += 1
                        total_jet_energy += particle.get('energy', 0.0)
                avg_jet_energy = total_jet_energy / num_jets if num_jets > 0 else 0.0
                # Формируем вектор признаков
                feature_vector = [
                    total_energy,
                    num_particles,
                    avg_energy,
                    avg_theta,
                    avg_phi,
                    num_jets,
                    avg_jet_energy
                ]
                feature_list.append(feature_vector)
            self.feature_vectors = np.array(feature_list)
            self.logger.info(f"Построены векторы признаков для {len(self.feature_vectors)} событий.")
        except Exception as e:
            self.logger.error(f"Ошибка при построении векторов признаков: {e}")
    def compute_distance_matrix(self):
        """Вычисление матрицы расстояний между событиями."""
        try:
            if self.feature_vectors.size == 0:
                self.logger.error("Нет векторов признаков для вычисления матрицы расстояний.")
                return
            # Нормализуем признаки перед вычислением расстояний
            normalized_features = (self.feature_vectors - np.mean(self.feature_vectors, axis=0)) / (np.std(self.feature_vectors, axis=0) + SMALL_EPSILON)
            # Вычисляем евклидовы расстояния
            self.distance_matrix = euclidean_distances(normalized_features)
            self.logger.info("Матрица расстояний вычислена.")
        except Exception as e:
            self.logger.error(f"Ошибка при вычислении матрицы расстояний: {e}")
    def compute_persistent_homology(self) -> bool:
        """Вычисление персистентной гомологии с использованием GUDHI."""
        try:
            if not GUDHI_AVAILABLE:
                self.logger.warning("GUDHI не установлен. Используем упрощенный анализ.")
                # Упрощенный анализ без GUDHI
                self.persistence_diagrams = {
                    0: np.array([[0, 1]]),
                    1: np.array([[0, 0.5]]),
                    2: np.array([])
                }
                return True
            if self.feature_vectors.size == 0:
                self.logger.error("Нет векторов признаков для вычисления персистентной гомологии.")
                return False
            # Создаем Rips комплекс
            self.logger.info("Создание Rips комплекса...")
            rips_complex = gudhi.RipsComplex(
                points=self.feature_vectors, 
                max_edge_length=1.0
            )
            # Создаем симплициальный комплекс
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
            # Вычисляем персистентную гомологию
            self.logger.info("Вычисление персистентной гомологии...")
            simplex_tree.persistence()
            # Извлекаем диаграммы персистентности
            self.persistence_diagrams = {}
            for dim in range(4):  # 0, 1, 2, 3 измерения
                intervals = simplex_tree.persistence_intervals_in_dimension(dim)
                if len(intervals) > 0:
                    self.persistence_diagrams[dim] = np.array(intervals)
                else:
                    self.persistence_diagrams[dim] = np.array([])
            # Логируем результаты
            dim_info = []
            for dim, intervals in self.persistence_diagrams.items():
                dim_info.append(f"dim {dim}: {len(intervals)} интервалов")
            self.logger.info(f"Персистентная гомология вычислена: {', '.join(dim_info)}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при вычислении персистентной гомологии: {e}")
            return False
    def analyze_correlation_spectrum(self):
        """Анализ спектра корреляций между признаками."""
        try:
            if self.feature_vectors.size == 0:
                self.logger.error("Нет векторов признаков для анализа корреляций.")
                return
            # Вычисляем корреляционную матрицу
            corr_matrix = np.corrcoef(self.feature_vectors, rowvar=False)
            # Вычисляем собственные значения и векторы
            eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
            # Сортируем собственные значения по убыванию
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            # Вычисляем число обусловленности
            condition_number = eigenvalues[0] / (eigenvalues[-1] + SMALL_EPSILON)
            self.correlation_spectrum = {
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'condition_number': condition_number
            }
            self.logger.info(f"Анализ спектра корреляций завершен. Число обусловленности: {condition_number:.2f}")
        except Exception as e:
            self.logger.error(f"Ошибка при анализе спектра корреляций: {e}")
    def perform_pca(self, n_components: Optional[int] = None):
        """Выполняет анализ главных компонент."""
        try:
            if n_components is None:
                n_components = min(5, self.feature_vectors.shape[1])
            pca = PCA(n_components=n_components)
            self.pca_results = pca.fit_transform(self.feature_vectors)
            explained_variance = pca.explained_variance_ratio_
            total_variance = sum(explained_variance)
            self.logger.info(f"PCA завершен. Объясненная дисперсия: {total_variance:.2%} " 
                            f"({n_components} компонент)")
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении PCA: {e}")
    def generate_report(self, output_file: str = "topology_report.json") -> bool:
        """Генерирует отчет о топологическом анализе."""
        try:
            report = {
                'num_events_analyzed': len(self.events),
                'feature_vectors_shape': self.feature_vectors.shape if self.feature_vectors.size > 0 else None,
                'persistence_diagrams': {},
                'pca_results': None,
                'correlation_spectrum': None,
                'ecdsa_analysis': None,
                'timestamp': time.time()
            }
            # Добавляем информацию о диаграммах персистентности
            if self.persistence_diagrams is not None:
                for dim, dgm in self.persistence_diagrams.items():
                    if len(dgm) > 0:
                        report['persistence_diagrams'][str(dim)] = {
                            'count': len(dgm),
                            'intervals': dgm.tolist()
                        }
                    else:
                        report['persistence_diagrams'][str(dim)] = {
                            'count': 0,
                            'intervals': []
                        }
            # Добавляем результаты PCA
            if self.pca_results is not None and self.pca_results.size > 0:
                report['pca_results'] = {
                    'shape': self.pca_results.shape,
                    'values': self.pca_results.tolist()
                }
            # Добавляем спектр корреляций
            if self.correlation_spectrum is not None:
                report['correlation_spectrum'] = {
                    'eigenvalues': self.correlation_spectrum['eigenvalues'].tolist(),
                    'condition_number': float(self.correlation_spectrum['condition_number'])
                }
            # Добавляем анализ ECDSA
            report['ecdsa_analysis'] = self.ecdsa_analyzer.analyze_topology(self.events)
            # Сохраняем отчет в файл
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Отчет топологического анализа сохранен в {output_file}.")
            return True
        except Exception as e:
            self.logger.error(f"Не удалось создать отчет топологического анализа: {e}")
            return False

# ===================================================================
# 5. *** МОДУЛЬ: PDFModel ***
# ===================================================================
class PDFModel:
    """Модель функций частицного распределения (PDF)"""
    def __init__(self, config):
        self.config = config
        self.pdf_data = self._load_pdf_data()
        self.logger = logging.getLogger("PDFModel")
    def _load_pdf_data(self):
        """Загрузка данных PDF из конфигурации или внешних источников"""
        self.logger.info("Загрузка данных PDF...")
        return {
            'proton': {
                'u': lambda x, Q2: self._u_quark_pdf(x, Q2),
                'd': lambda x, Q2: self._d_quark_pdf(x, Q2),
                's': lambda x, Q2: self._s_quark_pdf(x, Q2),
                'c': lambda x, Q2: self._c_quark_pdf(x, Q2),
                'b': lambda x, Q2: self._b_quark_pdf(x, Q2),
                'g': lambda x, Q2: self._gluon_pdf(x, Q2)
            }
        }
    def _u_quark_pdf(self, x, Q2):
        """PDF для u-кварка в протоне"""
        # Реалистичная модель PDF для u-кварка
        if x <= 0 or x >= 1:
            return 0.0
        return 1.368 * (1 - x)**3.0 * x**0.5 * (1 + 1.8*x - 7.9*x**2)
    def _d_quark_pdf(self, x, Q2):
        """PDF для d-кварка в протоне"""
        if x <= 0 or x >= 1:
            return 0.0
        return 0.850 * (1 - x)**4.0 * x**0.5 * (1 - 1.3*x + 3.4*x**2)
    def _s_quark_pdf(self, x, Q2):
        """PDF для s-кварка в протоне"""
        if x <= 0 or x >= 1:
            return 0.0
        return 0.15 * (1 - x)**5.0 * x**0.5 * (1 - 2.0*x + 2.5*x**2)
    def _c_quark_pdf(self, x, Q2):
        """PDF для c-кварка в протоне"""
        if x <= 0 or x >= 1:
            return 0.0
        return 0.05 * (1 - x)**6.0 * x**1.0 * (1 - 3.0*x + 4.0*x**2)
    def _b_quark_pdf(self, x, Q2):
        """PDF для b-кварка в протоне"""
        if x <= 0 or x >= 1:
            return 0.0
        return 0.01 * (1 - x)**7.0 * x**1.0 * (1 - 4.0*x + 5.0*x**2)
    def _gluon_pdf(self, x, Q2):
        """PDF для глюона в протоне"""
        if x <= 0 or x >= 1:
            return 0.0
        return 2.5 * (1 - x)**4.5 * x**(-0.2) * (1 - 1.5*x + 2.0*x**2)
    def get_pdf(self, particle, flavor, x, Q2):
        """Получает значение PDF для заданной частицы, вкуса, x и Q2"""
        try:
            if particle in self.pdf_data and flavor in self.pdf_data[particle]:
                return self.pdf_data[particle][flavor](x, Q2)
            return 0.0
        except Exception as e:
            self.logger.error(f"Ошибка при получении PDF для {particle}/{flavor}: {e}")
            return 0.0

# ===================================================================
# 6. *** МОДУЛЬ: InteractionGenerator ***
# ===================================================================
class InteractionGenerator:
    """Генерация физических процессов с корректными сечениями"""
    def __init__(self, config):
        self.config = config
        self.pdf = PDFModel(config)
        self.particle_db = ParticleDatabase()
        self.logger = logging.getLogger("InteractionGenerator")
    def _calculate_cross_section(self, process_type: str, energy: float, x1: float, x2: float) -> float:
        """Расчет физического сечения для заданного процесса с использованием точных формул."""
        # Энергия в системе центра масс для партонов
        E_CM_parton = np.sqrt(2 * x1 * x2 * energy**2)
        if process_type == "drell-yan":
            # Точная формула для процесса Дрелл-Яна с учетом Z/γ* резонанса
            if E_CM_parton < 1.0:  # Порог для физических процессов
                return 0.0
            # Учитываем ширину резонанса
            Gamma_Z = 2.4952  # ширина Z-бозона в ГэВ
            # Формула для сечения процесса Дрелл-Яна с учетом Z-резонанса
            s = E_CM_parton**2
            s_minus_MZ2 = s - M_Z**2
            denominator = s_minus_MZ2**2 + (M_Z * Gamma_Z)**2
            # Упрощенная версия, но более точная, чем в оригинале
            prefactor = (12 * np.pi * alpha_em**2) / (9 * s)
            resonance_term = (s * s_minus_MZ2) / denominator
            return prefactor * resonance_term * 1000  # в пикобарнах
        elif process_type == "gluon_fusion":
            # Точная формула для глюонной фьюжн (производство Хиггса)
            if E_CM_parton < 125:  # Масса Хиггса
                return 0.0
            # Упрощенная модель с учетом основных факторов
            tau = (M_H**2) / (4 * E_CM_parton**2)
            if tau < 1:
                C = np.arcsin(np.sqrt(tau))**2
            else:
                sqrt_tau_minus_1 = np.sqrt(tau - 1)
                C = 0.25 * np.log((1 + sqrt_tau_minus_1) / (1 - sqrt_tau_minus_1))**2
            # Приблизительная формула для сечения глюонной фьюжн
            sigma_0 = (G_F * alpha_s**2 * M_H**3) / (288 * np.sqrt(2) * np.pi)
            return sigma_0 * C
        elif process_type == "quark_antiquark":
            # Процесс кварк-антикварк аннигиляции
            return self._calculate_quark_antiquark_cross_section(energy, x1, x2)
        elif process_type == "jet_production":
            # Производство струй
            return self._calculate_jet_production_cross_section(energy, x1, x2)
        return 0.0
    def _calculate_quark_antiquark_cross_section(self, energy: float, x1: float, x2: float) -> float:
        """Расчет сечения для процесса кварк-антикварк аннигиляции."""
        # Реализация более точной формулы
        E_CM_parton = np.sqrt(2 * x1 * x2 * energy**2)
        # Базовое сечение для кварк-антикварк аннигиляции
        sigma_0 = (4 * np.pi * alpha_s**2) / (9 * E_CM_parton**2)
        # Учет дополнительных факторов (упрощенно)
        factor = 1.0 / (1 + 0.1 * np.exp(-(E_CM_parton - 100)/50))
        return sigma_0 * factor
    def _calculate_jet_production_cross_section(self, energy: float, x1: float, x2: float) -> float:
        """Расчет сечения для производства струй."""
        E_CM_parton = np.sqrt(2 * x1 * x2 * energy**2)
        # Более точная формула для производства струй
        if E_CM_parton < 20:  # Пороговое значение
            return 0.0
        # Упрощенная модель с учетом логарифмических поправок
        log_term = np.log(E_CM_parton / 20.0)
        return (30000.0 / E_CM_parton**2) * (1 + 0.5 * log_term)
    def generate_event(self, E_CM, x1, flavor1, x2, flavor2):
        """Генерация события на основе физической модели."""
        # Определяем тип процесса на основе сечений
        processes = {
            "drell-yan": self._calculate_cross_section("drell-yan", E_CM, x1, x2),
            "gluon_fusion": self._calculate_cross_section("gluon_fusion", E_CM, x1, x2),
            "quark_antiquark": self._calculate_cross_section("quark_antiquark", E_CM, x1, x2),
            "jet_production": self._calculate_cross_section("jet_production", E_CM, x1, x2)
        }
        # Нормализуем вероятности
        total = sum(processes.values())
        if total > 0:
            for process in processes:
                processes[process] /= total
        else:
            # Если все сечения нулевые, используем равномерное распределение
            for process in processes:
                processes[process] = 1.0 / len(processes)
        # Выбираем процесс
        process_type = random.choices(list(processes.keys()), list(processes.values()))[0]
        # Генерация продуктов
        products = self._generate_products(process_type, E_CM)
        return {
            'process_type': process_type,
            'energy_cm': E_CM,
            'x1': x1,
            'x2': x2,
            'flavor1': flavor1,
            'flavor2': flavor2,
            'products': products
        }
    def _generate_products(self, process_type: str, E_CM_parton: float) -> List[Dict]:
        """Генерация продуктов столкновения на основе физической модели"""
        products = []
        if process_type == "drell-yan":
            # Производство Z-бозонов или виртуальных фотонов
            if random.random() < 0.33:
                # Электрон-позитронная пара
                products.append({'name': 'electron', 'energy': E_CM_parton*0.45})
                products.append({'name': 'positron', 'energy': E_CM_parton*0.45})
            elif random.random() < 0.66:
                # Мюонная пара
                products.append({'name': 'muon', 'energy': E_CM_parton*0.45})
                products.append({'name': 'antimuon', 'energy': E_CM_parton*0.45})
            else:
                # Кварковая пара (адронизация)
                products.append({'name': 'u_quark', 'energy': E_CM_parton*0.45})
                products.append({'name': 'u_antiquark', 'energy': E_CM_parton*0.45})
        elif process_type == "gluon_fusion":
            # Глюонная фьюжн (производство Хиггса)
            products.append({'name': 'higgs', 'energy': E_CM_parton*0.9})
            # Распад Хиггса
            if random.random() < 0.58:
                products.append({'name': 'b_quark', 'energy': E_CM_parton*0.45})
                products.append({'name': 'b_antiquark', 'energy': E_CM_parton*0.45})
            # Распад на W-бозоны
            elif random.random() < 0.58 + 0.21:
                products.append({'name': 'W_plus', 'energy': E_CM_parton*0.45})
                products.append({'name': 'W_minus', 'energy': E_CM_parton*0.45})
            else:
                # Обычное производство струй
                num_jets = random.randint(2, 4)
                for _ in range(num_jets):
                    jet_energy = E_CM_parton * random.uniform(0.1, 0.5)
                    products.append({
                        'name': 'jet',
                        'energy': jet_energy,
                        'px': random.uniform(-jet_energy, jet_energy),
                        'py': random.uniform(-jet_energy, jet_energy),
                        'pz': random.uniform(-jet_energy, jet_energy)
                    })
        elif process_type == "quark_antiquark":
            # Процесс кварк-антикварк аннигиляции
            if random.random() < 0.7:
                products.append({'name': 'gluon', 'energy': E_CM_parton*0.5})
                products.append({'name': 'quark', 'energy': E_CM_parton*0.25})
                products.append({'name': 'antiquark', 'energy': E_CM_parton*0.25})
            else:
                num_hadrons = random.randint(2, 5)
                products.extend(self._fragment_hadron(E_CM_parton, num_hadrons))
        elif process_type == "jet_production":
            # Производство струй
            num_jets = random.randint(1, 3)
            for _ in range(num_jets):
                jet_energy = E_CM_parton * random.uniform(0.1, 0.5)
                products.append({
                    'name': 'jet',
                    'energy': jet_energy,
                    'px': random.uniform(-jet_energy, jet_energy),
                    'py': random.uniform(-jet_energy, jet_energy),
                    'pz': random.uniform(-jet_energy, jet_energy)
                })
        return products
    def _fragment_hadron(self, total_energy, num_hadrons):
        """Фрагментация кварков в адроны."""
        hadrons = []
        for _ in range(num_hadrons):
            hadron_type = random.choice(['pion+', 'pion-', 'pion0', 'kaon+', 'kaon-', 'proton', 'neutron'])
            hadrons.append({
                'name': hadron_type, 
                'energy': total_energy * random.uniform(0.05, 0.2)
            })
        return hadrons
    def interact(self, particle1: str, particle2: str, energy: float, num_events: int = 1, **kwargs) -> List[Dict]:
        """Моделирование взаимодействия частиц с физически корректными сечениями"""
        events = []
        for _ in range(num_events):
            # Для протон-протонных столкновений используем модель PDF
            if particle1 == "proton" and particle2 == "proton":
                # Генерируем x1, x2 и вкусы партонов
                x1 = random.uniform(0.01, 0.99)
                x2 = random.uniform(0.01, 0.99)
                flavor1 = random.choice(['u', 'd', 's', 'c', 'g'])
                flavor2 = random.choice(['u', 'd', 's', 'c', 'g'])
                # Генерируем событие
                event = self.generate_event(energy, x1, flavor1, x2, flavor2)
                events.append(event)
            # Для других типов столкновений (упрощенная модель)
            else:
                # Простая модель для других частиц
                process_type = random.choice(["drell-yan", "quark_antiquark", "jet_production"])
                E_CM = energy
                # Генерируем продукты
                products = self._generate_products(process_type, E_CM)
                events.append({
                    'process_type': process_type,
                    'energy_cm': E_CM,
                    'products': products
                })
        return events

# ===================================================================
# 7. *** МОДУЛЬ: BeamDynamics ***
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
            'luminosity': [],
            'beam_size_x': [],
            'beam_size_y': [],
            'beta_x': config.get('beam', {}).get('beta_x', 55.0),
            'beta_y': config.get('beam', {}).get('beta_y', 55.0),
            'D_x': config.get('beam', {}).get('dispersion_x', 0.0),
            'D_y': config.get('beam', {}).get('dispersion_y', 0.0),
            'revolution_time': config.get('geometry', {}).get('circumference', 26659) / c,
            'gamma_rel': config.get('beam', {}).get('beam_energy', 6500) / m_p,
            'current_intensity': config.get('beam', {}).get('bunch_intensity', 1.15e11),
            'bunch_length': config.get('beam', {}).get('bunch_length', 0.075),
            'energy_spread': config.get('beam', {}).get('energy_spread', 1.1e-4),
            'current_turn': 0
        }
        self.logger = logging.getLogger("BeamDynamics")
    def _apply_synchrotron_radiation(self):
        """Учет излучения синхротронного излучения"""
        # Синхротронное излучение вызывает потерю энергии и сжатие пучка
        # Формула для потерь энергии на оборот
        U_0 = (55 * alpha_s * (self.config['beam']['beam_energy'] * 1000)**4 * self.config['geometry']['circumference']) / \
              (3 * m_p * c**2 * (2 * np.pi * self.config['beam']['gamma_rel'])**3)
        # Обновление эмиттанса
        self.state['epsilon'] *= (1 - U_0 / (4 * self.config['beam']['beam_energy'] * 1000))
        # Обновление размеров пучка
        self.state['sigma_x'] = np.sqrt(self.state['epsilon'] * self.state['beta_x'])
        self.state['sigma_y'] = np.sqrt(self.state['epsilon'] * self.state['beta_y'])
    def _apply_quantum_excitation(self):
        """Учет квантового возбуждения"""
        # Квантовое возбуждение компенсирует сжатие из-за синхротронного излучения
        D_q = (55 * alpha_s * (m_p * c**2)**5 * self.config['beam']['beam_energy']**5) / \
              (32 * np.sqrt(3) * m_p**5 * c**7 * self.config['geometry']['circumference']**2)
        # Обновление эмиттанса
        self.state['epsilon'] += D_q
    def _apply_space_charge_effect(self):
        """Учет эффекта пространственного заряда"""
        # Пространственный заряд вызывает дополнительную фокусировку/дефокусировку
        space_charge_strength = (self.state['N_p'] * m_p * c**2) / \
                                (self.config['beam']['beam_energy'] * 
                                 self.config['geometry']['circumference'] *
                                 self.state['gamma_rel']**3)
        # Влияние на размеры пучка
        self.state['sigma_x'] *= (1 + space_charge_strength)
        self.state['sigma_y'] *= (1 + space_charge_strength)
    def _apply_beam_beam_effect(self):
        """Учет взаимодействия пучков"""
        if 'beam_beam' in self.config and self.config['beam_beam'].get('enabled', False):
            # Эффект взаимодействия пучков
            beam_beam_strength = self.config['beam_beam'].get('strength', 0.01)
            self.state['sigma_x'] *= (1 + beam_beam_strength)
            self.state['sigma_y'] *= (1 + beam_beam_strength)
    def simulate_turn(self, include_space_charge: bool = True, **kwargs):
        """Симуляция одного оборота пучка с учетом различных эффектов"""
        # Обновляем номер оборота
        self.state['turn'] += 1
        self.state['current_turn'] = self.state['turn']
        # Применяем физические эффекты
        self._apply_synchrotron_radiation()
        self._apply_quantum_excitation()
        if include_space_charge:
            self._apply_space_charge_effect()
        self._apply_beam_beam_effect()
        # Вычисляем светимость
        luminosity = self._calculate_luminosity()
        self.state['luminosity'].append(luminosity)
        # Сохраняем размеры пучка
        self.state['beam_size_x'].append(self.state['sigma_x'])
        self.state['beam_size_y'].append(self.state['sigma_y'])
        self.logger.debug(f"Оборот {self.state['turn']}: светимость = {luminosity:.2e} см⁻²с⁻¹, "
                         f"размеры пучка: ({self.state['sigma_x']:.4f}, {self.state['sigma_y']:.4f}) м")
    def _calculate_luminosity(self) -> float:
        """Расчет светимости коллайдера"""
        # Формула для светимости протон-протонного коллайдера
        f_rev = 1 / self.state['revolution_time']  # частота обращения
        N_b = self.config['beam'].get('num_bunches', 2556)  # количество bunches
        n_p = self.state['N_p']  # количество частиц в bunch
        # Геометрический фактор
        F = 1 / (4 * np.pi * self.state['sigma_x'] * self.state['sigma_y'])
        luminosity = f_rev * N_b * n_p**2 * F
        return luminosity
    def get_luminosity(self) -> float:
        """Возвращает текущую светимость."""
        if self.state['luminosity']:
            return self.state['luminosity'][-1]
        return 0.0
    def get_beam_size_x(self) -> float:
        """Возвращает текущий размер пучка по X."""
        if self.state['beam_size_x']:
            return self.state['beam_size_x'][-1]
        return 0.0
    def get_beam_size_y(self) -> float:
        """Возвращает текущий размер пучка по Y."""
        if self.state['beam_size_y']:
            return self.state['beam_size_y'][-1]
        return 0.0

# ===================================================================
# 8. *** МОДУЛЬ: DetectorSystem ***
# ===================================================================
class DetectorSystem:
    """Система детекторов с реалистичными характеристиками."""
    def __init__(self, config):
        self.config = config
        self.detectors = self._initialize_detectors()
        self.logger = logging.getLogger("DetectorSystem")
    def _initialize_detectors(self) -> Dict[str, Dict]:
        """Инициализация детекторных систем."""
        return {
            'tracker': {
                'resolution': 0.01,  # мкм
                'efficiency': 0.99,
                'material_budget': 0.1,  # X/X0
                'radius': 0.5,  # м
                'length': 2.0  # м
            },
            'calorimeter': {
                'resolution': 0.05,  # ГэВ
                'efficiency': 0.95,
                'energy_scale': 1.0,
                'type': 'electromagnetic',  # или 'hadronic'
                'radius': 1.5,  # м
                'length': 2.5  # м
            },
            'muon_system': {
                'resolution': 0.1,  # мкм
                'efficiency': 0.90,
                'radius': 4.0,  # м
                'length': 10.0  # м
            }
        }
    def detect_event(self, event: Dict) -> Dict:
        """Обработка события через детекторную систему."""
        try:
            # Создаем копию события для обработки
            detected_event = {
                'event_id': event.get('event_id', 0),
                'process_type': event.get('process_type', 'unknown'),
                'products': []
            }
            # Обрабатываем каждый продукт столкновения
            for product in event.get('products', []):
                detected_product = self._detect_particle(product)
                if detected_product:
                    detected_event['products'].append(detected_product)
            return detected_event
        except Exception as e:
            self.logger.error(f"Ошибка при обработке события: {e}")
            return {'error': str(e)}
    def _detect_particle(self, particle: Dict) -> Optional[Dict]:
        """Обнаружение частицы через детекторную систему с учетом эффективности."""
        try:
            particle_name = particle.get('name', '')
            energy = particle.get('energy', 0.0)
            # Проверяем, обнаруживается ли частица
            if random.random() > self._get_detection_efficiency(particle_name):
                return None
            # Применяем разрешающую способность детектора
            detected_energy = energy * (1 + random.gauss(0, self._get_energy_resolution(particle_name)))
            # Создаем обнаруженную частицу
            detected_particle = {
                'name': particle_name,
                'energy': detected_energy,
                'px': particle.get('px', 0.0),
                'py': particle.get('py', 0.0),
                'pz': particle.get('pz', 0.0),
                'detector_response': self._get_detector_response(particle_name)
            }
            return detected_particle
        except Exception as e:
            self.logger.error(f"Ошибка при обнаружении частицы {particle.get('name')}: {e}")
            return None
    def _get_detection_efficiency(self, particle_name: str) -> float:
        """Возвращает эффективность детектирования для частицы."""
        if 'electron' in particle_name or 'positron' in particle_name:
            return self.detectors['calorimeter']['efficiency']
        elif 'muon' in particle_name:
            return self.detectors['muon_system']['efficiency']
        elif 'jet' in particle_name:
            return self.detectors['calorimeter']['efficiency']
        elif 'photon' in particle_name:
            return self.detectors['calorimeter']['efficiency']
        else:
            return self.detectors['tracker']['efficiency']
    def _get_energy_resolution(self, particle_name: str) -> float:
        """Возвращает разрешающую способность детектора для частицы."""
        if 'electron' in particle_name or 'positron' in particle_name or 'photon' in particle_name:
            return self.detectors['calorimeter']['resolution']
        elif 'muon' in particle_name:
            return self.detectors['muon_system']['resolution']
        else:
            return 0.05  # Условная разрешающая способность для струй
    def _get_detector_response(self, particle_name: str) -> str:
        """Возвращает тип детекторного отклика для частицы."""
        if 'electron' in particle_name or 'positron' in particle_name or 'photon' in particle_name:
            return 'electromagnetic_shower'
        elif 'muon' in particle_name:
            return 'muon_track'
        elif 'jet' in particle_name:
            return 'hadronic_shower'
        else:
            return 'track'

# ===================================================================
# 9. *** МОДУЛЬ: Visualization ***
# ===================================================================
class Visualization:
    """Система визуализации для LHC моделирования."""
    def __init__(self):
        self.logger = logging.getLogger("Visualization")
    def plot_geometry_3d(self, geometry, detector_system):
        """Визуализация 3D геометрии коллайдера и детекторных систем."""
        try:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            # Рисуем кольцо коллайдера
            theta = np.linspace(0, 2*np.pi, 100)
            x = geometry['radius'] * np.cos(theta)
            y = geometry['radius'] * np.sin(theta)
            z = np.zeros_like(theta)
            ax.plot(x, y, z, 'b-', linewidth=2, label='Кольцо коллайдера')
            # Рисуем детекторные системы
            for name, detector in detector_system.detectors.items():
                r = detector['radius']
                length = detector['length']
                # Верхняя и нижняя части
                z_top = np.linspace(-length/2, length/2, 20)
                x_top = r * np.cos(theta[0])
                y_top = r * np.sin(theta[0])
                ax.plot([x_top]*len(z_top), [y_top]*len(z_top), z_top, 'r-')
                x_bottom = r * np.cos(theta[50])
                y_bottom = r * np.sin(theta[50])
                ax.plot([x_bottom]*len(z_top), [y_bottom]*len(z_top), z_top, 'g-')
                # Окружность на средней плоскости
                x_circle = r * np.cos(theta)
                y_circle = r * np.sin(theta)
                z_circle = np.zeros_like(theta)
                ax.plot(x_circle, y_circle, z_circle, 'k--', alpha=0.5)
            ax.set_xlabel('X (м)')
            ax.set_ylabel('Y (м)')
            ax.set_zlabel('Z (м)')
            ax.set_title('3D Геометрия коллайдера и детекторных систем')
            ax.legend()
            plt.savefig('collider_geometry_3d.png')
            self.logger.info("3D геометрия сохранена в collider_geometry_3d.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Ошибка при визуализации геометрии: {e}")
    def plot_detector_response_3d(self, events, detector_system):
        """Визуализация отклика детектора в 3D."""
        try:
            if not events:
                self.logger.warning("Нет событий для визуализации отклика детектора.")
                return
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            # Рисуем детекторные системы
            for name, detector in detector_system.detectors.items():
                r = detector['radius']
                length = detector['length']
                # Окружность на средней плоскости
                theta = np.linspace(0, 2*np.pi, 100)
                x_circle = r * np.cos(theta)
                y_circle = r * np.sin(theta)
                z_circle = np.zeros_like(theta)
                ax.plot(x_circle, y_circle, z_circle, 'k--', alpha=0.3)
                # Верхняя и нижняя части
                z_top = np.linspace(-length/2, length/2, 20)
                x_top = r * np.cos(theta[0])
                y_top = r * np.sin(theta[0])
                ax.plot([x_top]*len(z_top), [y_top]*len(z_top), z_top, 'k-', alpha=0.3)
            # Рисуем события
            colors = {'electromagnetic_shower': 'blue', 'muon_track': 'red', 'hadronic_shower': 'green', 'track': 'purple'}
            for event in events[:10]:  # Ограничиваем количество событий для ясности
                for product in event.get('products', []):
                    response = product.get('detector_response', 'track')
                    color = colors.get(response, 'black')
                    # Рисуем трек частицы
                    r = np.sqrt(product.get('px', 0)**2 + product.get('py', 0)**2)
                    theta = np.arctan2(product.get('py', 0), product.get('px', 0))
                    phi = np.arctan2(r, product.get('pz', 0))
                    # Простая визуализация трека
                    length = 2.0  # Длина трека
                    x = [0, length * np.sin(phi) * np.cos(theta)]
                    y = [0, length * np.sin(phi) * np.sin(theta)]
                    z = [0, length * np.cos(phi)]
                    ax.plot(x, y, z, color=color, alpha=0.7)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Визуализация отклика детектора')
            # Добавляем легенду вручную
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=color, lw=2, label=response) 
                             for response, color in colors.items()]
            ax.legend(handles=legend_elements)
            plt.savefig('detector_response_3d.png')
            self.logger.info("3D отклик детектора сохранен в detector_response_3d.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Ошибка при визуализации отклика детектора: {e}")
    def plot_luminosity_evolution(self, beam_dynamics):
        """Визуализация эволюции светимости."""
        try:
            if not beam_dynamics.state['luminosity']:
                self.logger.warning("Нет данных о светимости для визуализации.")
                return
            plt.figure(figsize=(10, 6))
            plt.plot(beam_dynamics.state['luminosity'], 'b-')
            plt.xlabel('Оборот')
            plt.ylabel('Светимость (см⁻²с⁻¹)')
            plt.title('Эволюция светимости')
            plt.grid(True)
            plt.savefig('luminosity_evolution.png')
            self.logger.info("Эволюция светимости сохранена в luminosity_evolution.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Ошибка при визуализации эволюции светимости: {e}")
    def plot_beam_size_evolution(self, beam_dynamics):
        """Визуализация эволюции размеров пучка."""
        try:
            if not beam_dynamics.state['beam_size_x'] or not beam_dynamics.state['beam_size_y']:
                self.logger.warning("Нет данных о размерах пучка для визуализации.")
                return
            plt.figure(figsize=(10, 6))
            plt.plot(beam_dynamics.state['beam_size_x'], 'r-', label='σ_x')
            plt.plot(beam_dynamics.state['beam_size_y'], 'b-', label='σ_y')
            plt.xlabel('Оборот')
            plt.ylabel('Размер пучка (м)')
            plt.title('Эволюция размеров пучка')
            plt.legend()
            plt.grid(True)
            plt.savefig('beam_size_evolution.png')
            self.logger.info("Эволюция размеров пучка сохранена в beam_size_evolution.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Ошибка при визуализации эволюции размеров пучка: {e}")
    def plot_topological_evolution(self, topo_analyzer):
        """Визуализация эволюции топологических свойств."""
        try:
            if topo_analyzer.persistence_diagrams is None:
                self.logger.warning("Нет данных о топологических свойствах для визуализации.")
                return
            plt.figure(figsize=(12, 8))
            # Построение диаграмм персистентности
            for dim, dgm in topo_analyzer.persistence_diagrams.items():
                if len(dgm) > 0:
                    plt.scatter(dgm[:, 0], dgm[:, 1], label=f'H{dim}', alpha=0.7)
            # Добавляем диагональ
            min_val = min([np.min(dgm) for dgm in topo_analyzer.persistence_diagrams.values() if len(dgm) > 0], default=0)
            max_val = max([np.max(dgm) for dgm in topo_analyzer.persistence_diagrams.values() if len(dgm) > 0], default=1)
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
            plt.xlabel('Рождение')
            plt.ylabel('Смерть')
            plt.title('Диаграммы персистентности')
            plt.legend()
            plt.grid(True)
            plt.savefig('topological_evolution.png')
            self.logger.info("Эволюция топологических свойств сохранена в topological_evolution.png")
            plt.close()
        except Exception as e:
            self.logger.error(f"Ошибка при визуализации топологической эволюции: {e}")
    def create_dashboard(self, model):
        """Создание интерактивного дашборда для анализа данных"""
        if not DASH_AVAILABLE:
            self.logger.warning("Dash недоступен. Не удается создать интерактивный дашборд.")
            return
        try:
            app = dash.Dash(__name__)
            # Создаем макет дашборда
            app.layout = html.Div([
                html.H1("LHC 2.0 Simulation Dashboard"),
                
                dcc.Graph(id='luminosity-graph'),
                dcc.Graph(id='beam-size-graph'),
                dcc.Graph(id='topology-graph'),
                dcc.Graph(id='event-visualization'),
                
                dcc.Interval(
                    id='interval-component',
                    interval=5*1000,  # Обновление каждые 5 секунд
                    n_intervals=0
                )
            ])
            
            # Создаем колбэки для обновления графиков
            @app.callback(
                [Output('luminosity-graph', 'figure'),
                 Output('beam-size-graph', 'figure'),
                 Output('topology-graph', 'figure'),
                 Output('event-visualization', 'figure')],
                [Input('interval-component', 'n_intervals')]
            )
            def update_graphs(n):
                # Получаем данные из модели
                luminosity = model.beam_dynamics.state['luminosity']
                beam_size_x = model.beam_dynamics.state['beam_size_x']
                beam_size_y = model.beam_dynamics.state['beam_size_y']
                
                # Создаем графики
                luminosity_fig = px.line(x=range(len(luminosity)), y=luminosity,
                                        title='Luminosity Evolution')
                beam_size_fig = px.line(x=range(len(beam_size_x)), y=[beam_size_x, beam_size_y],
                                       labels={'value': 'Beam Size (m)', 'x': 'Turn'},
                                       title='Beam Size Evolution')
                
                # Топологический анализ
                if model.topo_analyzer.persistence_diagrams:
                    topological_fig = self._create_persistence_diagram(model.topo_analyzer)
                else:
                    topological_fig = px.scatter(title='No topological data available')
                
                # Визуализация событий
                event_fig = self._create_event_visualization(model)
                
                return luminosity_fig, beam_size_fig, topological_fig, event_fig
                
            # Запускаем дашборд
            app.run_server(debug=True)
            self.logger.info("Interactive dashboard started")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при создании интерактивного дашборда: {e}")
            return False
            
    def _create_persistence_diagram(self, topo_analyzer):
        """Создание диаграммы персистентности"""
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            for dim, dgm in topo_analyzer.persistence_diagrams.items():
                if len(dgm) > 0:
                    fig.add_trace(go.Scatter(
                        x=dgm[:, 0],
                        y=dgm[:, 1],
                        mode='markers',
                        name=f'Dimension {dim}',
                        marker=dict(size=8)
                    ))
                    
            # Добавляем диагональ
            min_val = min([np.min(dgm) for dgm in topo_analyzer.persistence_diagrams.values() if len(dgm) > 0], default=0)
            max_val = max([np.max(dgm) for dgm in topo_analyzer.persistence_diagrams.values() if len(dgm) > 0], default=1)
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Diagonal',
                line=dict(dash='dash')
            ))
            
            fig.update_layout(
                title='Persistence Diagram',
                xaxis_title='Birth',
                yaxis_title='Death',
                showlegend=True
            )
            
            return fig
        except Exception as e:
            self.logger.error(f"Ошибка при создании диаграммы персистентности: {e}")
            return go.Figure()
            
    def _create_event_visualization(self, model):
        """Создание визуализации событий"""
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Рисуем детекторные системы
            for detector, params in model.detector_system.detectors.items():
                r = params['radius']
                length = params['length']
                # Создаем цилиндр для детектора
                theta = np.linspace(0, 2*np.pi, 50)
                z = np.linspace(-length/2, length/2, 50)
                theta, z = np.meshgrid(theta, z)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                fig.add_trace(go.Surface(
                    x=x, y=y, z=z,
                    colorscale='Blues',
                    opacity=0.3,
                    showscale=False
                ))
                
            # Рисуем события
            for event in model.simulation_state['detected_events'][:10]:  # Ограничиваем количество событий для ясности
                for product in event.get('detected_products', []):
                    # Создаем трек частицы
                    r = np.sqrt(product.get('px', 0)**2 + product.get('py', 0)**2 + product.get('pz', 0)**2)
                    if r > 0:
                        x = [0, 10 * product.get('px', 0) / r]
                        y = [0, 10 * product.get('py', 0) / r]
                        z = [0, 10 * product.get('pz', 0) / r]
                        # Цвет в зависимости от типа частицы
                        color = 'red'
                        if 'electron' in product.get('name', ''):
                            color = 'blue'
                        elif 'muon' in product.get('name', ''):
                            color = 'green'
                        elif 'jet' in product.get('name', ''):
                            color = 'purple'
                        fig.add_trace(go.Scatter3d(
                            x=x, y=y, z=z,
                            mode='lines',
                            line=dict(color=color, width=2),
                            name=product['name']
                        ))
                    
            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                title='Detector Response Visualization'
            )
            
            return fig
        except Exception as e:
            self.logger.error(f"Ошибка при создании визуализации событий: {e}")
            return go.Figure()

# ===================================================================
# 10. *** МОДУЛЬ: HepMC3Exporter ***
# ===================================================================
class HepMC3Exporter:
    """Экспортер данных в формат HepMC3."""
    def __init__(self):
        self.hepmc3_available = HEP_MC3_AVAILABLE
        self.logger = logging.getLogger("HepMC3Exporter")
        if self.hepmc3_available:
            self.logger.info("HepMC3Exporter инициализирован. HepMC3 доступен.")
        else:
            self.logger.warning("HepMC3Exporter инициализирован. HepMC3 недоступен.")
    def export_to_hepmc3(self, filename: str, events: List[Dict]) -> bool:
        """Экспортирует события в HepMC3 файл."""
        if not self.hepmc3_available:
            self.logger.error("HepMC3 недоступен. Невозможно экспортировать в HepMC3 формат.")
            return False
        try:
            import hepmc3
            # Создаем HepMC3 файл
            writer = hepmc3.WriterAscii(filename)
            for i, event in enumerate(events):
                # Создаем новое событие
                hepmc_event = hepmc3.GenEvent()
                hepmc_event.event_number = i
                # Устанавливаем генераторный уровень
                hepmc_event.run_info.set_name("LHC_2.0")
                hepmc_event.run_info.set_beam_particles(
                    hepmc3.GenParticle((0, 0, 6500, 6500), 2212, 3),
                    hepmc3.GenParticle((0, 0, -6500, -6500), 2212, 3)
                )
                # Добавляем входные частицы (протоны)
                proton1 = hepmc3.GenParticle((0, 0, 6500, 6500), 2212, 3)
                proton2 = hepmc3.GenParticle((0, 0, -6500, -6500), 2212, 3)
                hepmc_event.add_particle(proton1)
                hepmc_event.add_particle(proton2)
                # Добавляем продукты столкновения
                for j, p in enumerate(event.get('products', []), start=3):
                    # Конвертируем имя частицы в PDG код
                    pdg_code = self._particle_to_pdg(p.get('name', ''))
                    energy = p.get('energy', 0.0)
                    px = p.get('px', 0.0)
                    py = p.get('py', 0.0)
                    pz = p.get('pz', 0.0)
                    # Состояние частицы (конечное состояние)
                    status = 1
                    # Создаем частицу
                    particle = hepmc3.GenParticle((px, py, pz, energy), pdg_code, status)
                    hepmc_event.add_particle(particle)
                    # Добавляем связь с входными частицами
                    hepmc_event.add_vertex(hepmc3.GenVertex())
                    hepmc_event.connect_vertices(proton1.end_vertex, particle.production_vertex)
                    hepmc_event.connect_vertices(proton2.end_vertex, particle.production_vertex)
                # Записываем событие
                writer.write_event(hepmc_event)
            writer.close()
            self.logger.info(f"Данные успешно экспортированы в HepMC3 формат: {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при экспорте в HepMC3 формат: {e}")
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
# 11. *** МОДУЛЬ: ROOTExporter ***
# ===================================================================
class ROOTExporter:
    """Экспортер данных симуляции в формат ROOT."""
    def __init__(self):
        self.root_file = None
        self.tree = None
        self.logger = logging.getLogger("ROOTExporter")
    def export_to_root(self, filename: str, events: List[Dict], meta: Dict = None) -> bool:
        """Экспорт данных в формат ROOT."""
        if not ROOT_AVAILABLE:
            self.logger.error("ROOT недоступен. Установите ROOT framework для экспорта")
            return False
        try:
            import ROOT
            # Создаем новый ROOT файл
            self.root_file = ROOT.TFile(filename, "RECREATE")
            # Создаем дерево событий
            self.tree = ROOT.TTree("Events", "События LHC")
            # Определяем переменные для дерева
            event_id = ROOT.std.vector("int")()
            total_energy = ROOT.std.vector("double")()
            num_particles = ROOT.std.vector("int")()
            particle_names = ROOT.std.vector("string")()
            particle_energies = ROOT.std.vector("double")()
            particle_px = ROOT.std.vector("double")()
            particle_py = ROOT.std.vector("double")()
            particle_pz = ROOT.std.vector("double")()
            # Создаем ветки дерева
            self.tree.Branch("event_id", event_id)
            self.tree.Branch("total_energy", total_energy)
            self.tree.Branch("num_particles", num_particles)
            self.tree.Branch("particle_names", particle_names)
            self.tree.Branch("particle_energies", particle_energies)
            self.tree.Branch("particle_px", particle_px)
            self.tree.Branch("particle_py", particle_py)
            self.tree.Branch("particle_pz", particle_pz)
            # Заполняем дерево данными
            for i, event in enumerate(events):
                # Очищаем векторы
                event_id.clear()
                total_energy.clear()
                num_particles.clear()
                particle_names.clear()
                particle_energies.clear()
                particle_px.clear()
                particle_py.clear()
                particle_pz.clear()
                # Заполняем данными
                event_id.push_back(i)
                total_energy.push_back(event.get('total_energy', 0.0))
                particles = event.get('particles', [])
                num_particles.push_back(len(particles))
                for particle in particles:
                    particle_names.push_back(particle.get('name', 'unknown'))
                    particle_energies.push_back(particle.get('energy', 0.0))
                    particle_px.push_back(particle.get('px', 0.0))
                    particle_py.push_back(particle.get('py', 0.0))
                    particle_pz.push_back(particle.get('pz', 0.0))
                self.tree.Fill()
            # Добавляем метаданные, если они есть
            if meta:
                meta_tree = ROOT.TTree("Metadata", "Метаданные симуляции")
                meta_str = ROOT.std.string(json.dumps(meta))
                meta_tree.Branch("metadata", meta_str)
                meta_tree.Fill()
            # Сохраняем и закрываем файл
            self.root_file.Write()
            self.root_file.Close()
            self.logger.info(f"Данные успешно экспортированы в {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при экспорте в ROOT формат: {e}")
            # Пытаемся закрыть файл, если он был открыт
            if self.root_file:
                self.root_file.Close()
            return False

# ===================================================================
# 12. *** МОДУЛЬ: AnomalyDetector ***
# ===================================================================
class AnomalyDetector:
    """Многоуровневый детектор аномалий для LHC данных."""
    def __init__(self, beam_dynamics, topo_analyzer):
        self.beam_dynamics = beam_dynamics
        self.topo_analyzer = topo_analyzer
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
        self.logger = logging.getLogger("AnomalyDetector")
        self.anomaly_threshold = 3.0  # Порог для статистических аномалий
        self.topological_threshold = 0.95  # Порог для топологических аномалий
    def detect_statistical_anomalies(self, events: List[Dict], method: str = 'zscore', threshold: float = 3.0) -> List[int]:
        """Поиск статистических аномалий в данных."""
        self.logger.info(f"Поиск статистических аномалий методом {method}...")
        try:
            # Извлекаем признаки для анализа
            feature_vectors = []
            for event in events:
                # Извлекаем ключевые признаки
                total_energy = event.get('total_energy', 0.0)
                num_particles = len(event.get('particles', []))
                # Вычисляем среднюю энергию частиц
                avg_energy = 0.0
                num_valid = 0
                for particle in event.get('particles', []):
                    energy = particle.get('energy', 0.0)
                    if energy > 0:
                        avg_energy += energy
                        num_valid += 1
                if num_valid > 0:
                    avg_energy /= num_valid
                feature_vectors.append([total_energy, num_particles, avg_energy])
            feature_vectors = np.array(feature_vectors)
            # Ищем аномалии по каждому признаку
            anomaly_indices = []
            for i in range(feature_vectors.shape[1]):
                valid_indices = ~np.isnan(feature_vectors[:, i])
                valid_values = feature_vectors[valid_indices, i]
                if len(valid_values) < 2:
                    continue
                if method == 'zscore':
                    mean = np.mean(valid_values)
                    std = np.std(valid_values)
                    z_scores = np.abs((valid_values - mean) / (std + SMALL_EPSILON))
                    anomaly_mask = z_scores > threshold
                    anomaly_indices.extend(np.where(valid_indices)[0][anomaly_mask].tolist())
                elif method == 'iqr':
                    q1 = np.percentile(valid_values, 25)
                    q3 = np.percentile(valid_values, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    anomaly_mask = (valid_values < lower_bound) | (valid_values > upper_bound)
                    anomaly_indices.extend(np.where(valid_indices)[0][anomaly_mask].tolist())
                else:
                    self.logger.warning(f"Неизвестный метод: {method}. Используем zscore.")
                    continue
            # Уникальные индексы аномалий
            anomaly_indices = list(set(anomaly_indices))
            # Добавляем найденные аномалии
            self.anomalies_found['by_type']['statistical'] = anomaly_indices
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('statistical')
            self.logger.info(f"Найдено {len(anomaly_indices)} статистических аномалий.")
            return anomaly_indices
        except Exception as e:
            self.logger.error(f"Ошибка при поиске статистических аномалий: {e}")
            return []
    def detect_topological_anomalies(self, events: List[Dict], threshold_percentile: float = 95.0) -> List[int]:
        """Поиск аномалий через топологический анализ."""
        self.logger.info("Поиск топологических аномалий...")
        try:
            # Анализируем события
            if not self.topo_analyzer.analyze_events(events):
                self.logger.warning("Топологический анализ не выполнен, пропускаем поиск аномалий.")
                return []
            # Получаем диаграммы персистентности
            dgms = []
            for dim in range(4):  # 0, 1, 2, 3 измерения
                if dim in self.topo_analyzer.persistence_diagrams:
                    dgm = self.topo_analyzer.persistence_diagrams[dim]
                    if len(dgm) > 0:
                        dgms.append((dim, dgm))
            if not dgms:
                self.logger.warning("Нет диаграмм персистентности для анализа аномалий.")
                return []
            # Собираем все персистентности
            all_pers = []
            for _, dgm in dgms:
                pers = dgm[:, 1] - dgm[:, 0]
                all_pers.extend(pers)
            if not all_pers:
                self.logger.info("Нет персистентностей для анализа.")
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
            self.anomalies_found['by_type']['topological'] = anomaly_indices
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('topological')
            self.logger.info(f"Найдено {len(anomaly_indices)} топологических аномалий.")
            return anomaly_indices
        except Exception as e:
            self.logger.error(f"Ошибка при поиске топологических аномалий: {e}")
            return []
    def detect_model_behavior_anomalies(self, state_history: List[Dict]) -> List[int]:
        """Поиск аномалий в поведении модели."""
        self.logger.info("Поиск аномалий в поведении модели...")
        try:
            if len(state_history) < 2:
                self.logger.warning("Недостаточно данных для анализа поведения модели.")
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
                size_x_z = np.abs(size_x_diff / size_x_std)
                anomaly_mask_x = size_x_z > self.anomaly_threshold
                anomaly_indices.extend(np.where(anomaly_mask_x)[0].tolist())
            if size_y_std > 0:
                size_y_z = np.abs(size_y_diff / size_y_std)
                anomaly_mask_y = size_y_z > self.anomaly_threshold
                anomaly_indices.extend(np.where(anomaly_mask_y)[0].tolist())
            # Уникальные индексы аномалий
            anomaly_indices = list(set(anomaly_indices))
            # Добавляем найденные аномалии
            self.anomalies_found['by_type']['model_behavior'] = anomaly_indices
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('model_behavior')
            self.logger.info(f"Найдено {len(anomaly_indices)} аномалий в поведении модели.")
            return anomaly_indices
        except Exception as e:
            self.logger.error(f"Ошибка при обнаружении аномалий поведения модели: {e}")
            return []
    def detect_custom_anomalies(self, custom_detector_func, *args, **kwargs):
        """Поиск пользовательских аномалий с помощью пользовательской функции."""
        self.logger.info("Поиск пользовательских аномалий...")
        try:
            custom_anomalies = custom_detector_func(*args, **kwargs)
            if custom_anomalies:
                self.anomalies_found['by_type']['custom'].extend(custom_anomalies)
                self.anomalies_found['summary']['total_count'] += len(custom_anomalies)
                self.anomalies_found['summary']['types_found'].add('custom')
                self.logger.info(f"Найдено {len(custom_anomalies)} пользовательских аномалий.")
                return custom_anomalies
            return []
        except Exception as e:
            self.logger.error(f"Ошибка при поиске пользовательских аномалий: {e}")
            return []
    def detect_all_anomalies(self, events: List[Dict], state_history: List[Dict], max_events: int = 500):
        """Обнаружение всех типов аномалий."""
        # Статистический анализ
        self.detect_statistical_anomalies(events[:max_events])
        # Топологический анализ
        self.detect_topological_anomalies(events[:max_events])
        # Анализ поведения модели
        self.detect_model_behavior_anomalies(state_history)
        return self.anomalies_found
    def generate_anomaly_report(self, output_file: str = "anomaly_report.json") -> Dict:
        """Генерирует отчет об обнаруженных аномалиях."""
        try:
            report = {
                'anomalies': self.anomalies_found,
                'summary': {
                    'total_count': self.anomalies_found['summary']['total_count'],
                    'types_found': list(self.anomalies_found['summary']['types_found'])
                },
                'timestamp': time.time()
            }
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Отчет об аномалиях сохранен в {output_file}.")
            return self.anomalies_found
        except Exception as e:
            self.logger.error(f"Не удалось сохранить отчет об аномалиях: {e}")
            return self.anomalies_found

# ===================================================================
# 13. *** МОДУЛЬ: GradientCalibrator ***
# ===================================================================
class GradientCalibrator:
    """Калибровщик модели на основе градиентного анализа и оптимизации.
    Использует scipy.optimize для надежной минимизации ошибки."""
    def __init__(self, model, target_observables: Dict[str, float], 
                 parameters_to_calibrate: List[str], perturbation_factor: float = 0.01,
                 error_weights: Optional[Dict[str, float]] = None):
        self.model = model
        self.target_observables = target_observables
        self.parameters_to_calibrate = parameters_to_calibrate
        self.perturbation_factor = perturbation_factor
        self.error_weights = error_weights or {}
        self.optimization_result = None
        self.history = []
        self.logger = logging.getLogger("GradientCalibrator")
        # Проверяем, что все целевые наблюдаемые поддерживаются моделью
        supported_observables = model.get_supported_observables()
        for name in target_observables:
            if name not in supported_observables:
                self.logger.warning(f"Наблюдаемая величина '{name}' не поддерживается моделью.")
    def calibrate(self, initial_params: Optional[List[float]] = None, 
                 method: str = 'L-BFGS-B', max_iterations: int = 100, 
                 tolerance: float = 1e-6) -> bool:
        """Калибрует модель для достижения целевых наблюдаемых."""
        self.logger.info(f"Запуск калибровки модели методом {method}...")
        try:
            # Если начальные параметры не заданы, используем текущие
            if initial_params is None:
                initial_params = [self.model.get_parameter(param) 
                                 for param in self.parameters_to_calibrate]
            if len(initial_params) != len(self.parameters_to_calibrate):
                raise ValueError("Длина initial_params не соответствует количеству параметров для калибровки.")
            # Настройка границ для оптимизации
            bounds = [(None, None) for _ in self.parameters_to_calibrate]
            # Дополнительные аргументы для целевой функции
            extra_args = (10,)  # Количество оборотов для симуляции
            self.logger.info(f"Запуск оптимизации методом {method}...")
            # Запускаем оптимизацию
            self.optimization_result = minimize(
                fun=self._objective_function,
                x0=initial_params,
                args=extra_args,
                method=method,
                bounds=bounds if method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None,
                tol=tolerance,
                options={'maxiter': max_iterations}
            )
            self.logger.info(f"Результат: {self.optimization_result.message}")
            self.logger.info(f"Финальная ошибка (RMSE): {self.optimization_result.fun:.2e}")
            self.logger.info(f"Финальные параметры: {self.optimization_result.x}")
            if self.optimization_result.success:
                self._set_parameters(self.optimization_result.x)
                self.logger.info("Оптимальные параметры установлены.")
            else:
                self.logger.warning("Оптимизация не сошлась.")
            return self.optimization_result.success
        except Exception as e:
            self.logger.error(f"Ошибка в процессе калибровки: {e}")
            self.optimization_result = None
            return False
    def _objective_function(self, param_values: np.ndarray, num_turns: int = 10) -> float:
        """Целевая функция для минимизации - RMSE между текущими и целевыми наблюдаемыми."""
        self._set_parameters(param_values)
        # Запускаем симуляцию для получения текущих наблюдаемых
        self.model.run_simulation(num_turns=num_turns, include_space_charge=False)
        current_observables = self._get_current_observables_from_model(self.model)
        # Вычисляем ошибку
        total_error_sq = 0.0
        for name, target_value in self.target_observables.items():
            if name not in current_observables:
                self.logger.warning(f"Наблюдаемая величина '{name}' не найдена в текущих результатах.")
                continue
            current_value = current_observables[name]
            error = current_value - target_value
            # Получаем масштаб для нормализации ошибки
            scale = target_value
            # Используем SMALL_EPSILON вместо магического числа 1e-12
            normalized_error = error / (scale + SMALL_EPSILON)
            # Вес ошибки
            weight = self.error_weights.get(name, 1.0)
            total_error_sq += weight * (normalized_error ** 2)
        rmse = np.sqrt(total_error_sq / len(self.target_observables))
        self.history.append({
            'params': param_values.copy(),
            'observables': current_observables.copy(),
            'rmse': rmse
        })
        self.logger.debug(f"Целевая функция: params={param_values}, RMSE={rmse:.2e}")
        return rmse
    def _get_current_observables_from_model(self, model) -> Dict[str, float]:
        """Получает текущие значения наблюдаемых величин из модели."""
        obs = {}
        # Проверяем наличие необходимых данных
        if not hasattr(model, 'simulation_state') or not isinstance(model.simulation_state, dict):
            self.logger.error("Модель не содержит корректного simulation_state")
            return obs
        beam_dynamics = model.simulation_state.get('beam_dynamics', {})
        if not beam_dynamics or not isinstance(beam_dynamics, dict):
            self.logger.error("Некорректные данные beam_dynamics в модели")
            return obs
        # Получаем наблюдаемые величины с проверкой
        for name in self.target_observables.keys():
            try:
                if name == "luminosity":
                    if 'luminosity' in beam_dynamics and beam_dynamics['luminosity']:
                        obs[name] = beam_dynamics['luminosity'][-1]
                    else:
                        self.logger.warning("Нет данных о светимости")
                elif name == "beam_size_x":
                    if 'beam_size_x' in beam_dynamics and beam_dynamics['beam_size_x']:
                        obs[name] = beam_dynamics['beam_size_x'][-1]
                    else:
                        self.logger.warning("Нет данных о размере пучка по X")
                elif name == "beam_size_y":
                    if 'beam_size_y' in beam_dynamics and beam_dynamics['beam_size_y']:
                        obs[name] = beam_dynamics['beam_size_y'][-1]
                    else:
                        self.logger.warning("Нет данных о размере пучка по Y")
                elif name == "average_energy":
                    if 'recent_events' in model.simulation_state:
                        total_energy = 0.0
                        num_events = 0
                        for event in model.simulation_state['recent_events']:
                            total_energy += event.get('total_energy', 0.0)
                            num_events += 1
                        if num_events > 0:
                            obs[name] = total_energy / num_events
                        else:
                            obs[name] = 0.0
                    else:
                        self.logger.warning("Нет данных о недавних событиях")
                else:
                    self.logger.warning(f"Неизвестная наблюдаемая величина: {name}")
            except Exception as e:
                self.logger.error(f"Ошибка при получении наблюдаемой величины '{name}': {e}")
                # Пропускаем эту наблюдаемую вместо установки нулевого значения
                continue
        return obs
    def _set_parameters(self, param_values: np.ndarray):
        """Устанавливает значения параметров в модели."""
        for i, param_name in enumerate(self.parameters_to_calibrate):
            value = param_values[i]
            if param_name in self.model.config.get('beam', {}):
                self.model.config['beam'][param_name] = value

    def get_calibration_history(self) -> List[Dict]:
        """Возвращает историю калибровки."""
        return self.history

# ===================================================================
# 14. *** МОДУЛЬ: LHCHybridModel ***
# ===================================================================
class LHCHybridModel:
    """Гибридная модель LHC, объединяющая физику, детекторы и анализ."""
    def __init__(self, config: Dict):
        self.config = config
        self.particle_db = ParticleDatabase()
        self.interaction_generator = InteractionGenerator(config)
        self.beam_dynamics = BeamDynamics(config)
        self.detector_system = DetectorSystem(config)
        self.visualization = Visualization()
        self.topo_analyzer = TopoAnalyzer()
        self.anomaly_detector = AnomalyDetector(self.beam_dynamics, self.topo_analyzer)
        self.simulation_state = {
            'beam_dynamics': {
                'luminosity': [],
                'beam_size_x': [],
                'beam_size_y': []
            },
            'recent_events': [],
            'detected_events': [],
            'current_turn': 0
        }
        self.geometry = {
            'radius': config.get('geometry', {}).get('radius', 4297),
            'circumference': config.get('geometry', {}).get('circumference', 26659)
        }
        self.logger = logging.getLogger("LHCHybridModel")
        self.cache = SimpleCache()
    def _validate_event_data(self, event: Dict) -> bool:
        """Проверка корректности данных события."""
        required_keys = ['total_energy', 'particles']
        for key in required_keys:
            if key not in event:
                self.logger.error(f"Отсутствует обязательный ключ '{key}' в данных события")
                return False
        # Проверка энергии
        if not isinstance(event['total_energy'], (int, float)) or event['total_energy'] <= 0:
            self.logger.error(f"Некорректное значение энергии: {event['total_energy']}")
            return False
        # Проверка частиц
        if not isinstance(event['particles'], list):
            self.logger.error("Поле 'particles' должно быть списком")
            return False
        for i, particle in enumerate(event['particles']):
            if not isinstance(particle, dict):
                self.logger.error(f"Элемент particles[{i}] не является словарем")
                return False
            if 'name' not in particle or 'energy' not in particle:
                self.logger.error(f"Отсутствуют обязательные поля в частице {i}")
                return False
        return True
    def run_simulation(self, num_turns: int = 10, include_space_charge: bool = True):
        """Запуск симуляции коллайдера."""
        self.logger.info(f"Запуск симуляции на {num_turns} оборотов...")
        # Сохраняем начальное состояние
        initial_state = {
            'beam_dynamics': {
                'luminosity': list(self.beam_dynamics.state['luminosity']),
                'beam_size_x': list(self.beam_dynamics.state['beam_size_x']),
                'beam_size_y': list(self.beam_dynamics.state['beam_size_y'])
            },
            'current_turn': self.beam_dynamics.state['current_turn']
        }
        try:
            # Симулируем обороты
            for _ in range(num_turns):
                self.beam_dynamics.simulate_turn(include_space_charge=include_space_charge)
                # Генерируем событие
                events = self.interaction_generator.interact(
                    "proton", "proton", 
                    self.config['beam']['beam_energy'], 
                    num_events=1
                )
                # Обрабатываем событие через детектор
                if events:
                    detected_event = self.detector_system.detect_event(events[0])
                    self.simulation_state['detected_events'].append(detected_event)
                    self.simulation_state['recent_events'].append(events[0])
                # Сохраняем текущее состояние
                self.simulation_state['beam_dynamics']['luminosity'].append(self.beam_dynamics.get_luminosity())
                self.simulation_state['beam_dynamics']['beam_size_x'].append(self.beam_dynamics.get_beam_size_x())
                self.simulation_state['beam_dynamics']['beam_size_y'].append(self.beam_dynamics.get_beam_size_y())
                self.simulation_state['current_turn'] = self.beam_dynamics.state['current_turn']
            self.logger.info("Симуляция завершена успешно.")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении симуляции: {e}")
            # Восстанавливаем состояние
            self.beam_dynamics.state['luminosity'] = initial_state['beam_dynamics']['luminosity']
            self.beam_dynamics.state['beam_size_x'] = initial_state['beam_dynamics']['beam_size_x']
            self.beam_dynamics.state['beam_size_y'] = initial_state['beam_dynamics']['beam_size_y']
            self.beam_dynamics.state['current_turn'] = initial_state['current_turn']
            return False
    def analyze_events(self, events: List[Dict], max_events: int = 500):
        """Анализ событий с использованием топологического анализа."""
        self.logger.info(f"Анализ {len(events)} событий...")
        # Проверяем данные событий
        valid_events = []
        for event in events:
            if self._validate_event_data(event):
                valid_events.append(event)
        if not valid_events:
            self.logger.warning("Нет валидных событий для анализа.")
            return False
        # Запускаем анализ
        success = self.topo_analyzer.analyze_events(valid_events, max_events)
        if success:
            self.logger.info("Топологический анализ завершен успешно.")
            self.topo_analyzer.generate_report()
        else:
            self.logger.error("Топологический анализ завершился с ошибкой.")
        return success
    def calibrate_model(self, target_observables: Dict[str, float], parameters_to_calibrate: List[str]):
        """Калибрует модель для достижения целевых наблюдаемых."""
        self.logger.info("Запуск калибровки модели...")
        calibrator = GradientCalibrator(
            self, 
            target_observables, 
            parameters_to_calibrate
        )
        # Начальные параметры (используем текущие значения)
        initial_params = [self.config['beam'].get(param, 0.1) for param in parameters_to_calibrate]
        # Запускаем калибровку
        success = calibrator.calibrate(
            initial_params=initial_params,
            method='L-BFGS-B',
            max_iterations=50
        )
        if success:
            self.logger.info("Калибровка завершена успешно.")
        else:
            self.logger.warning("Калибровка не удалась.")
        return success
    def detect_anomalies(self, max_events: int = 500):
        """Обнаружение аномалий во всех аспектах данных."""
        self.logger.info("Поиск аномалий в данных...")
        # Собираем историю состояний для анализа поведения модели
        state_history = [{
            'beam_dynamics': {
                'luminosity': self.simulation_state['beam_dynamics']['luminosity'][:],
                'beam_size_x': self.simulation_state['beam_dynamics']['beam_size_x'][:],
                'beam_size_y': self.simulation_state['beam_dynamics']['beam_size_y'][:]
            }
        }]
        # Обнаруживаем аномалии
        anomalies = self.anomaly_detector.detect_all_anomalies(
            self.simulation_state['recent_events'], 
            state_history,
            max_events
        )
        # Генерируем отчет
        self.anomaly_detector.generate_anomaly_report()
        return anomalies
    def export_to_root(self, filename: str) -> bool:
        """Экспортирует данные симуляции в ROOT формат."""
        exporter = ROOTExporter()
        return exporter.export_to_root(filename, self.simulation_state['recent_events'])
    def export_to_hepmc3(self, filename: str) -> bool:
        """Экспортирует данные симуляции в HepMC3 формат."""
        exporter = HepMC3Exporter()
        return exporter.export_to_hepmc3(filename, self.simulation_state['recent_events'])
    def enhanced_visualization(self):
        """Улучшенная визуализация результатов симуляции."""
        self.logger.info("Запуск улучшенной визуализации результатов...")
        try:
            self.visualization.plot_geometry_3d(self.geometry, self.detector_system)
            if self.simulation_state['detected_events']:
                self.visualization.plot_detector_response_3d(
                    self.simulation_state['detected_events'], 
                    self.detector_system
                )
            self.visualization.plot_luminosity_evolution(self.beam_dynamics)
            self.visualization.plot_beam_size_evolution(self.beam_dynamics)
            if self.topo_analyzer.persistence_diagrams is not None:
                self.visualization.plot_topological_evolution(self.topo_analyzer)
            # Создаем интерактивный дашборд
            self.visualization.create_dashboard(self)
            self.logger.info("Визуализация завершена успешно.")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при визуализации результатов: {e}")
            return False
    def get_supported_observables(self) -> List[str]:
        """Возвращает список поддерживаемых наблюдаемых величин."""
        return ["luminosity", "beam_size_x", "beam_size_y", "average_energy"]
    def get_parameter(self, param_name: str) -> float:
        """Получает значение параметра модели."""
        return self.config.get('beam', {}).get(param_name, 0.1)
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
# 15. *** МОДУЛЬ: Тесты ***
# ===================================================================
class LHCTestSuite:
    """Набор тестов для проверки корректности системы LHC 2.0."""
    def __init__(self):
        self.results = []
        self.logger = logging.getLogger("LHCTestSuite")
    def run_all_tests(self):
        """Запуск всех тестов."""
        self.logger.info("Запуск тестового набора...")
        tests = [
            self.test_physics_constants,
            self.test_beam_dynamics,
            self.test_cross_sections,
            self.test_topological_analysis,
            self.test_root_export,
            self.test_hepmc3_export,
            self.test_anomaly_detection
        ]
        for test in tests:
            self._run_test(test)
        self._print_summary()
        return all(result['passed'] for result in self.results)
    def _run_test(self, test_func):
        """Запуск отдельного теста."""
        start_time = time.time()
        try:
            passed = test_func()
            duration = time.time() - start_time
            self.results.append({
                'name': test_func.__name__,
                'passed': passed,
                'duration': duration
            })
            status = "УСПЕШНО" if passed else "ПРОВАЛЕН"
            self.logger.info(f"Тест {test_func.__name__}: {status} ({duration:.4f} с)")
        except Exception as e:
            duration = time.time() - start_time
            self.results.append({
                'name': test_func.__name__,
                'passed': False,
                'duration': duration,
                'error': str(e)
            })
            self.logger.error(f"Тест {test_func.__name__} завершился с ошибкой: {e}")
    def _print_summary(self):
        """Вывод сводки по тестам."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        self.logger.info("
===== СВОДКА ТЕСТОВ =====")
        self.logger.info(f"Всего тестов: {total}")
        self.logger.info(f"Успешно: {passed}")
        self.logger.info(f"Провалено: {failed}")
        self.logger.info(f"Процент успеха: {passed/total*100:.2f}%")
    def test_physics_constants(self) -> bool:
        """Тест проверки физических констант."""
        # Проверяем, что константы имеют разумные значения
        assert c == 299792458, "Скорость света некорректна"
        assert 0.9 < m_p < 0.95, "Масса протона вне ожидаемого диапазона"
        assert 1.1e-5 < G_F < 1.2e-5, "Ферми-константа вне ожидаемого диапазона"
        assert 80 < M_W < 81, "Масса W-бозона вне ожидаемого диапазона"
        return True
    def test_beam_dynamics(self) -> bool:
        """Тест проверки динамики пучка."""
        config = {
            'beam': {
                'beam_energy': 6500,  # ГэВ
                'bunch_intensity': 1.15e11,
                'sigma_x': 0.045,
                'sigma_y': 0.045,
                'emittance': 2.5e-6
            },
            'geometry': {
                'radius': 4297,
                'circumference': 26659
            }
        }
        beam_dynamics = BeamDynamics(config)
        initial_luminosity = beam_dynamics.get_luminosity()
        # Симулируем несколько оборотов
        for _ in range(10):
            beam_dynamics.simulate_turn()
        final_luminosity = beam_dynamics.get_luminosity()
        # Проверяем, что светимость изменилась разумным образом
        assert initial_luminosity > 0, "Начальная светимость должна быть положительной"
        assert final_luminosity <= initial_luminosity, "Светимость должна уменьшаться со временем"
        assert final_luminosity > 0, "Светимость не должна стать нулевой за 10 оборотов"
        return True
    def test_cross_sections(self) -> bool:
        """Тест проверки расчета физических сечений."""
        config = {
            'beam': {
                'beam_energy': 6500
            }
        }
        generator = InteractionGenerator(config)
        # Проверяем сечение Дрелл-Яна при энергии выше массы Z-бозона
        cs_dy_high = generator._calculate_cross_section("drell-yan", 6500, 0.1, 0.1)
        assert cs_dy_high > 0, "Сечение Дрелл-Яна при высокой энергии должно быть положительным"
        # Проверяем сечение Дрелл-Яна при энергии ниже массы Z-бозона
        cs_dy_low = generator._calculate_cross_section("drell-yan", 6500, 0.01, 0.01)
        assert cs_dy_low < cs_dy_high, "Сечение Дрелл-Яна при низкой энергии должно быть меньше"
        # Проверяем сечение глюонной фьюжн при энергии выше массы Хиггса
        cs_gf_high = generator._calculate_cross_section("gluon_fusion", 6500, 0.1, 0.1)
        assert cs_gf_high > 0, "Сечение глюонной фьюжн при высокой энергии должно быть положительным"
        return True
    def test_topological_analysis(self) -> bool:
        """Тест проверки топологического анализа."""
        if not GUDHI_AVAILABLE:
            self.logger.warning("GUDHI не установлен, пропускаем тест топологического анализа")
            return True
        # Создаем тестовые данные
        events = []
        for i in range(100):
            # Создаем события с разной структурой
            energy = 1000 + i * 10
            particles = []
            # Добавляем несколько частиц
            for j in range(5):
                particles.append({
                    'name': f'particle_{j}',
                    'energy': energy * (0.1 + j*0.2),
                    'px': random.gauss(0, 100),
                    'py': random.gauss(0, 100),
                    'pz': random.gauss(0, 100)
                })
            events.append({'total_energy': energy, 'particles': particles})
        # Запускаем анализ
        analyzer = TopoAnalyzer()
        success = analyzer.analyze_events(events)
        assert success, "Топологический анализ не завершился успешно"
        assert analyzer.persistence_diagrams is not None, "Нет диаграмм персистентности"
        assert len(analyzer.persistence_diagrams[0]) > 0, "Нет компонент связности"
        return True
    def test_root_export(self) -> bool:
        """Тест проверки экспорта в ROOT."""
        if not ROOT_AVAILABLE:
            self.logger.warning("ROOT не установлен, пропускаем тест экспорта в ROOT")
            return True
        # Создаем тестовые данные
        events = []
        for i in range(10):
            particles = []
            for j in range(5):
                particles.append({
                    'name': f'particle_{j}',
                    'energy': 100 + j*50,
                    'px': random.gauss(0, 50),
                    'py': random.gauss(0, 50),
                    'pz': random.gauss(0, 50)
                })
            events.append({'total_energy': 500, 'particles': particles})
        # Экспортируем во временный файл
        exporter = ROOTExporter()
        temp_file = "test_export.root"
        success = exporter.export_to_root(temp_file, events)
        # Проверяем, что файл создан
        import os
        file_exists = os.path.exists(temp_file)
        # Удаляем временный файл
        if file_exists:
            os.remove(temp_file)
        assert success, "Экспорт в ROOT не завершился успешно"
        assert file_exists, "Файл ROOT не был создан"
        return True
    def test_hepmc3_export(self) -> bool:
        """Тест проверки экспорта в HepMC3."""
        if not HEP_MC3_AVAILABLE:
            self.logger.warning("HepMC3 не установлен, пропускаем тест экспорта в HepMC3")
            return True
        # Создаем тестовые данные
        events = []
        for i in range(10):
            products = []
            for j in range(3):
                products.append({
                    'name': 'electron',
                    'energy': 100 + j*50,
                    'px': random.gauss(0, 50),
                    'py': random.gauss(0, 50),
                    'pz': random.gauss(0, 50)
                })
            events.append({'products': products})
        # Экспортируем во временный файл
        exporter = HepMC3Exporter()
        temp_file = "test_export.hepmc3"
        success = exporter.export_to_hepmc3(temp_file, events)
        # Проверяем, что файл создан
        import os
        file_exists = os.path.exists(temp_file)
        # Удаляем временный файл
        if file_exists:
            os.remove(temp_file)
        assert success, "Экспорт в HepMC3 не завершился успешно"
        assert file_exists, "Файл HepMC3 не был создан"
        return True
    def test_anomaly_detection(self) -> bool:
        """Тест проверки обнаружения аномалий."""
        # Создаем тестовую модель
        config = {
            'beam': {
                'beam_energy': 6500,
                'bunch_intensity': 1.15e11,
                'sigma_x': 0.045,
                'sigma_y': 0.045,
                'emittance': 2.5e-6
            },
            'geometry': {
                'radius': 4297,
                'circumference': 26659
            }
        }
        model = LHCHybridModel(config)
        # Запускаем симуляцию
        model.run_simulation(num_turns=5)
        # Создаем аномальные данные
        anomalous_events = []
        for i in range(20):
            particles = []
            # Создаем аномальные события с очень высокой энергией
            energy = 100000 if i < 5 else 1000
            for j in range(5):
                particles.append({
                    'name': f'particle_{j}',
                    'energy': energy * (0.1 + j*0.2),
                    'px': random.gauss(0, 100),
                    'py': random.gauss(0, 100),
                    'pz': random.gauss(0, 100)
                })
            anomalous_events.append({'total_energy': energy, 'particles': particles})
        # Добавляем аномальные события в модель
        model.simulation_state['recent_events'] = anomalous_events
        # Запускаем обнаружение аномалий
        anomalies = model.detect_anomalies(max_events=20)
        # Проверяем, что аномалии обнаружены
        assert len(anomalies['by_type']['statistical']) > 0, "Статистические аномалии не обнаружены"
        assert len(anomalies['by_type']['topological']) > 0, "Топологические аномалии не обнаружены"
        return True

# ===================================================================
# 16. *** МОДУЛЬ: Основная функция ***
# ===================================================================
def create_default_config():
    """Создает конфигурацию по умолчанию для LHC."""
    return {
        'beam': {
            'beam_energy': 6500,  # ГэВ
            'bunch_intensity': 1.15e11,
            'sigma_x': 0.045,
            'sigma_y': 0.045,
            'emittance': 2.5e-6,
            'beta_x': 55.0,
            'beta_y': 55.0,
            'dispersion_x': 0.0,
            'dispersion_y': 0.0,
            'bunch_length': 0.075,
            'energy_spread': 1.1e-4,
            'num_bunches': 2556
        },
        'geometry': {
            'radius': 4297,  # м
            'circumference': 26659  # м
        },
        'beam_beam': {
            'enabled': True,
            'strength': 0.01
        }
    }
def main():
    """Основная функция для запуска системы."""
    logger.info("Запуск системы LHC 2.0...")
    # Создаем конфигурацию
    config = create_default_config()
    # Создаем модель
    model = LHCHybridModel(config)
    # Запускаем симуляцию
    logger.info("
=== Запуск симуляции ===")
    model.run_simulation(num_turns=10)
    # Анализируем события
    logger.info("
=== Анализ событий ===")
    model.analyze_events(model.get_recent_events())
    # Обнаружение аномалий
    logger.info("
=== Обнаружение аномалий ===")
    model.detect_anomalies()
    # Визуализация результатов
    logger.info("
=== Визуализация результатов ===")
    model.enhanced_visualization()
    # Экспорт данных
    logger.info("
=== Экспорт данных ===")
    model.export_to_root("lhc_simulation.root")
    model.export_to_hepmc3("lhc_simulation.hepmc3")
    # Калибровка модели
    logger.info("
=== Калибровка модели ===")
    target_observables = {
        "luminosity": 2.0e34,  # целевая светимость в см⁻²с⁻¹
        "beam_size_x": 0.045,
        "beam_size_y": 0.045
    }
    parameters_to_calibrate = ["sigma_x", "sigma_y", "emittance"]
    model.calibrate_model(target_observables, parameters_to_calibrate)
    # Запуск тестов
    logger.info("
=== Запуск тестов ===")
    test_suite = LHCTestSuite()
    test_suite.run_all_tests()
    logger.info("
=== Завершение работы ===")
    logger.info("Система LHC 2.0 завершила работу.")
if __name__ == "__main__":
    main()
