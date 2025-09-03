import random
import numpy as np
import pandas as pd


class MySVM:

    def __init__(self, n_iter: int = 10, learning_rate: float = 0.001, C: float = 1, sgd_sample=None, random_state=42):
        self.n_iter = n_iter  # Кол-во итераций
        self.learning_rate = learning_rate  # Шаг обучения
        self.weights = None  # Веса модели
        self.b = None        # Смещение
        self.C = C           # Параметр регуляризации
        self.sgd_sample = sgd_sample  # Размер подвыборки для SGD
        self.random_state = random_state  # Фиксированный seed для reproducibility

    def fit(self, X: pd.DataFrame, y: pd.Series):
        random.seed(self.random_state)  # Фиксируем random

        y_prepared = self.prepare_y(y)  # Преобразуем метки в -1 и 1
        self.weights = np.ones(X.shape[1])  # Инициализация весов
        self.b = 1.0  # Инициализация смещения

        # Определяем размер подвыборки
        if self.sgd_sample is None:
            sample_size = len(X)
        else:
            if 0 < self.sgd_sample < 1:
                sample_size = round(X.shape[0] * self.sgd_sample)
            else:
                sample_size = self.sgd_sample

        for i in range(self.n_iter):
            # Выбираем случайные строки для SGD
            if self.sgd_sample is not None:
                sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
            else:
                sample_rows_idx = list(range(X.shape[0]))

            fix_X = X.iloc[sample_rows_idx]
            fix_y = y_prepared.iloc[sample_rows_idx]

            # Обновление весов для каждого примера
            for j in range(len(fix_X)):
                x_i = fix_X.iloc[j]
                y_i = fix_y.iloc[j]

                if self.is_right_classification(x_i, y_i):
                    # Корректная классификация, обновляем веса только с L2-регуляризацией
                    self.weights = self.weights - self.learning_rate * 2 * self.weights
                else:
                    # Ошибка классификации, обновляем веса и смещение
                    self.weights = self.weights - self.learning_rate * (2 * self.weights - self.C * y_i * x_i)
                    self.b = self.b - self.learning_rate * (-(self.C * y_i))

    def predict(self, X: pd.DataFrame):
        # Предсказание классов (0/1) на основе линейного решения
        pred = ((X @ self.weights + self.b) >= 0).astype(int)
        return pred

    def prepare_y(self, y: pd.Series):
        # Преобразуем метки в -1 и 1
        return y.copy().apply(lambda v: 1 if v == 1 else -1)

    def is_right_classification(self, X_row, y_val):
        # Проверка правильности классификации (условие SVM)
        return y_val * (self.weights @ X_row + self.b) >= 1

    def get_loss_error(self, X: pd.DataFrame, y: pd.Series):
        # Вычисление SVM hinge loss + регуляризация
        distances = 1 - y * (X @ self.weights + self.b)
        hinge_loss = np.maximum(0, distances).mean()
        margin = np.dot(self.weights, self.weights)
        return margin + self.C * hinge_loss

    def get_coef(self):
        # Возвращаем веса и смещение
        return self.weights, self.b

