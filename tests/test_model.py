"""
Testes automatizados do modelo de predição de doenças cardíacas.

Este módulo valida se o modelo atende aos requisitos mínimos de desempenho
definidos como thresholds aceitáveis. Caso o modelo seja substituído por uma
nova versão, estes testes garantem que o novo modelo não degrade o desempenho.

Requisitos de desempenho estabelecidos:
    - Acurácia  >= 0.80 (80%)
    - Recall    >= 0.80 (80%) — prioridade em cenário médico (minimizar falsos negativos)
    - Precisão  >= 0.75 (75%)
    - F1-Score  >= 0.78 (78%)
    - AUC-ROC   >= 0.85 (85%)
"""

import os
import pytest
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)

# ── Configurações ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'heart_disease_model.pkl')
DATASET_URL = 'https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv'

# Thresholds de desempenho mínimo exigidos
THRESHOLD_ACCURACY  = 0.80
THRESHOLD_RECALL    = 0.80
THRESHOLD_PRECISION = 0.75
THRESHOLD_F1        = 0.78
THRESHOLD_AUC_ROC   = 0.85


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope='module')
def model():
    """Carrega o modelo treinado a partir do arquivo .pkl."""
    assert os.path.exists(MODEL_PATH), (
        f"Arquivo do modelo não encontrado: {MODEL_PATH}. "
        "Execute o notebook para gerar o arquivo heart_disease_model.pkl."
    )
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope='module')
def test_data():
    """
    Carrega o dataset original e retorna o conjunto de teste (holdout 20%),
    replicando exatamente a mesma divisão utilizada no treinamento
    (random_state=42, stratify=y).
    """
    df = pd.read_csv(DATASET_URL)
    X = df.drop('target', axis=1)
    y = df['target']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


@pytest.fixture(scope='module')
def predictions(model, test_data):
    """Gera predições e probabilidades para o conjunto de teste."""
    X_test, y_test = test_data
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_test, y_pred, y_proba


# ── Testes de Carregamento ─────────────────────────────────────────────────────
class TestModelLoading:
    """Testes de integridade do carregamento do modelo."""

    def test_model_file_exists(self):
        """Verifica se o arquivo do modelo existe."""
        assert os.path.exists(MODEL_PATH), f"Modelo não encontrado em: {MODEL_PATH}"

    def test_model_loads_without_error(self, model):
        """Verifica se o modelo é carregado sem erros."""
        assert model is not None

    def test_model_has_predict_method(self, model):
        """Verifica se o modelo possui o método predict."""
        assert hasattr(model, 'predict'), "Modelo não possui método 'predict'."

    def test_model_has_predict_proba_method(self, model):
        """Verifica se o modelo possui o método predict_proba."""
        assert hasattr(model, 'predict_proba'), "Modelo não possui método 'predict_proba'."


# ── Testes de Interface ────────────────────────────────────────────────────────
class TestModelInterface:
    """Testes de interface de entrada/saída do modelo."""

    def test_prediction_output_binary(self, predictions):
        """Verifica se as predições são binárias (0 ou 1)."""
        _, y_pred, _ = predictions
        unique_preds = set(np.unique(y_pred))
        assert unique_preds.issubset({0, 1}), (
            f"Predições devem ser 0 ou 1. Encontrado: {unique_preds}"
        )

    def test_probability_range(self, predictions):
        """Verifica se as probabilidades estão no intervalo [0, 1]."""
        _, _, y_proba = predictions
        assert np.all(y_proba >= 0.0) and np.all(y_proba <= 1.0), (
            "Todas as probabilidades devem estar entre 0 e 1."
        )

    def test_prediction_shape(self, model, test_data):
        """Verifica se o shape da predição corresponde ao número de amostras."""
        X_test, _ = test_data
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(X_test), (
            f"Shape incorreto: esperado {len(X_test)}, obtido {len(y_pred)}."
        )

    def test_single_sample_prediction(self, model, test_data):
        """Verifica se o modelo consegue fazer predição com uma única amostra."""
        X_test, _ = test_data
        single_sample = X_test.iloc[[0]]
        prediction = model.predict(single_sample)
        assert len(prediction) == 1


# ── Testes de Desempenho ───────────────────────────────────────────────────────
class TestModelPerformance:
    """
    Testes de desempenho do modelo contra os thresholds estabelecidos.

    Justificativa dos thresholds:
      - Acurácia >= 80%: mínimo aceitável para classificação médica auxiliar
      - Recall >= 80%:   prioridade máxima — minimizar falsos negativos (pacientes
                         doentes diagnosticados como saudáveis)
      - Precisão >= 75%: evitar excesso de alarmes falsos
      - F1 >= 78%:       equilíbrio entre precisão e recall
      - AUC-ROC >= 85%:  boa capacidade discriminatória geral do modelo
    """

    def test_accuracy_above_threshold(self, predictions):
        """Acurácia deve ser >= 80%."""
        y_test, y_pred, _ = predictions
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy >= THRESHOLD_ACCURACY, (
            f"Acurácia insuficiente: {accuracy:.4f} < {THRESHOLD_ACCURACY} (threshold)."
        )

    def test_recall_above_threshold(self, predictions):
        """Recall deve ser >= 80% (prioridade em contexto médico)."""
        y_test, y_pred, _ = predictions
        recall = recall_score(y_test, y_pred)
        assert recall >= THRESHOLD_RECALL, (
            f"Recall insuficiente: {recall:.4f} < {THRESHOLD_RECALL} (threshold). "
            "Um recall baixo implica muitos falsos negativos — risco clínico."
        )

    def test_precision_above_threshold(self, predictions):
        """Precisão deve ser >= 75%."""
        y_test, y_pred, _ = predictions
        precision = precision_score(y_test, y_pred)
        assert precision >= THRESHOLD_PRECISION, (
            f"Precisão insuficiente: {precision:.4f} < {THRESHOLD_PRECISION} (threshold)."
        )

    def test_f1_above_threshold(self, predictions):
        """F1-Score deve ser >= 78%."""
        y_test, y_pred, _ = predictions
        f1 = f1_score(y_test, y_pred)
        assert f1 >= THRESHOLD_F1, (
            f"F1-Score insuficiente: {f1:.4f} < {THRESHOLD_F1} (threshold)."
        )

    def test_roc_auc_above_threshold(self, predictions):
        """AUC-ROC deve ser >= 85%."""
        y_test, _, y_proba = predictions
        roc_auc = roc_auc_score(y_test, y_proba)
        assert roc_auc >= THRESHOLD_AUC_ROC, (
            f"AUC-ROC insuficiente: {roc_auc:.4f} < {THRESHOLD_AUC_ROC} (threshold)."
        )

    def test_print_metrics_report(self, predictions):
        """Imprime um relatório completo das métricas (sempre passa)."""
        y_test, y_pred, y_proba = predictions
        print("\n" + "=" * 50)
        print("  RELATÓRIO DE DESEMPENHO DO MODELO")
        print("=" * 50)
        print(f"  Acurácia  : {accuracy_score(y_test, y_pred):.4f}  (threshold >= {THRESHOLD_ACCURACY})")
        print(f"  Recall    : {recall_score(y_test, y_pred):.4f}  (threshold >= {THRESHOLD_RECALL})")
        print(f"  Precisão  : {precision_score(y_test, y_pred):.4f}  (threshold >= {THRESHOLD_PRECISION})")
        print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}  (threshold >= {THRESHOLD_F1})")
        print(f"  AUC-ROC   : {roc_auc_score(y_test, y_proba):.4f}  (threshold >= {THRESHOLD_AUC_ROC})")
        print("=" * 50)
        assert True  # Este teste sempre passa — serve para visualizar o relatório
