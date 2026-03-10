"""
Heart Disease Prediction API
Backend Flask para predição de doenças cardíacas utilizando modelo de ML embarcado.
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# ── Carregamento do modelo ────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'heart_disease_model.pkl')

model = None

def load_model():
    """Carrega o modelo de ML a partir do arquivo .pkl."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[OK] Modelo carregado com sucesso: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[ERRO] Arquivo do modelo não encontrado: {MODEL_PATH}")
        raise


# ── Utilitários de validação ──────────────────────────────────────────────────
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]

FEATURE_RANGES = {
    'age':      (1,   120),
    'sex':      (0,   1),
    'cp':       (0,   3),
    'trestbps': (50,  250),
    'chol':     (50,  600),
    'fbs':      (0,   1),
    'restecg':  (0,   2),
    'thalach':  (50,  250),
    'exang':    (0,   1),
    'oldpeak':  (0.0, 10.0),
    'slope':    (0,   2),
    'ca':       (0,   4),
    'thal':     (0,   3),
}


def validate_input(data: dict) -> tuple[bool, str]:
    """
    Valida os dados de entrada.
    Retorna (True, '') se válido, ou (False, mensagem_de_erro) se inválido.
    """
    for feature in FEATURE_NAMES:
        if feature not in data:
            return False, f"Campo obrigatório ausente: '{feature}'"
        try:
            value = float(data[feature])
        except (ValueError, TypeError):
            return False, f"Valor inválido para '{feature}': deve ser numérico."
        lo, hi = FEATURE_RANGES[feature]
        if not (lo <= value <= hi):
            return False, (
                f"Valor fora do intervalo para '{feature}': "
                f"esperado entre {lo} e {hi}, recebido {value}."
            )
    return True, ''


# ── Rotas da API ──────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health_check():
    """Verifica se a API está funcionando e se o modelo está carregado."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Recebe dados clínicos de um paciente e retorna a predição do modelo.

    Body (JSON):
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal

    Returns (JSON):
        prediction: 0 (sem doença) ou 1 (com doença)
        prediction_label: string descritiva
        probability_no_disease: probabilidade de não ter doença
        probability_disease: probabilidade de ter doença
    """
    if model is None:
        return jsonify({'error': 'Modelo não carregado. Tente novamente mais tarde.'}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Body da requisição inválido ou vazio.'}), 400

    is_valid, error_msg = validate_input(data)
    if not is_valid:
        return jsonify({'error': error_msg}), 400

    # Montar array de features na ordem correta
    features = np.array([[float(data[f]) for f in FEATURE_NAMES]])

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    return jsonify({
        'prediction': prediction,
        'prediction_label': 'Doença Cardíaca Detectada' if prediction == 1 else 'Sem Doença Cardíaca',
        'probability_no_disease': round(float(probabilities[0]), 4),
        'probability_disease': round(float(probabilities[1]), 4),
    })


@app.route('/features', methods=['GET'])
def get_features():
    """Retorna a lista de features esperadas pelo modelo, com seus intervalos válidos."""
    return jsonify({
        'features': FEATURE_NAMES,
        'ranges': FEATURE_RANGES
    })


# ── Inicialização ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
