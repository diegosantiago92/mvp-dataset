# 🫀 CardioScan — Predição de Doenças Cardíacas com Machine Learning

MVP da disciplina de Qualidade de Software, Segurança e Sistemas Inteligentes. 

---

## 📋 Descrição do Projeto

Sistema completo de machine learning para predição de doenças cardíacas com base em dados clínicos. O projeto inclui:

- **Notebook Colab** com todo o pipeline de ML (EDA → pré-processamento → modelagem → avaliação → exportação)
- **Backend Flask** que serve o modelo de ML embarcado via API REST
- **Frontend HTML/CSS/JS** com formulário de predição
- **Testes automatizados** com PyTest para validação do modelo

**Dataset**: [Heart Disease UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) — 303 pacientes, 13 atributos clínicos, classificação binária (doença cardíaca: sim/não)

---

## 🗂️ Estrutura do Repositório

```
mvp/
├── backend/
│   ├── app.py               # API Flask
│   └── requirements.txt     # Dependências Python
├── frontend/
│   ├── css/
│   │   └── styles.css       # Estilos da interface
│   ├── js/
│   │   └── script.js        # Lógica e chamadas à API
│   └── index.html           # Interface web
├── model/
│   └── heart_disease_model.pkl  # Modelo treinado (gerado pelo notebook)
├── tests/
│   ├── conftest.py
│   └── test_model.py        # Testes PyTest
└── README.md
notebook_heart_disease.ipynb  # Notebook Google Colab
```

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.10+
- pip

### 1. Gerar o Modelo (Notebook Colab)

1. Acesse o notebook `notebook_heart_disease.ipynb` no [Google Colab](https://colab.research.google.com/)
2. Execute todas as células do início ao fim (**Executar tudo**)
3. Ao final, o arquivo `heart_disease_model.pkl` será gerado
4. Faça o download do `heart_disease_model.pkl` e coloque-o na pasta `mvp/model/`

### 2. Instalar Dependências

```bash
cd mvp/backend
pip install -r requirements.txt
```

### 3. Executar o Backend

```bash
cd mvp/backend
python app.py
```

A API estará disponível em: `http://localhost:5000`

### 4. Executar o Frontend

Abra o arquivo `mvp/frontend/index.html` diretamente no navegador.

> **Dica**: Se preferir, use uma extensão como _Live Server_ (VS Code) para servir o arquivo.

### 5. Executar os Testes

```bash
cd mvp
pytest tests/test_model.py -v
```

---

## 🔌 Endpoints da API

| Método | Rota       | Descrição                            |
|--------|------------|--------------------------------------|
| GET    | `/health`  | Verifica se a API está funcionando   |
| GET    | `/features`| Lista as features e intervalos válidos |
| POST   | `/predict` | Realiza a predição para um paciente  |

### Exemplo de Requisição POST `/predict`

```json
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```

### Resposta

```json
{
  "prediction": 1,
  "prediction_label": "Doença Cardíaca Detectada",
  "probability_no_disease": 0.1234,
  "probability_disease": 0.8766
}
```

---

## 🧪 Testes Automatizados

Os testes verificam os seguintes requisitos de desempenho mínimo:

| Métrica   | Threshold |
|-----------|-----------|
| Acurácia  | ≥ 80%     |
| Recall    | ≥ 80%     |
| Precisão  | ≥ 75%     |
| F1-Score  | ≥ 78%     |
| AUC-ROC   | ≥ 85%     |

> O **Recall** recebe prioridade máxima no contexto médico, pois falsos negativos (paciente doente diagnosticado como saudável) representam risco crítico.

---

## 🔐 Considerações de Segurança (Desenvolvimento de Software Seguro)

Dados médicos são extremamente sensíveis. As seguintes boas práticas deveriam ser aplicadas em um ambiente de produção:

1. **Anonimização**: remoção de todos os dados identificadores (nome, CPF, data de nascimento exata)
2. **Pseudonimização**: substituição de identificadores por tokens aleatórios, sem possibilidade de re-identificação sem a chave
3. **Criptografia em trânsito**: uso obrigatório de HTTPS/TLS para todas as comunicações
4. **Criptografia em repouso**: dados armazenados devem ser criptografados (AES-256)
5. **Controle de acesso**: autenticação e autorização (JWT, OAuth2) para acesso à API
6. **Auditoria**: logs de todas as requisições e predições, com identificação do usuário
7. **Validação de inputs**: a API já realiza validação de ranges para evitar dados inválidos ou ataques de injeção
8. **Minimização de dados**: coletar apenas os dados estritamente necessários para a predição

---

## 🛠️ Tecnologias Utilizadas

- **Machine Learning**: Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Joblib
- **Backend**: Flask, Flask-CORS
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Testes**: PyTest

---

## ⚠️ Aviso Importante

Este sistema é uma ferramenta **acadêmica** e **não substitui** avaliação médica profissional. Sempre consulte um cardiologista para diagnóstico e tratamento.
