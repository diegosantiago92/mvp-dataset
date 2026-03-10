  // ── Configuração da API ──────────────────────────────────────────────────
  const API_URL = 'http://localhost:5000';

  // ── Definição dos campos ─────────────────────────────────────────────────
  const FIELDS = [
    { name:'age',      label:'Idade',                   type:'number', placeholder:'ex: 52', min:1,   max:120, tip:'Idade do paciente em anos' },
    { name:'sex',      label:'Sexo',                    type:'select', options:[['0','Feminino'],['1','Masculino']] },
    { name:'cp',       label:'Dor no Peito',             type:'select', options:[['0','Assintomático'],['1','Angina Típica'],['2','Angina Atípica'],['3','Não Anginosa']] },
    { name:'trestbps', label:'Pressão Arterial (mmHg)', type:'number', placeholder:'ex: 120', min:50,  max:250, tip:'Pressão arterial em repouso' },
    { name:'chol',     label:'Colesterol (mg/dl)',      type:'number', placeholder:'ex: 200', min:50,  max:600, tip:'Colesterol sérico total' },
    { name:'fbs',      label:'Glicemia em Jejum >120',  type:'select', options:[['0','Não'],['1','Sim']], tip:'Glicemia em jejum > 120 mg/dl' },
    { name:'restecg',  label:'ECG em Repouso',          type:'select', options:[['0','Normal'],['1','Anomalia ST-T'],['2','Hipertrofia VE']] },
    { name:'thalach',  label:'FC Máxima (bpm)',         type:'number', placeholder:'ex: 150', min:50,  max:250, tip:'Frequência cardíaca máxima atingida' },
    { name:'exang',    label:'Angina no Exercício',     type:'select', options:[['0','Não'],['1','Sim']] },
    { name:'oldpeak',  label:'Depressão ST',            type:'number', placeholder:'ex: 1.0', min:0,   max:10,  step:'0.1', tip:'Depressão ST induzida por exercício' },
    { name:'slope',    label:'Inclinação ST',           type:'select', options:[['0','Descendente'],['1','Plana'],['2','Ascendente']] },
    { name:'ca',       label:'Vasos Coloridos (0–3)',   type:'select', options:[['0','0'],['1','1'],['2','2'],['3','3']] },
    { name:'thal',     label:'Talassemia',              type:'select', options:[['0','Normal'],['1','Defeito Fixo'],['2','Defeito Reversível'],['3','Outro']] },
  ];

  // ── Gera os campos dinamicamente ─────────────────────────────────────────
  function buildForm() {
    const grid = document.getElementById('formGrid');
    grid.innerHTML = '';
    FIELDS.forEach(f => {
      const div = document.createElement('div');
      div.className = 'field';

      const lbl = document.createElement('label');
      lbl.htmlFor = f.name;
      lbl.textContent = f.label;
      if (f.tip) {
        const tip = document.createElement('span');
        tip.className = 'tooltip';
        tip.textContent = '?';
        tip.setAttribute('data-tip', f.tip);
        lbl.appendChild(tip);
      }

      let input;
      if (f.type === 'select') {
        input = document.createElement('select');
        input.id = f.name;
        input.name = f.name;
        f.options.forEach(([val, txt]) => {
          const opt = document.createElement('option');
          opt.value = val;
          opt.textContent = txt;
          input.appendChild(opt);
        });
      } else {
        input = document.createElement('input');
        input.type = 'number';
        input.id = f.name;
        input.name = f.name;
        input.placeholder = f.placeholder || '';
        input.min = f.min;
        input.max = f.max;
        if (f.step) input.step = f.step;
      }

      div.appendChild(lbl);
      div.appendChild(input);
      grid.appendChild(div);
    });
  }

  // ── Coleta os dados do formulário ─────────────────────────────────────────
  function getFormData() {
    const data = {};
    for (const f of FIELDS) {
      const el = document.getElementById(f.name);
      const val = el.value.trim();
      if (val === '' || val === null) return null;
      data[f.name] = parseFloat(val);
    }
    return data;
  }

  // ── Mostra/oculta elementos ───────────────────────────────────────────────
  function setLoading(loading) {
    const btn = document.getElementById('btnPredict');
    const spinner = document.getElementById('spinner');
    const btnText = document.getElementById('btnText');
    btn.disabled = loading;
    spinner.classList.toggle('active', loading);
    btnText.textContent = loading ? 'Analisando...' : 'Analisar Risco Cardíaco';
  }

  function showError(msg) {
    const box = document.getElementById('errorBox');
    box.textContent = '⚠️ ' + msg;
    box.style.display = 'block';
    document.getElementById('result').style.display = 'none';
  }

  function hideError() {
    document.getElementById('errorBox').style.display = 'none';
  }

  // ── Renderiza o resultado ─────────────────────────────────────────────────
  function showResult(data) {
    const resultEl = document.getElementById('result');
    const isDisease = data.prediction === 1;

    resultEl.className = isDisease ? 'result-disease' : 'result-safe';
    document.getElementById('resultIcon').textContent = isDisease ? '⚠️' : '✅';
    document.getElementById('resultLabel').textContent = data.prediction_label;
    document.getElementById('resultSub').textContent = isDisease
      ? 'Risco elevado identificado — consulte um cardiologista.'
      : 'Indicadores dentro do padrão esperado.';

    const pNo  = (data.probability_no_disease * 100).toFixed(1);
    const pYes = (data.probability_disease    * 100).toFixed(1);

    document.getElementById('probNo').textContent  = pNo  + '%';
    document.getElementById('probYes').textContent = pYes + '%';

    // Animação das barras (pequeno delay)
    setTimeout(() => {
      document.getElementById('barNo').style.width  = pNo  + '%';
      document.getElementById('barYes').style.width = pYes + '%';
    }, 50);

    resultEl.style.display = 'block';
  }

  // ── Predição ──────────────────────────────────────────────────────────────
  async function predict() {
    hideError();

    const data = getFormData();
    if (!data) {
      showError('Por favor, preencha todos os campos antes de analisar.');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Erro desconhecido no servidor.');
      }

      showResult(result);
    } catch (err) {
      if (err.name === 'TypeError') {
        showError('Não foi possível conectar à API. Certifique-se de que o backend está rodando em localhost:5000.');
      } else {
        showError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }

  // ── Inicialização ─────────────────────────────────────────────────────────
  buildForm();