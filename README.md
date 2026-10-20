## Structure

```text
portfolio-risk-engine/
├── app.py                   # Application interactive Streamlit
├── risk_engine.py           # Fonctions de calcul partagées par l'app et le notebook
├── portfolio_risk_engine.ipynb
├── portfolio_prices_selected.csv # data
├── data_source.json         # Provenance et période du CSV courant
├── generate_demo_data.py    # Régénération du CSV de démonstration
├── portfolio_risk_report_final.pdf  # Rapport courant, marqué DEMO si CSV synthétique
├── requirements.txt
└── README.md
```

## Installation et démarrage

Python 3.10 ou plus récent est recommandé.

### macOS / Linux

```bash
cd portfolio-risk-engine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows (PowerShell)

```powershell
cd portfolio-risk-engine
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
