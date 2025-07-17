#### Autores: Adrián Herrera, Patrick F. Bárcena y Carlos Moreno

🧠 Deep Q-Learning aplicado a Trading Algorítmico
Este proyecto implementa un agente de Deep Q-Learning (DQL) para la toma de decisiones en un entorno de trading. El objetivo fue entrenar un modelo que aprende a comprar, vender o mantener posiciones en acciones de Microsoft (MSFT) mediante un proceso de exploración y explotación, y luego comparar su desempeño contra una estrategia tradicional Buy & Hold.

🎯 Objetivo
Desarrollar un agente basado en Reinforcement Learning capaz de maximizar el rendimiento de una inversión a través de decisiones secuenciales, evaluando métricas financieras clave como rendimiento total, volatilidad, drawdown máximo y Sharpe Ratio.

🧪 Resumen del Experimento
Agente Q-Learning básico: Entrenado con 100 episodios para entender la mecánica del entorno.

Agente Deep Q-Learning (DQL): Versión robusta entrenada con 500 episodios para capturar patrones complejos del mercado.

Comparación: Backtesting del agente en datos reales y contraste contra Buy & Hold.

Resultados Destacados:

| Métrica                    | Buy & Hold | DQL          |
| -------------------------- | ---------- | ------------ |
| Rendimiento Total (%)      | +368%      | **+46,977%** |
| Max Drawdown (%)           | -37.15%    | **-5.67%**   |
| Volatilidad Anualizada (%) | 30.12%     | **11.86%**   |
| Sharpe Ratio               | 1.00       | **8.74**     |

⚙️ Tecnologías Usadas
Python 3.x

PyTorch (entrenamiento del agente DQL)

Pandas, NumPy, Matplotlib (análisis y visualización)

Jupyter Notebook (desarrollo y presentación del proyecto)

🗂️ Estructura del Proyecto:

005_RL-DQL_Trading/

├── data/      # Datos históricos de MSFT

│   └── MSFT_5yr.csv

├── models/           # Pesos del modelo DQL entrenado

│   └── dql_model.pth

├── notebooks/        # Notebook principal con el análisis completo

│   └── report.ipynb

├── utils/            # Código del agente DQL

│   └── rl_agent.py

├── README.md         # Descripción del proyecto

└── requirements.txt  # Dependencias necesarias

🚀 Cómo Ejecutar

1️⃣ Clona el repositorio:
git clone <URL_del_repo>
cd 005_RL-DQL_Trading

2️⃣ Instala las dependencias:
pip install -r requirements.txt

3️⃣ Abre el notebook:
jupyter notebook notebooks/report.ipynb



