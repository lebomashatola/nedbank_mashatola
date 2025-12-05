# Fantasy Sports Team Optimizer

## 📘 Introduction

The **Fantasy Sports Team Optimizer** is an end‑to‑end AI system that:

* Collects real sports data (ESPN / TheSportsDB / local CSV)
* Encodes player historical performance using **Topological Data Analysis (TDA)** via Persistence Landscapes
* Uses an **AI-powered recommendation model** (Hugging Face, Apache 2.0) to predict future player performance
* Runs an optimization algorithm to select the **best possible fantasy lineup** under salary & roster constraints
* Displays results in an interactive **Streamlit dashboard** with player profiles, fatigue, and positions

The project is engineered as a modular, configurable system with:

* Config files for dataset selection and model choice
* Shell scripts for fully automated execution
* A unified `main.sh` pipeline to run the full workflow

---

## 📂 Repository Structure

```
fantasy_team_optimizer/
│
├── configs/
│   └── config.yaml
│
├── data/
│   └── (placeholder for downloaded or local data)
│
├── scripts/
│   ├── obtain_data.sh
│   ├── run_tda.sh
│   ├── train_model.sh
│   ├── run_optimizer.sh
│   └── main.sh
│
├── src/
│   ├── data_loader.py
│   ├── tda_encoder.py
│   ├── model_training.py
│   ├── optimizer.py
│   └── visualize.py
│
├── streamlit/
│   └── team.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the Project

### **1️⃣ Install Dependencies**

```
pip install -r requirements.txt
```

or using conda:

```
conda create -n fantasy python=3.10 -y
conda activate fantasy
pip install -r requirements.txt
```

---

## **2️⃣ Run Each Component Manually**

### **Download Data**

```
sh scripts/obtain_data.sh
```

This downloads player data from ESPN / TheSportsDB based on the config.

### **Run TDA Encoding**

```
sh scripts/run_tda.sh
```

This runs the Gudhi-based persistence landscape encoder and outputs features.

### **Train the AI Recommendation Model**

```
sh scripts/train_model.sh
```

This trains the Hugging Face regression model (Apache 2.0) + XGBoost + RandomForest ensemble.

### **Run the Lineup Optimizer**

```
sh scripts/run_optimizer.sh
```

This produces the optimal fantasy team given budget + constraints.

---

## **3️⃣ Run the Entire Workflow Automatically**

You can execute everything from start to finish using:

```
sh scripts/main.sh
```

This performs:

1. Data acquisition
2. TDA encoding
3. ML training
4. Lineup optimization
5. Saves the final result into `output/team.json`

---

## **4️⃣ Launch the Streamlit Dashboard**

To visualize the selected lineup:

```
streamlit run streamlit/team.py
```

This displays:

* Player cards
* Stats, fatigue, nationality
* Position layout (pitch/court visualization)

---

## 🤖 Models Used

### **Hugging Face Model (Apache 2.0 Licensed)**

We integrate:

* `microsoft/deberta-v3-small` (Apache 2.0)

Used as a tabular‑text hybrid encoder for enhanced AI-based player performance prediction.

### **Classical ML Models**

* XGBoost
* RandomForestRegressor (fallback model)

---

## 🔧 Configuration

All configuration happens in `configs/config.yaml`:

* Select dataset source: `espn`, `thesportsdb`, or `local_csv`
* Select model: `huggingface`, `xgboost`, `randomforest`
* Select league (if available)

Example:

```yaml
model: "huggingface"
dataset: "espn"
league: "English Premier League"
```

---

## 📦 Shell Scripts Overview

### **obtain_data.sh**

Downloads new player stats.

### **run_tda.sh**

Runs TDA → Persistence Landscape encoding.

### **train_model.sh**

Trains the selected ML model.

### **run_optimizer.sh**

Runs integer programming lineup selection.

### **main.sh**

Full pipeline automation.


