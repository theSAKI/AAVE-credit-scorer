# Aave V2 DeFi Credit Scoring Model

A Python application that analyzes Aave V2 protocol transactions and assigns a **credit score (0–1000)** to DeFi wallets. The system uses behavioral feature engineering and a hybrid rule-based + ML model to assess user reliability, risk, and potential bot-like behavior.

---

## 💡 Bonus Concept

This scoring framework can be adapted into a real-time risk engine for DeFi lending protocols to dynamically adjust interest rates, collateral requirements, or access based on wallet behavior.

---

## 🎯 Project Overview

The system processes a raw JSON file containing \~100K Aave V2 transactions, extracts wallet behaviors, and computes a credit score per wallet. This score reflects the wallet's history of deposits, borrows, repayments, liquidations, and more — enabling **automated trust signals** in decentralized finance.

---

## 📋 Features

* **Behavioral Feature Engineering**
  Extracts 30+ behavioral features such as days active, deposit/borrow ratios, liquidation count, transaction timing, and bot-like patterns.

* **Hybrid Credit Scoring Algorithm**
  Combines rule-based heuristics and a Random Forest ML model to assign credit scores between 0 and 1000.

* **Automated Markdown Report**
  Generates a rich `analysis_updated.md` report with detailed score distribution, high-risk vs. low-risk behavior summaries, and feature importance rankings.

* **One-Step Execution**
  Just run `credit_scorer.py` with a JSON file to generate scores, visualizations, and analysis.

---

## 🛠️ Technologies Used

| Tech                | Purpose                                            |
| ------------------- | -------------------------------------------------- |
| Python 3.9+         | Core programming language                          |
| Pandas, NumPy       | Data processing and manipulation                   |
| Scikit-learn        | Machine learning (Isolation Forest, Random Forest) |
| Matplotlib, Seaborn | Visualization of score distributions and trends    |

---

## 📦 Installation

Clone the repository and set up the environment:

```bash
git clone <your-repo-url>
cd aave-credit-scorer

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1️⃣ Place your transaction data

Put your `user-transactions.json` inside the `data/` directory:

```
aave-credit-scorer/
└── data/
    └── user-transactions.json
```

### 2️⃣ Run the scoring pipeline

```bash
python credit_scorer.py data/user-transactions.json
```

### 3️⃣ Generate the markdown report

```bash
python update_analysis.py
```

---

## 📁 Project Structure

```
aave-credit-scorer/
├── data/
│   └── user_transactions.json         # Raw transaction data (100K+ records)
├── credit_scorer.py                   # Main processing and scoring script
├── update_analysis.py                 # Report generator from results
├── analysis.md                        # Template markdown for the final report
├── wallet_credit_scores.csv           # Wallets with final credit scores
├── analysis_results.json              # Score distributions and stats
├── analysis_updated.md                # Populated final report
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 📊 Output Files

* **wallet\_credit\_scores.csv**
  Contains wallet addresses and their assigned `final_score`.

* **analysis\_results.json**
  Includes:

  * Mean, median, std deviation, min, max scores
  * Score range distribution
  * Behavioral stats for high- and low-scoring wallets

* **analysis\_updated.md**
  Human-readable markdown report combining all results — great for sharing insights with stakeholders or teams.

---

## ⚠️ Important Notes

* **Input Format**
  The input JSON must include fields like `userWallet`, `action`, `timestamp`, and nested `actionData.amount`, `actionData.assetSymbol`.

* **Performance**
  On standard machines, scoring 100K records takes 1–3 minutes.

* **Extensibility**
  The scoring logic is modular. You can easily:

  * Add new behavioral features
  * Replace the ML model
  * Extend the scoring logic to other protocols like Compound or MakerDAO

---

## 📝 Assignment Requirements Compliance

This project fully satisfies the Aave V2 internship assignment requirements:

✅ Ingests a sample of 100K raw DeFi transactions

✅ Assigns a credit score (0–1000) per wallet

✅ Scores are derived **solely from behavioral data**

✅ Implements a **one-step script** to generate scores

✅ Includes a **markdown-based report**

✅ Fully documented with setup and execution instructions

---

## 👤 Author

**Mohammed Faris Sait**
---


