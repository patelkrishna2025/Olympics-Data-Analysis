# 🏅 Olympics Intelligence System — v2

## 📁 Project Structure

```
Olympics-Analysis/
├── app.py                                    ← ✨ NEW — Streamlit Dashboard (8 tabs)
├── Olympics_Data_Analysis.ipynb              ← UPDATED — Added Chatbot, CV & ML sections
├── Summer-Olympic-medals-1976-to-2008.csv    ← Dataset (15,433 rows × 11 cols)
├── README.md                                 ← This file
│
├── chatbot/                  ← ✨ NEW — Olympics Q&A Chatbot
│   └── olympics_chatbot.py
│
├── cv_module/                ← ✨ NEW — Computer Vision module
│   └── sports_cv.py
│
├── models/                   ← ✨ NEW — ML Models
│   └── olympics_models.py    (MedalPredictor · CountryScorer · SportDomination)
│
├── exports/                  ← ✨ NEW — Saved charts & reports
├── assets/                   ← ✨ NEW — Static assets
└── utils/                    ← ✨ NEW — Shared utility helpers
```

## 🚀 How to Run

```bash
pip install streamlit pandas numpy plotly scikit-learn opencv-python pillow matplotlib seaborn
streamlit run app.py
```

## 📊 Dashboard Tabs (v2)

| # | Tab | Description |
|---|-----|-------------|
| 1 | 🏅 Overview | KPIs, medals by year, type donut, heatmap, host cities |
| 2 | 🌍 Countries | Leaderboard, choropleth map, stacked medal breakdown |
| 3 | 🏃 Athletes | Top athletes bar, athlete×sport heatmap, athlete search |
| 4 | 🏊 Sports | Sport trends, country×sport heatmap, domination table |
| 5 | 👫 Gender Trends | Participation over years, women's share trend |
| 6 | 🔮 ML Predictor | RF + GBM medal type predictor with probability bars |
| 7 | 👁️ CV Analysis | **NEW** — Sports image analyser: mood, blur, crowd, ring colour |
| 8 | 💬 Chatbot | **NEW** — Olympics Q&A assistant with quick buttons |

## 📓 Notebook Updates

| Section | Content |
|---------|---------|
| Original (1–34) | Existing EDA cells (fixed Colab upload → portable path) |
| 35–36 | 💬 Chatbot init + demo conversation |
| 37–39 | 👁️ CV Analysis: synthetic test, full analysis, filter gallery |
| 40–45 | 🔮 ML: MedalPredictor training, predictions, CountryScorer, SportDomination, charts |

## 🔧 v2 Changes
- ✅ Streamlit Dashboard `app.py` — 8 fully interactive tabs
- ✅ `chatbot/olympics_chatbot.py` — Rule-based NLP, 12 query types
- ✅ `cv_module/sports_cv.py` — Motion blur, crowd density, Olympic ring colour, 10 filters
- ✅ `models/olympics_models.py` — RF+GBM Medal Predictor, CountryScorer, SportDomination
- ✅ New folders: `chatbot/`, `cv_module/`, `models/`, `exports/`, `assets/`, `utils/`
- ✅ Notebook: 12 new cells, fixed Colab dependency
