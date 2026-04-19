"""
=============================================================
 Olympics Intelligence System
 MODULE: ML Models
 - MedalPredictor  : Predicts medal type (Gold/Silver/Bronze)
 - CountryScorer   : Ranks countries by weighted performance
 - SportDomination : Identifies country-sport strengths
=============================================================
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  MEDAL TYPE PREDICTOR
# ─────────────────────────────────────────────
class MedalPredictor:
    """
    Predicts medal type (Gold / Silver / Bronze) from country, sport, gender, year.
    Uses Random Forest + GBM ensemble.
    """

    def __init__(self):
        self.rf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        self.gbm = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                               max_depth=4, random_state=42)
        self.le_country = LabelEncoder()
        self.le_sport   = LabelEncoder()
        self.le_gender  = LabelEncoder()
        self.le_medal   = LabelEncoder()
        self.feature_cols = []
        self.trained = False
        self.metrics = {}

    def fit(self, df: pd.DataFrame) -> "MedalPredictor":
        data = df.dropna(subset=["Country", "Sport", "Gender", "Medal", "Year"]).copy()
        data["country_enc"] = self.le_country.fit_transform(data["Country"].astype(str))
        data["sport_enc"]   = self.le_sport.fit_transform(data["Sport"].astype(str))
        data["gender_enc"]  = self.le_gender.fit_transform(data["Gender"].astype(str))
        data["medal_enc"]   = self.le_medal.fit_transform(data["Medal"].astype(str))

        # Country medal history features
        country_gold  = data[data["Medal"] == "Gold"]["Country"].value_counts()
        country_total = data["Country"].value_counts()
        data["country_gold_rate"] = (
            data["Country"].map(country_gold).fillna(0) /
            data["Country"].map(country_total).fillna(1)
        )

        self.feature_cols = ["country_enc", "sport_enc", "gender_enc",
                              "Year", "country_gold_rate"]

        X = data[self.feature_cols].fillna(0).values
        y = data["medal_enc"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.rf.fit(X_train, y_train)
        self.gbm.fit(X_train, y_train)

        rf_acc  = accuracy_score(y_test, self.rf.predict(X_test))
        gbm_acc = accuracy_score(y_test, self.gbm.predict(X_test))

        self.metrics = {
            "RF Accuracy":  round(rf_acc * 100, 1),
            "GBM Accuracy": round(gbm_acc * 100, 1),
            "Train Size":   len(X_train),
            "Test Size":    len(X_test),
            "Classes":      list(self.le_medal.classes_),
        }
        self.trained = True
        self._country_gold_rate = (
            df[df["Medal"] == "Gold"]["Country"].value_counts() /
            df["Country"].value_counts()
        ).fillna(0)
        print(f"[MedalPredictor] RF={rf_acc:.3f}  GBM={gbm_acc:.3f}")
        return self

    def predict(self, country: str, sport: str, gender: str, year: int) -> dict:
        if not self.trained:
            return {"medal": "Gold", "confidence": 0.33}
        try:
            ce = self.le_country.transform([country])[0]
        except ValueError:
            ce = 0
        try:
            se = self.le_sport.transform([sport])[0]
        except ValueError:
            se = 0
        try:
            ge = self.le_gender.transform([gender])[0]
        except ValueError:
            ge = 0
        gr = float(self._country_gold_rate.get(country, 0.0))
        X  = np.array([[ce, se, ge, year, gr]])
        rf_proba  = self.rf.predict_proba(X)[0]
        gbm_proba = self.gbm.predict_proba(X)[0]
        ensemble  = (rf_proba + gbm_proba) / 2
        best_idx  = np.argmax(ensemble)
        medal     = self.le_medal.inverse_transform([best_idx])[0]
        return {
            "medal":       medal,
            "confidence":  round(float(ensemble[best_idx]) * 100, 1),
            "probabilities": {
                cls: round(float(p) * 100, 1)
                for cls, p in zip(self.le_medal.classes_, ensemble)
            }
        }

    def feature_importance(self) -> pd.DataFrame:
        if not self.trained:
            return pd.DataFrame()
        return pd.DataFrame({
            "Feature":    self.feature_cols,
            "Importance": self.rf.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
#  COUNTRY PERFORMANCE SCORER
# ─────────────────────────────────────────────
class CountryScorer:
    """
    Computes weighted medal scores and rankings per country.
    Gold=3, Silver=2, Bronze=1.
    """

    def __init__(self):
        self.scores: pd.DataFrame = pd.DataFrame()

    def fit(self, df: pd.DataFrame) -> "CountryScorer":
        weight = {"Gold": 3, "Silver": 2, "Bronze": 1}
        df = df.copy()
        df["medal_weight"] = df["Medal"].map(weight).fillna(0)

        agg = df.groupby("Country").agg(
            Gold   = ("Medal", lambda x: (x == "Gold").sum()),
            Silver = ("Medal", lambda x: (x == "Silver").sum()),
            Bronze = ("Medal", lambda x: (x == "Bronze").sum()),
            Total  = ("Medal", "count"),
            Score  = ("medal_weight", "sum"),
        ).reset_index()
        agg["Gold_Rate"] = (agg["Gold"] / agg["Total"]).round(3)
        agg["Rank"] = agg["Score"].rank(ascending=False, method="min").astype(int)
        self.scores = agg.sort_values("Score", ascending=False).reset_index(drop=True)
        return self

    def leaderboard(self, top_n: int = 20) -> pd.DataFrame:
        return self.scores.head(top_n)

    def country_detail(self, country: str) -> dict:
        row = self.scores[self.scores["Country"].str.lower() == country.lower()]
        if len(row) == 0:
            return {}
        r = row.iloc[0]
        return {
            "Country": r["Country"],
            "Rank": int(r["Rank"]),
            "Score": int(r["Score"]),
            "Gold": int(r["Gold"]),
            "Silver": int(r["Silver"]),
            "Bronze": int(r["Bronze"]),
            "Total": int(r["Total"]),
            "Gold Rate": f"{r['Gold_Rate']*100:.1f}%",
        }


# ─────────────────────────────────────────────
#  SPORT DOMINATION ANALYSER
# ─────────────────────────────────────────────
class SportDominationAnalyser:
    """
    Identifies which countries dominate each sport by gold medal share.
    """

    def analyse(self, df: pd.DataFrame) -> pd.DataFrame:
        gold_df = df[df["Medal"] == "Gold"]
        sport_country = gold_df.groupby(["Sport", "Country"]).size().reset_index(name="Golds")
        total_per_sport = gold_df.groupby("Sport").size().reset_index(name="Total_Golds")
        merged = sport_country.merge(total_per_sport, on="Sport")
        merged["Share_%"] = (merged["Golds"] / merged["Total_Golds"] * 100).round(1)
        # Keep top-1 per sport
        top_per_sport = (
            merged.sort_values("Golds", ascending=False)
            .groupby("Sport").first()
            .reset_index()
            .rename(columns={"Country": "Dominant_Country"})
            [["Sport", "Dominant_Country", "Golds", "Total_Golds", "Share_%"]]
            .sort_values("Total_Golds", ascending=False)
        )
        return top_per_sport
