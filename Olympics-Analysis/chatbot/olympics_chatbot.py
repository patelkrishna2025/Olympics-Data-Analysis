"""
=============================================================
 Olympics Intelligence System
 MODULE: Olympics Q&A Chatbot
 Rule-based NLP chatbot for Olympics dataset exploration
=============================================================
"""
import pandas as pd
import numpy as np
import re


class OlympicsChatbot:
    """
    Rule-based chatbot that answers questions about the Olympics dataset.
    Supports queries about medals, countries, athletes, sports, gender trends, years.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._preprocess()

    def _preprocess(self):
        df = self.df
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        # Medal numeric weight for scoring
        df["medal_weight"] = df["Medal"].map({"Gold": 3, "Silver": 2, "Bronze": 1}).fillna(0)
        self.df = df

    def answer(self, question: str) -> str:
        q = question.lower().strip()

        # ── Greeting ────────────────────────────────────────────────────────
        if any(w in q for w in ["hello", "hi", "hey", "namaste", "hola"]):
            return (
                "🏅 Hello! I'm your **Olympics Data Assistant**.\n\n"
                "Try asking:\n"
                "- *How many medals did USA win?*\n"
                "- *Top 5 athletes?*\n"
                "- *Most popular sports?*\n"
                "- *Gold medals by year?*\n"
                "- *Women participation trend?*\n"
                "- *Which country dominated swimming?*\n"
                "- *Best year for India?*"
            )

        # ── Total medals ─────────────────────────────────────────────────────
        if re.search(r"how many (total |overall )?(medals|records|entries)", q):
            total  = len(self.df)
            gold   = (self.df["Medal"] == "Gold").sum()
            silver = (self.df["Medal"] == "Silver").sum()
            bronze = (self.df["Medal"] == "Bronze").sum()
            return (
                f"🏅 **Total Medal Records: {total:,}**\n"
                f"- 🥇 Gold   : {gold:,}\n"
                f"- 🥈 Silver : {silver:,}\n"
                f"- 🥉 Bronze : {bronze:,}"
            )

        # ── Country-specific medal query ──────────────────────────────────────
        country_match = re.search(
            r"(medal|win|gold|silver|bronze|performance).{0,30}(usa|united states|india|china|russia|germany|australia|soviet|uk|france|japan|kenya|cuba|brazil|canada|italy|south korea|great britain)",
            q
        ) or re.search(
            r"(usa|united states|india|china|russia|germany|australia|soviet|uk|france|japan|kenya|cuba|brazil|canada|italy|south korea|great britain).{0,30}(medal|win|gold|silver|bronze|performance)",
            q
        )
        if country_match:
            kw_map = {
                "usa": "United States", "united states": "United States",
                "india": "India", "china": "China", "russia": "Russia",
                "germany": "Germany", "australia": "Australia",
                "soviet": "Soviet Union", "uk": "Great Britain",
                "great britain": "Great Britain", "france": "France",
                "japan": "Japan", "kenya": "Kenya", "cuba": "Cuba",
                "brazil": "Brazil", "canada": "Canada", "italy": "Italy",
                "south korea": "South Korea",
            }
            country_key = None
            for kw, full in kw_map.items():
                if kw in q:
                    country_key = full
                    break
            if country_key:
                sub = self.df[self.df["Country"].str.contains(country_key, case=False, na=False)]
                if len(sub) == 0:
                    return f"❌ No records found for **{country_key}**."
                gold   = (sub["Medal"] == "Gold").sum()
                silver = (sub["Medal"] == "Silver").sum()
                bronze = (sub["Medal"] == "Bronze").sum()
                top_sport = sub["Sport"].value_counts().idxmax()
                top_athlete = sub["Athlete"].value_counts().idxmax()
                return (
                    f"🌍 **{country_key} — Olympic Summary:**\n"
                    f"- 🥇 Gold   : {gold}\n"
                    f"- 🥈 Silver : {silver}\n"
                    f"- 🥉 Bronze : {bronze}\n"
                    f"- 🏆 Total  : {gold+silver+bronze}\n"
                    f"- 🏃 Best Sport  : {top_sport}\n"
                    f"- ⭐ Top Athlete : {top_athlete}"
                )

        # ── Top countries ─────────────────────────────────────────────────────
        if re.search(r"(top|best|leading|dominant).*(countr|nation)", q) or \
           re.search(r"(countr|nation).*(top|best|leading)", q):
            n   = self._extract_n(q, 5)
            top = self.df["Country"].value_counts().head(n)
            rows = "\n".join([f"  {i+1}. **{c}** — {v:,} medals"
                              for i, (c, v) in enumerate(top.items())])
            return f"🌍 **Top {n} Countries:**\n{rows}"

        # ── Top athletes ──────────────────────────────────────────────────────
        if re.search(r"(top|best|greatest|most medals).*(athlete|player|person|champion)", q) or \
           re.search(r"(athlete|player).*(top|best|most)", q):
            n   = self._extract_n(q, 5)
            top = self.df["Athlete"].value_counts().head(n)
            rows = "\n".join([f"  {i+1}. **{a}** — {v} medals"
                              for i, (a, v) in enumerate(top.items())])
            return f"🏃 **Top {n} Athletes:**\n{rows}"

        # ── Search athlete ────────────────────────────────────────────────────
        athlete_search = re.search(r"(search|find|tell me about|who is|athlete[:\s]+)(.+)", q)
        if athlete_search:
            keyword = athlete_search.group(2).strip()
            sub = self.df[self.df["Athlete"].str.lower().str.contains(keyword, na=False)]
            if len(sub) == 0:
                return f"❌ No athlete found matching **'{keyword}'**."
            row0 = sub.iloc[0]
            gold   = (sub["Medal"] == "Gold").sum()
            silver = (sub["Medal"] == "Silver").sum()
            bronze = (sub["Medal"] == "Bronze").sum()
            sports = sub["Sport"].unique().tolist()
            return (
                f"🏃 **{row0['Athlete']}**\n"
                f"- Country : {row0.get('Country', 'N/A')}\n"
                f"- Sport(s): {', '.join(sports)}\n"
                f"- 🥇 Gold   : {gold}\n"
                f"- 🥈 Silver : {silver}\n"
                f"- 🥉 Bronze : {bronze}\n"
                f"- Years active: {sorted(sub['Year'].dropna().unique().tolist())}"
            )

        # ── Top sports ────────────────────────────────────────────────────────
        if re.search(r"(top|popular|most medals|best|dominant).*(sport|event|discipline)", q) or \
           re.search(r"(sport|event).*(top|popular|most)", q):
            n   = self._extract_n(q, 5)
            top = self.df["Sport"].value_counts().head(n)
            rows = "\n".join([f"  {i+1}. **{s}** — {v:,} medals"
                              for i, (s, v) in enumerate(top.items())])
            return f"🏊 **Top {n} Sports:**\n{rows}"

        # ── Country dominates sport ───────────────────────────────────────────
        if re.search(r"(dominat|best|top).*(swim|gymnastics|athletics|aquatics|rowing|boxing|wrestling|cycling)", q) or \
           re.search(r"(swim|gymnastics|athletics|aquatics|rowing|boxing|wrestling|cycling).*(dominat|best|who|country)", q):
            sports_kw = {
                "swim": "Aquatics", "aquatics": "Aquatics",
                "gymnastics": "Gymnastics", "athletics": "Athletics",
                "rowing": "Rowing", "boxing": "Boxing",
                "wrestling": "Wrestling", "cycling": "Cycling",
            }
            sport = None
            for kw, full in sports_kw.items():
                if kw in q:
                    sport = full
                    break
            if sport:
                sub = self.df[self.df["Sport"] == sport]
                top_c = sub["Country"].value_counts().head(3)
                rows  = "\n".join([f"  {i+1}. **{c}** — {v}" for i, (c, v) in enumerate(top_c.items())])
                return f"🏅 **Top countries in {sport}:**\n{rows}"

        # ── Gold medal count by year ──────────────────────────────────────────
        if re.search(r"(gold|medal).*(year|trend|over time)", q) or \
           re.search(r"(year|trend).*(gold|medal)", q):
            gold_yr = self.df[self.df["Medal"] == "Gold"].groupby("Year").size()
            rows = "\n".join([f"  {int(yr)}: {cnt} gold medals" for yr, cnt in gold_yr.items()])
            return f"🥇 **Gold Medals Per Year:**\n{rows}"

        # ── Gender ────────────────────────────────────────────────────────────
        if re.search(r"(gender|women|men|female|male|participation)", q):
            gd = self.df["Gender"].value_counts()
            total = len(self.df)
            women = gd.get("Women", 0)
            men   = gd.get("Men", 0)
            # by year
            yr_gender = self.df.groupby(["Year","Gender"]).size().unstack(fill_value=0)
            lines = []
            for yr in sorted(yr_gender.index):
                w = yr_gender.loc[yr].get("Women", 0)
                m = yr_gender.loc[yr].get("Men", 0)
                lines.append(f"  {int(yr)}: Men={m}, Women={w}")
            return (
                f"👫 **Gender Participation:**\n"
                f"- Men   : {men:,} ({men/total*100:.1f}%)\n"
                f"- Women : {women:,} ({women/total*100:.1f}%)\n\n"
                f"📅 **By Year:**\n" + "\n".join(lines)
            )

        # ── Specific year query ───────────────────────────────────────────────
        year_match = re.search(r"\b(197[0-9]|198[0-9]|199[0-9]|200[0-9])\b", q)
        if year_match:
            yr  = int(year_match.group())
            sub = self.df[self.df["Year"] == yr]
            if len(sub) == 0:
                return f"❌ No data for year **{yr}**."
            city   = sub["City"].iloc[0] if "City" in sub.columns else "N/A"
            top_c  = sub["Country"].value_counts().idxmax()
            top_s  = sub["Sport"].value_counts().idxmax()
            gold   = (sub["Medal"] == "Gold").sum()
            total  = len(sub)
            return (
                f"🏟️ **{yr} Olympics — {city}:**\n"
                f"- Total Medals : {total:,}\n"
                f"- Gold Medals  : {gold:,}\n"
                f"- Top Country  : {top_c}\n"
                f"- Top Sport    : {top_s}"
            )

        # ── Most gold medals ──────────────────────────────────────────────────
        if re.search(r"most gold", q):
            top = self.df[self.df["Medal"] == "Gold"]["Country"].value_counts().head(5)
            rows = "\n".join([f"  {i+1}. **{c}** — {v} gold" for i, (c, v) in enumerate(top.items())])
            return f"🥇 **Most Gold Medals (All-Time):**\n{rows}"

        # ── Help / fallback ───────────────────────────────────────────────────
        return (
            "🤔 I didn't quite understand that. Try:\n"
            "- *How many medals did China win?*\n"
            "- *Top 10 athletes?*\n"
            "- *Who dominated swimming?*\n"
            "- *Gold medals by year?*\n"
            "- *Gender participation trend?*\n"
            "- *Best year for USA?*\n"
            "- *Top 5 countries?*\n"
            "- *1996 Olympics summary?*"
        )

    def _extract_n(self, text: str, default: int = 5) -> int:
        m = re.search(r"\b(\d+)\b", text)
        return int(m.group(1)) if m else default
