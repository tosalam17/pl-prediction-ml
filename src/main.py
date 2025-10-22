import numpy as np
from prem_predict import Prem_Predictor

def logistic(x):
    """Convert score to probability (sigmoid)"""
    return 1 / (1 + np.exp(-x))

def main():
    pl = Prem_Predictor()

    # --- Fixture ---
    home_team = "Liverpool"
    away_team = "Manchester Utd"

    print(f"\n⚽ Prediction: {home_team} vs {away_team}\n")

    # --- Build feature differences ---
    features = pl.build_match_features(home_team, away_team)

    # --- Weighted model (hand-tuned) ---
    score = (
        0.15 * features.get("xG_Diff", 0)
        + 0.25 * features.get("Shots Per 90_Diff", 0)
        + 0.15 * features.get("Avg Possession_Diff", 0)
        + 0.35 * features.get("Goals Per 90_Diff", 0)
        + 0.10 * features.get("Clean Sheet%_Diff", 0)
        + 0.05 * features.get("Keeper Save%_Diff", 0)
        + 0.10 * features.get("Home Advantage", 0)
    )

    # --- Convert score → probabilities ---
    p_home = logistic(score)
    p_away = 1 - p_home
    p_draw = 1 - abs(p_home - p_away) * 0.6

    # Normalize to sum to 1
    total = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home/total, p_draw/total, p_away/total

    # --- Determine predicted result ---
    if p_home > max(p_draw, p_away):
        result = "🏠 Home Win"
    elif p_away > max(p_draw, p_home):
        result = "🚗 Away Win"
    else:
        result = "🤝 Draw"

    # --- Output ---
    print(f"Predicted Result: {result}")
    print(f"{home_team}: {p_home*100:.1f}%  |  Draw: {p_draw*100:.1f}%  |  {away_team}: {p_away*100:.1f}%")
    print("\n✅ Prediction complete.\n")

if __name__ == "__main__":
    main()
