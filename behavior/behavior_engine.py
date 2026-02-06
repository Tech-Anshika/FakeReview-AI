# behavior/behavior_engine.py

def calculate_behavior_score(row):
    """
    Calculate behavior-based risk score for a single review
    Returns: score (0–100) and reasons (list)
    """

    score = 0
    reasons = []

    # 1️⃣ Verified purchase
    if "verified_purchase" in row and row["verified_purchase"] is False:
        score += 25
        reasons.append("Unverified purchase")

    # 2️⃣ Helpful votes
    if "helpful_vote" in row and row["helpful_vote"] == 0:
        score += 10
        reasons.append("No helpful votes")

    # 3️⃣ Rating vs text length mismatch
    if "rating" in row and "text" in row:
        if row["rating"] == 5 and len(str(row["text"])) < 30:
            score += 15
            reasons.append("Very short 5-star review")

    # 4️⃣ User review burst (precomputed flag or heuristic)
    if "user_review_burst" in row and row["user_review_burst"] is True:
        score += 30
        reasons.append("Suspicious review burst behavior")

    # Cap the score
    score = min(score, 100)

    return score, reasons
