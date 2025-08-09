print("Hypothesis 3: There is a relationship between the presence of industry jargon in the subtitle of a crowdfunding campaign and its chances of success.")

import os
try:
    import pandas as pd
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from scipy.stats import chi2_contingency, anderson_ksamp, mannwhitneyu, spearmanr
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except ModuleNotFoundError as e:
    print("Missing library:", e.name)
    print('Please run: pip install -r requirements.txt in the terminal')
    exit(1)

# Loading data
file = os.path.join("output", "Tabletop_Games_cleaned.csv")
df = pd.read_csv(file)
df["state"] = pd.to_numeric(df["state"], errors="coerce")
df["blurb"] = df["blurb"].astype(str)

MIN_JARGON = 0 
MAX_JARGON = 4 # Easily changable numbers that filter outliers when counting jargon terms in singular blurbs

# List of boardgame-related jargon terms
jargon_terms_raw = ["Abstract", "Action Drafting", "Allowance System", "Action Points", "Action Queue", "Action Retrieval", "Action Selection",
"Action Timer", "Adjacency", "Advantage Token", "Admin Time", "Alliances", "Alpha Gaming", "Alpha Player", "Ameritrash",
"American-style", "Analysis Paralysis", "Area Control", "Area Control Game", "Area Enclosure", "Area Influence", "Area Majority",
"Area Movement", "Area-Impulse", "ARG", "Auction", "Auction Compensation", "Auction Dexterity", "Auction Dutch", "Automatic Resource Growth",
"Balance", "Battle Card Driven", "Beer & Pretzels Game", "Betting", "Bets As Wagers", "BGG", "BGG Patron", "Bias", "Bidding",
"Bingo", "Bits", "Blind Bid", "Block Wargame", "Bluff", "BPA", "Brain Burny", "Bribery", "Broken", "Campaign", "Card Drafting",
"Catch the Leader", "Catch-up Mechanism", "CCG", "CDG", "Chaining", "Chit", "Chit-Pull System", "Choke Point", "Chrome",
"Closed Drafting", "Closed Economy Auction", "COIN", "Collectible Game", "Communication Limits", "Competitive Game", "Components",
"Con", "Connections", "Constrained Bidding", "Contracts", "Co-op", "Cooperative Game", "Core Group", "Crayon Rail System",
"Critical Hits and Failures", "Crunchy", "Cube Tower", "Custom Dice", "Deck Building", "Deck Construction", "Decision Scale",
"Decision Space", "Deduction", "Defector", "Delayed Purchase", "Designer Game", "Dexterity Game", "Dice Drafting", "Dice Game",
"Dice Manipulation", "Dice Rolling", "Dice-Fest", "Dice Workers", "Die Icon Resolution", "Die Pips","Different Dice Movement",
"Different Worker Types", "Draft", "Dry", "Dudes on a Map", "Dungeon Crawl", "Dungeon Master", "Economic Game",
"Elapsed Real Time Ending", "Enclosure", "End Game Bonuses", "Engine Builder", "Engine Building", "Eurogame", "Event Deck",
"Events", "Experience Game", "Expansion", "Fast-paced", "Family Game", "Fiddly", "Filler", "Fixed Placement", "Flavor Text",
"Flip the Table", "Fluff", "Follow Action", "Follow Suit", "Friendly Tie", "Game Abbreviations", "Game Master", "Game System",
"Gamer", "Gamers' Game", "Gamey", "Gateway Game", "Going Nuclear", "Grognard", "Grok", "Hand Limit", "Hand Management",
"Hardcore Gamer", "Hate Drafting", "Hex", "Hidden Movement", "Hidden Roles", "Hidden Victory Points", "Hit Points", "Hybrid Games",
"Immersive", "Input Randomness", "Isometric", "Kingmaker", "Kingmaking", "King of the hill", "LCG", "Legacy Game", "Light Game",
"Line of Sight", "Luck Mitigation", "Mathy", "Meaty", "Mechanic", "Mechanisms", "Meeple", "Meta-game", "Min-Maxing", "Miniatures",
"Minis", "Modular Board", "Modular Board Game", "Multi-player Solitaire", "Mystery-solving", "Narrative Choice", "Negotiation",
"Network and Route Building", "NSR", "Off-Suit", "OGL", "Orthogonal", "Output Randomness", "Paper-and-Pencil", "Passed Action Token",
"Pathfinder", "Party Game", "Pattern Building", "Pattern Movement", "Pattern Recognition", "Perfect Information Game",
"Physical Removal", "Pick-up and Deliver", "Pieces as Map", "Player Agency", "Player Elimination", "Player Interaction",
"Playtest", "Point Salad", "Power Creep", "Print and Play", "Programmed Movement", "P&P", "Push Your Luck", "Quarterbacking",
"Rage Quit", "Random Production", "Real-Time", "Recipe Fulfillment", "Replay Value", "Replayable", "Resolution", "Resource Queue",
"Resource to Move", "Role Order", "Roll and Move", "Roll and Write", "Roll20", "Round", "Rules Lawyer", "Scenario",
"Secret Unit Deployment", "Sealed Bid", "Set Collection", "Setup Time", "Shelf of Shame", "Shelfie", "Simulation", "Skirmish",
"Sleeves", "Slog", "Social Deduction", "Solitaire Game", "Speed Matching", "Static Capture", "Stat-Based", "Storytelling",
"Strategy", "Strategic", "Table Hog", "Table Presence", "Table Talk", "Tableau", "Take That", "Tap", "Tech Tree", "Technology Tree",
"The Geek", "Thematic Game", "Theme", "Thinky", "Tile Laying", "Tile Placement", "Tile-laying Game", "Time Track", "TRPG",
"Trick-Taking", "Turn Order", "Turn-Based", "TTRPG", "Turtling", "Unbalanced", "Uptime", "Variable Phase Order",
"Variable Player Powers", "Variable Set-up", "Victory Condition", "Victory Points", "Voting", "Wagering", "Wargame",
"Weight", "Win Condition", "Worker Placement", "Zone of Control", "ZoC"]

# Special groups of jargon terms
dice_terms = {"d4", "d6", "d8", "d10", "d12", "d20"}
dnd_group = {"dnd", "d&d", "5e", "5th edition"}
themed_pattern = re.compile(r"\b\w+-themed\b")

# Function generating different variants for each term (e.g. "Dice rolling" -> "Dice-rolling" & "Dicerolling")
def generate_variants(term):
    variants = {term.lower()}
    if " " in term:
        variants.update({term.lower().replace(" ", "-"), term.lower().replace(" ", "")})
    if "-" in term:
        variants.update({term.lower().replace("-", " "), term.lower().replace("-", "")})
    return variants

print("Creating jargon variants")

jargon_variants = set()
for term in jargon_terms_raw:
    jargon_variants.update(generate_variants(term))

print("Searching for jargon terms in blurbs - please wait.")

# Functions for checking if jargon is present and how many jargon terms are in a blurb.
def contains_jargon(text):
    text = text.lower()
    words = set(re.findall(r'\b\w+\b', text))
    if words & dice_terms or words & dnd_group or themed_pattern.search(text):
        return True
    return any(re.search(r'\b' + re.escape(term) + r'\b', text) for term in jargon_variants)

def count_unique_jargon(text):
    text = text.lower()
    return sum(1 for term in jargon_variants if re.search(r'\b' + re.escape(term) + r'\b', text))

def extract_terms_grouped(text):
    text = text.lower()
    found = set(term for term in jargon_variants if re.search(r'\b' + re.escape(term) + r'\b', text))
    words = set(re.findall(r'\b\w+\b', text))
    if words & dice_terms: found.add("dice_term")
    if words & dnd_group: found.add("dnd_term")
    if themed_pattern.search(text): found.add("themed_term")
    return found

df["has_jargon"] = df["blurb"].apply(contains_jargon)
df["jargon_count"] = df["blurb"].apply(count_unique_jargon)

# Statistical analysis (presence of jargon)
contingency = pd.crosstab(df["has_jargon"], df["state"])
chi2, p, _, _ = chi2_contingency(contingency)
success_rates = df.groupby("has_jargon")["state"].mean()
n = contingency.values.sum()
min_dim = min(contingency.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

print("\n=== Jargon Presence Analysis ===")
print(f"Chi2 = {chi2:.3f}, p = {p:.20f}")
print(f"Success rate (jargon=True): {success_rates.get(True, 0):.4f}")
print(f"Success rate (jargon=False): {success_rates.get(False, 0):.4f}")
print(f"Cramér's V = {cramers_v:.4f}")

median_percent = df.groupby("has_jargon")["percent_funded"].median()
print(f"Median % funded (jargon=True): {median_percent.get(True, 0):.2f}")
print(f"Median % funded (jargon=False): {median_percent.get(False, 0):.2f}")

# Statistical analysis (jargon terms amount in each blurb)
filtered = df[(df["jargon_count"] >= MIN_JARGON) & (df["jargon_count"] <= MAX_JARGON)].copy()
filtered["jargon_grouped"] = filtered["jargon_count"].apply(lambda x: str(x) if x < 3 else "3+")

print(f"\n🔍 Filtered {len(filtered)} campaigns with {MIN_JARGON}–{MAX_JARGON} jargon terms.")

corr, corr_p = spearmanr(filtered["jargon_count"], filtered["state"]) #Starting with Spearman's correlation
print(f"Spearman correlation: r = {corr:.3f}, p = {corr_p:.20f}")

# Continuing with logistic regression
  # Variables
filtered["jargon_sq"] = filtered["jargon_count"] ** 2
X_linear = sm.add_constant(filtered[["jargon_count"]])
X_quad = sm.add_constant(filtered[["jargon_count", "jargon_sq"]])
y = filtered["state"]
 # Linear regression model
model_linear = sm.Logit(y, X_linear).fit(disp=0)
print("\n=== Logistic Regression (Linear) ===")
print(model_linear.summary())
odds_ratio_linear = np.exp(model_linear.params["jargon_count"])
print(f"Odds ratio (jargon_count): {odds_ratio_linear:.4f}")

# Nonlinear regression model
model_quad = sm.Logit(y, X_quad).fit(disp=0)
print("\n=== Logistic Regression (With Quadratic Term) ===")
print(model_quad.summary())
odds_ratio_quad = np.exp(model_quad.params["jargon_count"])
print(f"Odds ratio (jargon_count): {odds_ratio_quad:.4f}")

# Test of significance of the quadratic component
p_quad = model_quad.pvalues["jargon_sq"]
print(f"Quadratic term p-value: {p_quad:.4f}")
if p_quad < 0.05:
    print("✅ Quadratic term is statistically significant - non-linearity likely.")
else:
    print("ℹ️ Quadratic term is not statistically significant - no strong evidence of non-linearity.")

# Bar chart 
grouped = filtered.groupby("jargon_grouped")["state"].agg(["count", "mean"]).reindex(["0", "1", "2", "3+"]).reset_index()

plt.figure(figsize=(10, 6))
plt.bar(grouped["jargon_grouped"], grouped["mean"] * 100, color="steelblue")
for i, row in grouped.iterrows():
    plt.text(i, row["mean"] * 100 + 1, f'{row["count"]}', ha='center')
plt.xlabel("Number of jargon terms in campaign")
plt.ylabel("Campaign success rate (%)")
plt.title(f"Impact of jargon term count on campaign success\n(campaigns with {MIN_JARGON} to {MAX_JARGON} terms, grouped 3+)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
plt.savefig("H3_Bar_Chart.jpg", format="jpg", dpi=300)
plt.close()

# Most commonly used jargon terms in blurbs
success_terms = defaultdict(int)
fail_terms = defaultdict(int)

for _, row in df.iterrows():
    terms = extract_terms_grouped(row["blurb"])
    counter = success_terms if row["state"] == 1 else fail_terms
    for term in terms:
        counter[term] += 1

top_s = sorted(success_terms.items(), key=lambda x: x[1], reverse=True)[:20]
top_u = sorted(fail_terms.items(), key=lambda x: x[1], reverse=True)[:20]

print("\n Top 20 jargon terms in SUCCESSFUL campaigns (campaign count):")
for term, count in top_s:
    print(f"{term}: {count}")

print("\n Top 20 jargon terms in UNSUCCESSFUL campaigns (campaign count):")
for term, count in top_u:
    print(f"{term}: {count}")

# A chart comparing most common jargon terms in succesful and failed campaigns
df_s = pd.DataFrame(top_s, columns=["term", "success_count"])
df_u = pd.DataFrame(top_u, columns=["term", "unsuccess_count"])

merged = pd.merge(df_s, df_u, on="term", how="outer").fillna(0)
merged["total"] = merged["success_count"] + merged["unsuccess_count"]
merged = merged.sort_values("total").reset_index(drop=True)

indices = np.arange(len(merged))
bar_height = 0.4

plt.figure(figsize=(14, 10))
plt.barh(indices + bar_height/2, merged["success_count"], height=bar_height, color="green", label="Successful")
plt.barh(indices - bar_height/2, merged["unsuccess_count"], height=bar_height, color="red", label="Unsuccessful")
plt.yticks(indices, merged["term"])
plt.xlabel("Number of campaigns containing the term")
plt.title("Comparison of jargon term presence in successful and unsuccessful campaigns")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("H3_Horizontal_Bar_Chart.jpg", format="jpg", dpi=300)
plt.close()
