print("\nHypothesis 2: There is a relationship between the presence of certain special characters and typographic features in a campaign's subtitle and its chance of success.\n")
print("H2a: The presence of exclamation points in a campaign's subtitle increases its chance of success.\n")
print("H2b: The presence of emoji in a campaign's subtitle increases its chance of success.\n")
print("H2c: The presence of capital letters (CAPS LOCK) in a campaign's subtitle increases its chance of success..\n")
print("H2d: Using single periods in a subtitle decreases the chance of success.\n")

import os
try:
    import pandas as pd
    import re
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import chi2_contingency, mannwhitneyu, anderson
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except ModuleNotFoundError as e:
    print("Missing library:", e.name)
    print('Please run: pip install -r requirements.txt in the terminal')
    exit(1)

# Loading data
file = os.path.join("output", "Tabletop_Games_cleaned.csv")
df = pd.read_csv(file)

df.dropna(subset=["blurb", "name", "percent_funded", "state"], inplace=True)
df["blurb"] = df["blurb"].astype(str)
df["name"] = df["name"].astype(str)
df["percent_funded"] = df["percent_funded"].astype(float)
df["state"] = df["state"].astype(int)  # 0 lub 1
df["success"] = df["state"]

# Defining variables
emoji_pattern = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # emotikony
    "\U0001F300-\U0001F5FF"  # symbole i piktogramy
    "\U0001F680-\U0001F6FF"  # transport i symbole
    "\U0001F1E0-\U0001F1FF"  # flagi
    "]+", flags=re.UNICODE)

    #Including the list of exceptions of what would not be considered as "CAPS LOCK" usage
KNOWN_EXCEPTIONS = {
    "CCG", "CDG", "COIN", "DND", "D&D", "GM", "HP", "PNP", "RPG", "MM", "STL", "TCG", "TV", "CNC", "AI", "HBCU",
    "WW", "COVID", "COVID-19", "PVC", "LED", "PDF", "OSR", "USA", "DM", "AWOL", "CC", "PC", "PU",
    "ELECTROM", "OGL", "NSR", "FASA", "SAT", "ACT", "MCAT", "GED", "MTG", "FDM", "IRIS", "SLA", "USPTO", "SCI-FI",
    *["II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"],
    "NPC"
}

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY"
}

COUNTRY_CODES = {
    "US", "UK", "DE", "FR", "IT", "ES", "NL", "PL", "SE", "NO", "DK", "FI",
    "BE", "AT", "CH", "IE", "CZ", "SK", "HU", "PT", "RO", "BG", "HR", "SI",
    "EE", "LT", "LV", "GR", "CY", "MT", "LU", "RU", "UA", "BY", "TR", "IL",
    "AE", "SA", "IR", "IN", "CN", "JP", "KR", "HK", "SG", "MY", "TH", "VN",
    "ID", "PH", "AU", "NZ", "ZA", "EG", "NG", "BR", "AR", "MX", "CL", "CO",
    "PE", "VE", "CA"
}

def is_exception(word):
    w = re.sub(r'[^A-Z0-9\-]', '', word.upper())
    if w in KNOWN_EXCEPTIONS or w in US_STATES or w in COUNTRY_CODES:
        return True
    if len(w) == 1:
        return True
    if w.startswith("D") and w[1:].isdigit():
        return True
    if "RPG" in w:
        return True
    return False

df["has_exclamation"] = df["blurb"].str.contains("!", regex=False).astype(int)
df["has_emoji"] = df["blurb"].apply(lambda x: int(bool(emoji_pattern.search(x))))

def has_capslock_pattern(row):
    blurb_words = row["blurb"].split()
    name_words = set(w for w in row["name"].split() if w.isupper())

    streak = 0
    single_caps_words = 0

    for w in blurb_words:
        if w.isalpha() and w.isupper() and not is_exception(w) and w not in name_words:
            streak += 1
            if streak >= 2:
                return True
            single_caps_words += 1
        else:
            streak = 0

    return single_caps_words == 1

df["has_caps_lock"] = df.apply(has_capslock_pattern, axis=1)

    #Using re to find single dots, that are not connected to numbers
dot_pattern = re.compile(r"(?<!\d)(?<!\.)\.(?!\d)(?!\.)")

df["has_dot"] = df["blurb"].apply(lambda x: int(bool(dot_pattern.search(x))))

def count_non_numeric_dots(text):
    return len(dot_pattern.findall(text))

df["dot_count"] = df["blurb"].apply(count_non_numeric_dots)

    #Defining V Cramer test
def cramers_v(conf_matrix):
    chi2 = chi2_contingency(conf_matrix)[0]
    n = conf_matrix.sum().sum()
    min_dim = min(conf_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

# Proceeding with the test - checking if there are any exclamation marks, emojis, CAPS LOCK'ed words or singular non-numerical dots.
# Statistical tests - Chi square + V Cramer

binary_features = ["has_exclamation", "has_emoji", "ends_with_dot", "has_caps_lock", "has_dot"]

for col in binary_features:
    ct = pd.crosstab(df[col], df["success"])
    chi2, p, _, _ = chi2_contingency(ct)
    success_rate = df.groupby(col)["success"].mean()
    mean_percent = df.groupby(col)["percent_funded"].mean()
    print(f"\n{col}:")
    print(f"Chi2 = {chi2:.3f}, p = {p:.4f}")
    print(f"Success rate when feature present: {success_rate.get(1, 0):.3f}")
    print(f"Success rate when feature not present: {success_rate.get(0, 0):.3f}")
    print(f"Cramér's V = {cramers_v(ct):.3f}")
    print(f"Average % of funding when feature present: {mean_percent.get(1, 0):.2f}")
    print(f"Average % of funding when feature not present: {mean_percent.get(0, 0):.2f}")

# Mann-Whitney test for the number of singular non-numerical dots
success_dots = df.loc[df["success"] == 1, "dot_count"]
fail_dots = df.loc[df["success"] == 0, "dot_count"]
U, p = mannwhitneyu(success_dots, fail_dots)
print(f"\nMann-Whitney test (number of dots): U={U:.2f}, p={p:.4f}")
print(f"Average dot number (success): {success_dots.mean():.3f}")
print(f"Average dot number (failure): {fail_dots.mean():.3f}")

# Logistic regression model for binary features
model_bin = smf.logit("success ~ has_exclamation + has_emoji + has_caps_lock", data=df).fit(disp=False)
print(model_bin.summary())

# % of letters in the entire blurb written with CAPS LOCK (ignoring the exceptions)

def percent_caps_with_exceptions(row):
    blurb_words = row["blurb"].split()
    name_words = set(row["name"].split())
    total_letters = total_caps = 0

    for w in blurb_words:
        if is_exception(w) or w in name_words:
            continue
        letters = [c for c in w if c.isalpha()]
        total_letters += len(letters)
        total_caps += sum(c.isupper() for c in letters)

    return (total_caps / total_letters * 100) if total_letters > 0 else 0

df["blurb_caps_pct"] = df.apply(percent_caps_with_exceptions, axis=1)

print("\nAverage percentage of capital letters by final state:")
print(df.groupby("success")["blurb_caps_pct"].mean())

caps_success = df.loc[df["success"] == 1, "blurb_caps_pct"]
caps_fail = df.loc[df["success"] == 0, "blurb_caps_pct"]

# Anderson-Darling test for CAPS LOCK %
ad_success = anderson(caps_success)
ad_fail = anderson(caps_fail)
print("\nAnderson-Darling test for percentage of letters written in capital letters:")
print(f"Success - stats: {ad_success.statistic:.4f}, crit values: {ad_success.critical_values}")
print(f"Failure - stats: {ad_fail.statistic:.4f}, crit values: {ad_fail.critical_values}")

# Mann-Whitney test
u_stat, p_val = mannwhitneyu(caps_success, caps_fail)
print(f"\nMann-Whitney test for CAPS LOCK %: U = {u_stat:.2f}, p = {p_val:.4f}")

# Boxplot
plt.figure(figsize=(6, 5))
sns.boxplot(x="success", y="blurb_caps_pct", data=df)
plt.xticks([0,1], ["Failure", "Success"])
plt.title("Percentage of capital letters in blurb by final state")
plt.ylabel("Capital letters %")
plt.show()
plt.savefig("H2_Boxplot.jpg", format="jpg", dpi=300)
plt.close()

# Logistic regression with percentage caps (nonlinearity assumption)
df['caps_pct_sq'] = df['blurb_caps_pct'] ** 2
X = sm.add_constant(df[['blurb_caps_pct', 'caps_pct_sq']])
y = df['success']
model_quad = sm.Logit(y, X).fit(disp=False)
print(model_quad.summary())

b1, b2 = model_quad.params['blurb_caps_pct'], model_quad.params['caps_pct_sq']
opt_caps = -b1 / (2 * b2) if b2 != 0 else np.nan
print(f"Optimal % of CAPS letters: {opt_caps:.2f}%")

# Visualization
x_vals = np.linspace(0, 100, 500)
X_plot = sm.add_constant(pd.DataFrame({
    'blurb_caps_pct': x_vals,
    'caps_pct_sq': x_vals**2
}))
y_preds = model_quad.predict(X_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_preds, label="Success probability", color='blue')
plt.axvline(opt_caps, color='red', linestyle='--', label=f"Optimal % of CAPS: ({opt_caps:.2f}%)")
plt.xlabel("Percentage of capital letters in blurb")
plt.ylabel("Probability of success")
plt.title("The Impact of Capital Letters in a Blurb on Campaign Success")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("H2_Chart.jpg", format="jpg", dpi=300)
plt.close()
