import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, mannwhitneyu, anderson
import statsmodels.formula.api as smf
import statsmodels.api as sm

# --- Wczytanie i przygotowanie danych ---
plik = r"D:\WSB\Data\DO BADANIA\Dane_do_badania.csv"
df = pd.read_csv(plik)

# Usuwamy wiersze z brakami w ważnych kolumnach i konwertujemy typy
df.dropna(subset=["blurb", "name", "percent_funded", "state"], inplace=True)
df["blurb"] = df["blurb"].astype(str)
df["name"] = df["name"].astype(str)
df["percent_funded"] = df["percent_funded"].astype(float)
df["state"] = df["state"].astype(int)  # 0 lub 1
df["success"] = df["state"]

# --- Definicje pomocnicze ---

# Wzorzec emoji (sklejony, bez powtarzania regex.compile)
emoji_pattern = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # emotikony
    "\U0001F300-\U0001F5FF"  # symbole i piktogramy
    "\U0001F680-\U0001F6FF"  # transport i symbole
    "\U0001F1E0-\U0001F1FF"  # flagi
    "]+", flags=re.UNICODE)

# Zbiory wyjątków do CAPS LOCK jako set dla szybkiego lookupu
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
    w = re.sub(r'[^A-Z0-9\-]', '', word.upper())  # Czyszczenie do samych wielkich liter, cyfr i "-"
    if w in KNOWN_EXCEPTIONS or w in US_STATES or w in COUNTRY_CODES:
        return True
    if len(w) == 1:
        return True
    if w.startswith("D") and w[1:].isdigit():
        return True
    if "RPG" in w:
        return True
    return False

# --- Tworzenie cech ---

df["has_exclamation"] = df["blurb"].str.contains("!", regex=False).astype(int)
df["has_emoji"] = df["blurb"].apply(lambda x: int(bool(emoji_pattern.search(x))))
df["ends_with_dot"] = df["blurb"].str.contains(r"(?<!\.)\.(?!\.)\s*$").astype(int)

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

# Kropki niebędące częścią liczby
dot_pattern = re.compile(r"(?<!\d)(?<!\.)\.(?!\d)(?!\.)")

df["has_dot"] = df["blurb"].apply(lambda x: int(bool(dot_pattern.search(x))))

def count_non_numeric_dots(text):
    return len(dot_pattern.findall(text))

df["dot_count"] = df["blurb"].apply(count_non_numeric_dots)

# --- Funkcje statystyczne ---

def cramers_v(conf_matrix):
    chi2 = chi2_contingency(conf_matrix)[0]
    n = conf_matrix.sum().sum()
    min_dim = min(conf_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

# --- Wyświetlanie statystyk i testy ---

binary_features = ["has_exclamation", "has_emoji", "ends_with_dot", "has_caps_lock", "has_dot"]

for col in binary_features:
    ct = pd.crosstab(df[col], df["success"])
    chi2, p, _, _ = chi2_contingency(ct)
    success_rate = df.groupby(col)["success"].mean()
    mean_percent = df.groupby(col)["percent_funded"].mean()
    print(f"\n{col}:")
    print(f"Chi2 = {chi2:.3f}, p = {p:.4f}")
    print(f"Odsetek sukcesów gdy cecha=1: {success_rate.get(1, 0):.3f}")
    print(f"Odsetek sukcesów gdy cecha=0: {success_rate.get(0, 0):.3f}")
    print(f"Cramér's V = {cramers_v(ct):.3f}")
    print(f"Średni % finansowania gdy cecha=1: {mean_percent.get(1, 0):.2f}")
    print(f"Średni % finansowania gdy cecha=0: {mean_percent.get(0, 0):.2f}")

# --- Test Mann-Whitneya dla liczby kropek ---
success_dots = df.loc[df["success"] == 1, "dot_count"]
fail_dots = df.loc[df["success"] == 0, "dot_count"]
U, p = mannwhitneyu(success_dots, fail_dots)
print(f"\nTest Mann-Whitneya liczby kropek: U={U:.2f}, p={p:.4f}")
print(f"Średnia liczba kropek (sukces): {success_dots.mean():.3f}")
print(f"Średnia liczba kropek (porażka): {fail_dots.mean():.3f}")

# --- Model logistyczny dla cech binarnych ---
model_bin = smf.logit("success ~ has_exclamation + has_emoji + ends_with_dot + has_caps_lock", data=df).fit(disp=False)
print(model_bin.summary())

# --- Procent wielkich liter w blurbie z wyjątkami i nazwami ---

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

print("\nŚredni procent wielkich liter wg sukcesu:")
print(df.groupby("success")["blurb_caps_pct"].mean())

caps_success = df.loc[df["success"] == 1, "blurb_caps_pct"]
caps_fail = df.loc[df["success"] == 0, "blurb_caps_pct"]

# Test Andersona-Darlinga
ad_success = anderson(caps_success)
ad_fail = anderson(caps_fail)
print("\nTest Andersona-Darlinga dla procentu caps:")
print(f"Sukces - statystyka: {ad_success.statistic:.4f}, krytyczne wartości: {ad_success.critical_values}")
print(f"Porażka - statystyka: {ad_fail.statistic:.4f}, krytyczne wartości: {ad_fail.critical_values}")

# Test Mann-Whitneya
u_stat, p_val = mannwhitneyu(caps_success, caps_fail)
print(f"\nTest Mann-Whitneya dla procentu caps: U = {u_stat:.2f}, p = {p_val:.4f}")

# --- Wizualizacja boxplot ---
plt.figure(figsize=(6, 5))
sns.boxplot(x="success", y="blurb_caps_pct", data=df)
plt.xticks([0,1], ["Porażka", "Sukces"])
plt.title("Procent wielkich liter w blurbie wg sukcesu")
plt.ylabel("Procent liter wielkich")
plt.show()

# --- Regresja logistyczna z efektem kwadratowym dla procentu caps ---
df['caps_pct_sq'] = df['blurb_caps_pct'] ** 2
X = sm.add_constant(df[['blurb_caps_pct', 'caps_pct_sq']])
y = df['success']
model_quad = sm.Logit(y, X).fit(disp=False)
print(model_quad.summary())

b1, b2 = model_quad.params['blurb_caps_pct'], model_quad.params['caps_pct_sq']
opt_caps = -b1 / (2 * b2) if b2 != 0 else np.nan
print(f"Optymalny procent wielkich liter: {opt_caps:.2f}%")

# Wizualizacja paraboli
x_vals = np.linspace(0, 100, 500)
X_plot = sm.add_constant(pd.DataFrame({
    'blurb_caps_pct': x_vals,
    'caps_pct_sq': x_vals**2
}))
y_preds = model_quad.predict(X_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_preds, label="Prawdopodobieństwo sukcesu", color='blue')
plt.axvline(opt_caps, color='red', linestyle='--', label=f"Optymalny procent ({opt_caps:.2f}%)")
plt.xlabel("Procent wielkich liter w blurbie")
plt.ylabel("Prawdopodobieństwo sukcesu")
plt.title("Wpływ wielkich liter w blurbie na sukces kampanii")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Analiza rekordów z najwyższym % caps ---
top_caps = df.nlargest(10, "blurb_caps_pct")[["blurb_caps_pct", "blurb"]]
for _, row in top_caps.iterrows():
    print(f"Procent caps: {row['blurb_caps_pct']:.2f} | Blurb: {row['blurb']}\n")

# --- Próbkowanie rekordów z procentem caps między 30 a 35 ---
filtered = df[(df["blurb_caps_pct"].between(30, 35))]
sampled = filtered.sample(n=min(10, len(filtered)), random_state=42)
for _, row in sampled.iterrows():
    print(f"Procent caps: {row['blurb_caps_pct']:.2f} | Blurb: {row['blurb']}\n")
