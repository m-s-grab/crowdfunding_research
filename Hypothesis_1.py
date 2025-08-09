print("\nHypothesis 1: There is a relationship between the characteristics of a crowdfunding campaign's title "
      "and subtitle and the likelihood of achieving its financial goal.\n")
print("H1a: Title length negatively correlates with the optimal subtitle length.\n")

import os
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.stats import anderson, anderson_ksamp, pearsonr, spearmanr, mannwhitneyu, pointbiserialr
    from nltk.corpus import stopwords
    import nltk
    import matplotlib.ticker as mtick
except ModuleNotFoundError as e:
    print("Missing library:", e.name)
    print('Please run: pip install -r requirements.txt in the terminal')
    exit(1)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Loading data
file = os.path.join("output", "Tabletop_Games_cleaned.csv")
if not os.path.isfile(file):
    print('File not found! Please use the "0. Cleaning and preparing.py" first.')
    exit(1)
df = pd.read_csv(file).dropna(subset=["name", "blurb", "percent_funded", "state"])
df["name"] = df["name"].astype(str)
df["blurb"] = df["blurb"].astype(str)
df["percent_funded"] = df["percent_funded"].astype(float)
df["success"] = df["state"].astype(int)

# Defining variables
df["title_len_words"] = df["name"].str.split().str.len()
df["blurb_len_words"] = df["blurb"].str.split().str.len()
df["title_len_words_sq"] = df["title_len_words"] ** 2
df["blurb_len_words_sq"] = df["blurb_len_words"] ** 2
df["interaction_words"] = df["title_len_words"] * df["blurb_len_words"]

print(f"Longest_title(words): {df['title_len_words'].max()} words")
print(f"Longest_blurb(words): {df['blurb_len_words'].max()} words")

# Defining normality test (Anderson-Darling)
def normality_tests(df):
    for success_val in [0, 1]:
        label = "successful" if success_val == 1 else "failed"
        print(f"\n✅ Anderson-Darling test for final state = {label}:")
        subset = df[df["success"] == success_val]
        for col in ["title_len_words", "blurb_len_words"]:
            result = anderson(subset[col])
            crit_val = result.critical_values[2]
            stat = result.statistic
            msg = "normal" if stat < crit_val else "not normal"
            print(f"{col}: stat={stat:.4f}, 5% crit={crit_val:.4f} ({msg})")

# Medians
print("\nMeans of title/blurb length by final state\n")
print(df.groupby("success")[["title_len_words", "blurb_len_words"]].mean())

# Defining correlations
def show_correlations(group_name, data):
    print(f"\nCorrelations ({group_name}):")
    print("Pearson:\n", data.corr(method='pearson'))
    print("Spearman:\n", data.corr(method='spearman'))

show_correlations("final state = 1", df[df['success'] == 1][['title_len_words', 'blurb_len_words']])
show_correlations("final state = 0", df[df['success'] == 0][['title_len_words', 'blurb_len_words']])

# Logistic regression
print('\nLogistic regression: word count')
model_words = smf.logit("success ~ title_len_words + blurb_len_words + title_len_words_sq + blurb_len_words_sq + interaction_words", data=df).fit()
print(model_words.summary())

# Heatmap
title_range = np.arange(1, df['title_len_words'].max()+1)
blurb_range = np.arange(1, df['blurb_len_words'].max()+1)
grid = pd.DataFrame([(t, b) for t in title_range for b in blurb_range], columns=["title_len_words", "blurb_len_words"])
grid["title_len_words_sq"] = grid["title_len_words"] ** 2
grid["blurb_len_words_sq"] = grid["blurb_len_words"] ** 2
grid["interaction_words"] = grid["title_len_words"] * grid["blurb_len_words"]
grid["predicted_prob"] = model_words.predict(grid)

plt.figure(figsize=(12, 8))
sns.heatmap(grid.pivot(index="blurb_len_words", columns="title_len_words", values="predicted_prob"),
            cmap="coolwarm", cbar_kws={"label": "Likehood of success"})
plt.xlabel("Title length (words)")
plt.ylabel("Blurb length (words)")
plt.title("Heatmap: Impact of Title and Blur Length")
plt.gca().invert_yaxis()
plt.savefig("H1_Heatmap.jpg", format="jpg", dpi=300)
plt.close()

# Wyświetlenie jako obraz
display(Image("wykres_codespace.jpg"))

# Optimal blurb length
print("\nOptimal blurb length for title length")
beta = model_words.params
def opt_blurb(title_len): return (-beta['blurb_len_words'] - beta['interaction_words'] * title_len) / (2 * beta['blurb_len_words_sq'])
t_range = np.arange(5, 21)
optimal_blurbs = opt_blurb(t_range)
for t, b in zip(t_range, optimal_blurbs):
    print(f"Title: {t} words -> Optimal blurb: {b:.2f} words")

plt.plot(t_range, optimal_blurbs, marker='o')
plt.title("Optimal blurb length relative to title")
plt.xlabel("Title length (words)")
plt.ylabel("Optimal blurb length (words)")
plt.grid(True)
plt.show()
plt.savefig("H1_Optimal_blurb.jpg", format="jpg", dpi=300)
plt.close()

print("\nH1b: Repeating the title content in a campaign's subtitle negatively impacts its likelihood of achieving its goal.\n")

# Defining repeated percent of the title in blurbs - skipping stopwords!
def title_in_blurb(title, blurb):
    title_words = set(word.lower() for word in title.split() if word.lower() not in stop_words)
    blurb_words = set(word.lower() for word in blurb.split() if word.lower() not in stop_words)
    return len(title_words & blurb_words) / len(title_words) if title_words else 0.0

df["title_repeated_percent"] = df.apply(lambda row: title_in_blurb(row["name"], row["blurb"]), axis=1)
df["title_repeated_percent_sq"] = df["title_repeated_percent"] ** 2

# Analysis
group_success = df[df["success"] == 1]["title_repeated_percent"]
group_failure = df[df["success"] == 0]["title_repeated_percent"]

# Anderson
stat_ad, crit_vals, sig = anderson_ksamp([group_success, group_failure])
print(f"\nAnderson-Darling test: stat={stat_ad:.3f}, sig={sig:.3f}")

# Mann-Whitney
u_stat, p_val = mannwhitneyu(group_success, group_failure, alternative='two-sided')
print(f"U = {u_stat:.3f}, p = {p_val:.4f}")

# Correlation
r_pb, p_pb = pointbiserialr(df["success"], df["title_repeated_percent"])
print(f"Correlation: r = {r_pb:.3f}, p = {p_pb:.4f}")

# Violin plot
plt.figure(figsize=(8, 5))
sns.violinplot(x="success", y="title_repeated_percent", hue="success", data=df,
               palette={0: "red", 1: "green"}, inner="quartile", legend=False)
plt.xticks([0, 1], ["Failed", "Successful"])
plt.title("Title repetition in blurb vs. success")
plt.xlabel("Campaign status")
plt.ylabel("Title repetition percentage")
plt.ylim(0, 1)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.tight_layout()
plt.show()
plt.savefig("H1_Violin_plot.jpg", format="jpg", dpi=300)
plt.close()

# Nonlinear Logistic Regression Model: % of Title repeated
model_repeat = smf.logit("success ~ title_repeated_percent + title_repeated_percent_sq", data=df).fit()
print(model_repeat.summary())

# Nonlinear regression graph
x_vals = np.linspace(0, 1, 100)
pred_df = pd.DataFrame({"title_repeated_percent": x_vals, "title_repeated_percent_sq": x_vals**2})
pred_df["predicted_prob"] = model_repeat.predict(pred_df)

plt.plot(x_vals, pred_df["predicted_prob"])
plt.xlabel("Title repetition percentage")
plt.ylabel("Success probability")
plt.title("Nonlinear regression: title repetition")
plt.grid(True)
plt.show()
plt.savefig("H1_Regression_graph.jpg", format="jpg", dpi=300)
plt.close()

# Linear regression: % funded vs % title repeated
X = sm.add_constant(df['title_repeated_percent'])
model_ols = sm.OLS(df['percent_funded'], X).fit()
print(model_ols.summary())

# Optimal % title repeated
b1 = model_repeat.params["title_repeated_percent"]
b2 = model_repeat.params["title_repeated_percent_sq"]
if b2 != 0:
    optimal_percent = -b1 / (2 * b2)
    print(f"Optimal % title repeated: {optimal_percent:.2%}")
else:
    print("No quadratic term – optimum cannot be determined.")
