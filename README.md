**Repository Structure**
- /data – Contains:
  - A few example `.csv` files that were already preprocessed (e.g., final campaign states updated).
  - Several additional raw, unprocessed `.csv` files (randomly selected) to demonstrate that the script correctly filters them out when merging data
- Scripts – Python scripts for cleaning, preparing, and analyzing the dataset.

Original data comes from the WebRobots Kickstarter Dataset (https://webrobots.io/kickstarter-datasets/).

**How to Run**
1. Install requirements:
<pre>pip install -r requirements.txt</pre>

2. Then start by running "0.Cleaning_and_preparing.py"
This script will:
- Merge CSV files from the /data folder into a single file.
- Filter only campaigns from the "Tabletop Games" category.
- Perform data cleaning:
  - Remove duplicates (keeping the most up-to-date records).
  - Remove campaigns with non-English blurbs.
- Apply other necessary preprocessing steps.

**Research Hypotheses**
The hypotheses were formulated based on a thorough review of the literature on crowdfunding campaign success factors:

H1: There is a relationship between the characteristics of a crowdfunding campaign's title and subtitle and the likelihood of achieving the funding goal.
H1a: Title length negatively correlates with the optimal subtitle length.
H1b: Repeating the title content in the subtitle negatively impacts the campaign’s chances of success.

H2: There is a relationship between the presence of certain special characters and typographic features in the campaign subtitle and the likelihood of success.
H2a: The presence of exclamation marks in the subtitle increases the likelihood of success.
H2b: The presence of emojis in the subtitle increases the likelihood of success.
H2c: The use of capital letters (CAPS LOCK) in the subtitle increases the likelihood of success.
H2d: The use of single dots in the subtitle decreases the likelihood of success.

H3: There is a relationship between the presence of industry-specific jargon words in the subtitle of a crowdfunding campaign and its chances of success.

**Statistical Methods Used**
-Anderson–Darling test – Checks whether quantitative data follows a distribution close to normal. Unlike the Shapiro–Wilk test, it has no restrictions on sample size.
-Mann–Whitney U test – Non-parametric alternative to the Student’s t-test, used when the assumption of normality is not met. Compares two independent groups (successful vs. unsuccessful campaigns).
-Chi-squared test of independence – Checks whether two categorical variables are statistically independent by comparing observed vs. expected counts.
-Cramér’s V coefficient – Measures the strength of association between two categorical variables when the Chi-squared test shows a significant relationship.
-Spearman’s rank correlation – Measures the strength and direction of the relationship between two variables based on their ranks rather than raw values.
-Linear regression analysis – Explains how the value of a dependent variable (e.g., percent funded) changes under the influence of one or more independent variables.
-Logistic regression analysis – Models the relationship between one or more independent variables and a binary dependent variable (0 = failure, 1 = success), using the maximum likelihood estimation method.

**Notes**
The sample CSV files provided are only a small subset of the full Kickstarter dataset.

The research was focused exclusively on Tabletop Games campaigns.

If a particular aspect of the study was explored in more depth (e.g., further analysis of the number of periods rather than just their presence), it indicates that — unlike in other cases — the relationship was statistically significant for the full research sample, making it worthwhile to investigate the topic in greater detail.
