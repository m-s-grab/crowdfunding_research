import os
try:
    import pandas as pd
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
except ModuleNotFoundError as e:
    print("Missing library:", e.name)
    print('Please run: pip install -r requirements.txt')
    exit(1)

# Data paths
data_folder = os.path.join("data")
input_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".csv")]
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "Tabletop_Games_cleaned.csv")

# Loading and filtering data by "Tabletop Games" category only
df_all = pd.concat((pd.read_csv(f, low_memory=False) for f in input_files), ignore_index=True)
df = df_all[df_all['category'].astype(str).str.contains("Tabletop Games", case=False, na=False)].copy()

# Adding goal in USD and % funded columns
df['goal'] = pd.to_numeric(df['goal'], errors='coerce')
df['pledged'] = pd.to_numeric(df['pledged'], errors='coerce')
df['static_usd_rate'] = pd.to_numeric(df['static_usd_rate'], errors='coerce')
df['usd_pledged'] = pd.to_numeric(df['usd_pledged'], errors='coerce')

df['goal_usd'] = df['goal'] * df['static_usd_rate']
df['percent_funded'] = (df['pledged'] / df['goal']).round(4)

# Deduplication by chosing the most recent campaign record (most funded) 
def pick_highest_usd(group):
    max_val = group['usd_pledged'].max()
    top = group[group['usd_pledged'] == max_val]
    return top.sample(n=1, random_state=42) if len(top) > 1 else top
    #Finding campaigns with same name and launching time (records duplicated over time)
df = df.groupby(['name', 'launched_at'], group_keys=False).apply(pick_highest_usd)

#Filtering the language - english blurbs only
DetectorFactory.seed = 0
def is_english(text):
    try:
        return detect(str(text)) == 'en'
    except LangDetectException:
        return False

df = df[df['blurb'].apply(is_english)]

# Changing campaign final state to 0/1
df['state'] = df['state'].map({'successful': 1, 'failed': 0})

# Removing any failed campaigns that would have 100% and more funding or succesful campaigns with less than 100% funding.
df = df[~((df['state'] == 0) & (df['percent_funded'] >= 1.0))]
df = df[~((df['state'] == 1) & (df['percent_funded'] < 1.0))]

# Removing campaigns with funding goal being 15 USD or less
df_filtered = df[df['goal_usd'] >= 15].copy()

# Removing outliers from succesful campaigns (very small amount of campaigns tends to get over 2000% of the goal reached)
# Using the IQR method
df_success = df_filtered[df_filtered['state'] == 1].copy()
Q1 = df_success['percent_funded'].quantile(0.25)
Q3 = df_success['percent_funded'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df_success = df_success[df_success['percent_funded'] <= upper_bound]

# Combining failed and successful campaigns again
df_failed = df_filtered[df_filtered['state'] == 0]
df_cleaned = pd.concat([df_success, df_failed], ignore_index=True)

# Removing unnecessary columns
columns_to_drop = [
    'country', 'creator', 'currency_symbol', 'currency_trailing_code',
    'disable_communication', 'friends', 'is_backing', 'is_starrable',
    'is_starred', 'location', 'permissions', 'photo', 'profile', 'slug',
    'source_url', 'spotlight', 'staff_pick', 'state_changed_at',
    'unread_messages_count', 'unseen_activity_count', 'is_disliked',
    'is_launched', 'is_liked', 'prelaunch_activated', 'video'
]
df_cleaned = df_cleaned.drop(columns=[col for col in columns_to_drop if col in df_cleaned.columns])

# Saving the database
df_cleaned.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"✅ Final data saved to: {output_file}")
print(f"📊 The number of campaigns after the cleanup: {len(df_cleaned)}")
