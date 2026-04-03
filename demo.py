import pandas as pd

# 1. Load the original raw data
# We use the semicolon separator because the UCI dataset is formatted that way
try:
    df = pd.read_csv('data.csv', sep=';')
except:
    df = pd.read_csv('data.csv')

# 2. Extract a small batch (e.g., 100 students) for the demo
demo_df = df.head(100)

# 3. Remove the 'Target' column
# The AI dashboard must predict the result, so we remove the "answer" first
if 'Target' in demo_df.columns:
    demo_df = demo_df.drop('Target', axis=1)

# 4. Save as a new demo file for the dashboard
demo_df.to_csv('demo_students.csv', index=False, sep=';')

print("✅ Success! 'demo_students.csv' created with 100 entries.")
