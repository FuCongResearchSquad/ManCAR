import pandas as pd
import json
import os
import pandas as pd
import json
import os
MIN_LEN = 5

with open('../raw/Arts_Crafts_and_Sewing/meta_Arts_Crafts_and_Sewing.jsonl', 'r', encoding='utf-8') as f:
    data = f.readlines()
    meta = [json.loads(line) for line in data]
with open('../raw/Arts_Crafts_and_Sewing/Arts_Crafts_and_Sewing.jsonl', 'r', encoding='utf-8') as f:
    data = f.readlines()
    review = [json.loads(line) for line in data]

meta_map = {r['parent_asin']: r for r in meta}
review_map = {f"{r['user_id']}_{r['parent_asin']}": r for r in review}

train_df = pd.read_csv('../raw/Arts_Crafts_and_Sewing/Arts_Crafts_and_Sewing.train.csv')
valid_df = pd.read_csv('../raw/Arts_Crafts_and_Sewing/Arts_Crafts_and_Sewing.valid.csv')
test_df = pd.read_csv('../raw/Arts_Crafts_and_Sewing/Arts_Crafts_and_Sewing.test.csv')


train_df = train_df[train_df['rating'] > 3]
train_df = train_df[train_df['parent_asin'].apply(lambda x: meta_map[x]['title'] is not None)]
train_df = train_df[train_df['history'].notnull()]
train_df = train_df[train_df['history'].apply(lambda x: all([meta_map[his]['title'] is not None for his in x.split()]))]
train_df = train_df[train_df['history'].apply(lambda x: len(x.split()) >= MIN_LEN)]



valid_df = valid_df[valid_df['rating'] > 3]
valid_df = valid_df[valid_df['history'].notnull()]
valid_df = valid_df[valid_df['history'].apply(lambda x: x is not None and len(x.split()) >= MIN_LEN)]
print("finish filter history < 5 or None")
train_asin = set(train_df['parent_asin'].unique())
train_asin.update(train_df['history'].str.split().explode().unique())
valid_df = valid_df[valid_df['parent_asin'].apply(lambda x: x in train_asin)]
print("finish filter parent_asin not in train_df")
valid_df = valid_df[valid_df['history'].apply(lambda x: all([his in train_asin for his in x.split()]))]
print("finish filter history not in train_df")

test_df = test_df[test_df['rating'] > 3]
test_df = test_df[test_df['history'].notnull()]
test_df = test_df[test_df['history'].apply(lambda x: len(x.split()) >= MIN_LEN)]
print("finish filter history < 5 or None")
test_df = test_df[test_df['parent_asin'].apply(lambda x: x in train_asin)]
print("finish filter parent_asin not in train_df")
test_df = test_df[test_df['history'].apply(lambda x: all([i in train_asin for i in x.split()]))]
print("finish filter history not in train_df")

os.makedirs('../processed/Arts_Crafts_and_Sewing', exist_ok=True)
train_df.to_csv('../processed/Arts_Crafts_and_Sewing/Arts_Crafts_and_Sewing.train.csv', index=False)
valid_df.to_csv('../processed/Arts_Crafts_and_Sewing/Arts_Crafts_and_Sewing.valid.csv', index=False)
test_df.to_csv('../processed/Arts_Crafts_and_Sewing/Arts_Crafts_and_Sewing.test.csv', index=False)