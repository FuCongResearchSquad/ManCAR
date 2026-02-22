import pandas as pd
import json
import os

RAW_DATA_PATH = "./raw/Arts_Crafts_and_Sewing/"
PRO_DATA_PATH = "./processed/Arts_Crafts_and_Sewing"

TARGET_PATH = "./processed/Arts_Crafts_and_Sewing/"
os.makedirs(TARGET_PATH, exist_ok=True)

def process_item_csv():

    
    train_path = os.path.join(PRO_DATA_PATH, "Arts_Crafts_and_Sewing.train.csv")
    if not os.path.exists(train_path):
        print(f"can't find {train_path}")
        return

    train_df = pd.read_csv(train_path)

    if 'Yelp' in RAW_DATA_PATH:
        asins = set(train_df['business_id'].unique())
        asins.update(train_df['history'].str.split().explode().unique())
    else:
        asins = set(train_df['parent_asin'].unique())
        history_asins = train_df['history'].str.split().explode().unique()
        asins.update(history_asins)

    asins = {a for a in asins if pd.notnull(a)}

    sorted_asins = sorted(list(asins))
    asin_to_id = {asin: i + 1 for i, asin in enumerate(sorted_asins)}
    
    print(f"ID train datasets have {len(asin_to_id)} items")

    meta_path = os.path.join(RAW_DATA_PATH, "meta_Arts_Crafts_and_Sewing.jsonl")
    meta_dict = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                m = json.loads(line)
                meta_dict[m.get('parent_asin')] = m
    item_rows = []
    
    for p_asin in sorted_asins:
        i_id = asin_to_id[p_asin]
        m_data = meta_dict.get(p_asin, {})
        
        title = m_data.get('title', '')
        description = m_data.get('description', '')
        
        cats = m_data.get('categories', [])
        if isinstance(cats, list):
            if len(cats) > 0 and isinstance(cats[0], list):
                cat_str = ", ".join([str(i) for i in cats[0]])
            else:
                cat_str = ", ".join([str(i) for i in cats])
        else:
            cat_str = str(cats)
        text_content = f"Title: {title}\nDescription: {description}\nCategories: {cat_str}"
        
        item_rows.append({
            'parent_asin': p_asin,
            'item_id': i_id,
            'second_cate_id': 0,
            'third_cate_id': 0,
            'store_id': m_data.get('store_id', 0),
            'text': text_content,
            'text_emb': ""
        })


    item_df = pd.DataFrame(item_rows)
    output_file = os.path.join(TARGET_PATH, "Arts_Crafts_and_Sewing.item.csv")
    item_df.to_csv(output_file, index=False)
    

if __name__ == "__main__":
    process_item_csv()