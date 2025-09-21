import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('np20ng.csv')

df_50k, _ =train_test_split(
    df,
    train_size=50000,
    stratify=df['category'],
    random_state=42
)

train_df,test_df=train_test_split(
    df_50k,
    train_size=0.8,
    stratify=df_50k['category'],
    random_state=42
)

train_df.to_csv('nepali_news_train_40k.csv',index=False)
test_df.to_csv('nepali_news_test_10k.csv',index=False)