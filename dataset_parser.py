import pandas as pd
from sklearn.model_selection import train_test_split

#Loading the dataset
df=pd.read_csv('np20ng.csv')

#Applying Stratified Sampling to extract 20k rows 
df_sampled, _ =train_test_split(
    df,
    train_size=50000,
    stratify=df['category'],
    random_state=42
)


# df_sampled.to_csv('custom_nepali_news_dataset.csv',index=False)
df_sampled.to_csv('50k_news_dataset.csv',index=False)

print(df_sampled['category'].value_counts(normalize=True))
