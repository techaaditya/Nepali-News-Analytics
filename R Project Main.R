getwd()
setwd("C:/Users/user/Desktop/R Semester Project")
getwd()

#installing required packages.
install.packages(c("readr","dplyr","stringr","tm","quanteda","tidytext", 
"ggplot2","wordcloud","RColorBrewer","caret","e1071","randomForest" ,
"udpipe","textrank"))

#Loading the Libraries.
library(readr);library(dplyr);library(stringr);library(tm);library(quanteda);
library(tidytext);library(ggplot2);library(wordcloud);library(RColorBrewer);
library(caret);library(e1071);library(randomForest);
library(udpipe);library(textrank)

# Phase 1 :Data Collection and Preprocessing

#Loading the custom Nepali News Dataset.
news_data<-read_csv("50k_news_dataset.csv",show_col_types = FALSE)

#Initial Data Exploration
str(news_data)
head(news_data)
tail(news_data)
summary(news_data)
dim(news_data)

#Checking and Removing missing values.
colSums(is.na(news_data))
news_data<-na.omit(news_data)
dim(news_data)

#Checking and Removing Duplicate Rows.
sum(duplicated(news_data))
news_data[duplicated(news_data), ]
news_data<-news_data[!duplicated(news_data), ]

#need to see whether it is essential or not ###
#Text Preprocessing Pipeline
#Function for Nepali Text Cleaning
clean_nepali_text<-function(text) {
  text<-as.character(text)
  text<-gsub("<.*?>","",text)
  text<-gsub("http\\S+|www\\S+|https\\S+","",text)
  text<-gsub("\\S+@\\S+","",text)
  text<-gsub("\\s+"," ",text)
  text<-trimws(text)
  return(text)
}
#cleaning the content and headline
news_data$cleaned_text<-clean_nepali_text(news_data$content)
news_data$cleaned_headline<-clean_nepali_text(news_data$heading)
#creating a corpus from cleaned text
news_corpus<-corpus(news_data,text_field = "cleaned_text")
#adding document variables for modeling by category and source.
docvars(news_corpus,"category")<-news_data$category
docvars(news_corpus,"source")<-news_data$source

#Advanced Text Processing
news_tokens<-tokens(news_corpus,
                    remove_punct = TRUE,
                    remove_symbols = TRUE,
                    remove_numbers = TRUE,
                    remove_url = TRUE)
news_dfm<-dfm(news_tokens)
#trimming sparse terms from dfm(document feature matrix)
news_dfm<-dfm_trim(news_dfm,min_docfreq = 0.01,docfreq_type = "prop")

# Phase 2 : Exploratory Data Analysis(EDA)

# Category and Source Distribution 
source_counts <- count(news_data, source, sort = TRUE)
source_counts
category_counts <- count(news_data, category, sort = TRUE)
category_counts

#Category Distribution Analysis
category_counts <- table(news_data$category)
ggplot(data = as.data.frame(category_counts), aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme_minimal() +
  labs(title = "News Articles by Category",
       x = "Category",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45,hjust=1))

# Source Distribution Analysis
source_counts <- table(news_data$source)
ggplot(as.data.frame(source_counts), aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  theme_minimal() +
  labs(title = "News Articles by Source",
       x = "Source",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45,hjust=1))

# Heading Character Length Analysis
heading_character_length <- nchar(news_data$heading)
summary(heading_character_length)
hist(
  heading_character_length,
  main = "Distribution of Heading Character Lengths",
  xlab = "Number of Characters",
  ylab = "Frequency",
  col = "red",
  border = "black")
grid()

# Content Character Length Analysis
content_character_length <- nchar(news_data$content)
summary(content_character_length)
hist(
  content_character_length,
  main = "Distribution of content Character Lengths",
  xlab = "Number of Characters",
  ylab = "Frequency",
  col = "blue",
  border = "black")
grid()

# Heading Word Count Analysis
heading_word_count <- str_count(news_data$heading, "\\w+")
summary(heading_word_count)
hist(
  heading_word_count,
  main = "Distribution of Heading Word Counts",
  xlab = "Number of Words",
  ylab = "Frequency",
  col = "grey",
  border = "black"
)
grid()

# Content Word Count Analysis
content_word_count <- str_count(news_data$content, "\\w+")
summary(content_word_count)
hist(
  content_word_count,
  main = "Distribution of Content Word Counts",
  xlab = "Words per Content",
  ylab = "Frequency",
  col = "orange",
  border = "black"
)
grid()


# Content Length Analysis
news_data$content_char_length <- nchar(news_data$cleaned_text)
news_data$content_word_count <- stringr::str_count(news_data$cleaned_text, "\\w+")

summary(news_data[, c("content_char_length", "content_word_count")])

# Content Length by Word Count.
ggplot(news_data, aes(x = content_word_count)) +
  geom_histogram(bins = 50, fill = "orange", alpha = 0.7) +
  labs(
    title = "Distribution of Article Content Length (Words)",
    x = "Number of Words",
    y = "Frequency"
  ) +
  theme_minimal()

#Content Length by Character Count.
ggplot(news_data, aes(x = content_char_length)) +
  geom_histogram(bins = 50, fill = "blue", alpha = 0.7) +
  labs(
    title = "Distribution of Article Content Length (Characters)",
    x = "Number of Characters",
    y = "Frequency"
  ) +
  theme_minimal()

#Content Length by Category
ggplot(news_data, aes(x = category, y = content_word_count)) +
  geom_boxplot(fill = "lightcoral", alpha = 0.7) +
  labs(
    title = "Article Content Length by Category",
    x = "Category",
    y = "Number of Words"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Frequency Analysis and Visualization
#Top terms analysis
top_features<-topfeatures(news_dfm,50)
top_features_df<-data.frame(
  Term=names(top_features),
  Frequency=as.numeric(top_features)
)
print(head(top_features_df,10))

#Creating frequency bar chart
ggplot(head(top_features_df,20),aes(x=reorder(Term,Frequency),
 y= Frequency))+geom_col(fill="steelblue")+coord_flip()+
  labs(title = "Top 20 Most Frequent Terms",
       x="Terms",y="Frequency")+theme_minimal()

#Generate word cloud
set.seed(123)
wordcloud(words = top_features_df$Term[1:50],
          freq = top_features_df$Frequency[1:50],
          min.freq = 2,
          max.words = 100,
          random.order = FALSE,
          rot.per = 0.35,
          colors = brewer.pal(8, "Dark2"))

# N gram Analysis:
# Bigram Analysis:
news_bigrams<-tokens_ngrams(news_tokens,n=2)
bigram_dfm<-dfm(news_bigrams)
top_bigrams<-topfeatures(bigram_dfm,28)

#Visualize the bigrams
bigram_df<-data.frame(
  Bigram=names(top_bigrams),
  Frequency=as.numeric(top_bigrams)
)

ggplot(bigram_df,aes(x=reorder(Bigram,Frequency),y=Frequency))+
  geom_col(fill="darkgreen")+
  coord_flip()+
  labs(title = "Top 20 Bigrams",
       x="Bigrams",y="Frequency")+theme_minimal()

#Trigram Analysis:
news_trigrams<-tokens_ngrams(news_tokens,n=3)
trigram_dfm<-dfm(news_trigrams)
top_trigrams<-topfeatures(trigram_dfm,20)

#Visualize the trigrams
trigram_df <- data.frame(
  Trigram = names(top_trigrams),
  Frequency = as.numeric(top_trigrams)
)

ggplot(trigram_df, aes(x = reorder(Trigram, Frequency), y = Frequency)) +
  geom_col(fill = "purple") +
  coord_flip() +
  labs(
    title = "Top 20 Trigrams",
    x = "Trigrams",
    y = "Frequency"
  ) +
  theme_minimal()

# Lexical Diversity Analysis

# Calculate lexical diversity (Type-Token Ratio)
calculate_lexical_diversity <- function(tokens) {
types <- length(unique(unlist(tokens)))
tokens_total <- length(unlist(tokens))
return(types / tokens_total)
}

# Lexical diversity by category
lexical_diversity <- news_data %>%
group_by(category) %>%
summarise(
avg_lexical_diversity = mean(sapply(strsplit(cleaned_text, "\\s+"), 
function(x) length(unique(x)) / length(x))),.groups = "drop")

print(lexical_diversity)

# Visualize lexical diversity
ggplot(lexical_diversity, aes(x = reorder(category, avg_lexical_diversity), 
y = avg_lexical_diversity)) +geom_col(fill = "purple", 
alpha = 0.7) +coord_flip() +
labs(title = "Average Lexical Diversity by Category",
x = "Category", y = "Lexical Diversity (TTR)") +
theme_minimal()
