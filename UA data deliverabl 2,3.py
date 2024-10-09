import numpy as np
import pandas as pd
new_data = pd.read_csv('new_data.csv')
new_data.head()
calls= pd.read_csv('callsf0d4f5a.csv')
sentiment= pd.read_csv('sentiment_statisticscc1e57a.csv')
reasons= pd.read_csv('reason18315ff.csv')
customer= pd.read_csv('customers2afd6ea.csv')
new_data['call_transcript'].iloc[1]
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to clean the transcript text
def clean_transcript(text):
    # Remove patterns like \n\nCustomer: and \n\nAgent:
    text = re.sub(r'\n\n(Customer|Agent):', '', text)
    # Remove special characters and lowercase the text
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return cleaned_text

# Apply the cleaning function to the call_transcript column
new_data['cleaned_transcript'] = new_data['call_transcript'].apply(clean_transcript)

# Display cleaned transcripts for verification
print(new_data['cleaned_transcript'].head())
new_data['call_transcript'].iloc[1]
# Use TF-IDF to identify important words, excluding common stop words
tfidf = TfidfVectorizer(stop_words='english')

# Fit the vectorizer on the cleaned transcripts
tfidf_matrix = tfidf.fit_transform(new_data['cleaned_transcript'])

# Get the feature names (words) and their TF-IDF scores
words = tfidf.get_feature_names_out()
# Sum the TF-IDF values for each word across all documents
word_sums = tfidf_matrix.sum(axis=0)

# Create a DataFrame of words and their summed TF-IDF scores
import pandas as pd
word_freq = pd.DataFrame({'word': words, 'tfidf': word_sums.A1})

# Sort by TF-IDF scores to identify the most frequent words/issues
word_freq.sort_values(by='tfidf', ascending=False, inplace=True)
print(word_freq.head(20))  # Top 20 recurring words
reasons.head(6)
reasons.groupby(reasons['primary_call_reason']).value_counts()
reasons['primary_call_reason'].value_counts()
# Assuming 'word_freq' contains frequent words from TF-IDF
# and 'primary_call_reason' is a column in your dataset

# Createed a word_freq dataframe 
# word_freq = pd.DataFrame({'word': ['change', 'baggage', 'upgrade', 'seat', 'flight'], 'tfidf': [0.9, 0.8, 0.7, 0.6, 0.5]})

# Sample 'primary_call_reason' data (replace with actual data)
# new_data['primary_call_reason'] have the reasons column
primary_call_reason_map = {
    'Voluntary Change': ['change', 'flight', 'book', 'date'],
    'Baggage': ['baggage', 'lost', 'claim'],
    'Upgrade': ['upgrade', 'seat', 'reserve'],
    'Mileage Plus': ['mileage', 'points', 'reward'],
    'Booking': ['book', 'reserve', 'confirm'],
    'Seating': ['seat', 'window', 'aisle'],
    # Add more mappings based on your understanding
}
# Create a function to map words to primary call reasons
def map_words_to_reasons(word):
    for reason, keywords in primary_call_reason_map.items():
        if word in keywords:
            return reason
    return 'Other'  # If no match found

# Apply this function to the 'word' column of word_freq
word_freq['primary_call_reason'] = word_freq['word'].apply(map_words_to_reasons)
# Now, summarize how often words are related to each call reason
reason_summary = word_freq.groupby('primary_call_reason').agg({
    'word': 'count',  # Number of words mapped to this reason
    'tfidf': 'sum'    # Sum of tfidf scores for this reason
}).reset_index()

# Sort by tfidf to see most impactful reasons
reason_summary = reason_summary.sort_values(by='tfidf', ascending=False)

# Show the summary
print(reason_summary)

# Output insights for self-service options
print("Self-service opportunities in IVR:")
for index, row in reason_summary.iterrows():
    reason = row['primary_call_reason']
    if reason != 'Other':
        print(f"Consider self-service options for Reason= {reason}, based on frequent terms like: {primary_call_reason_map[reason]}.")
import pandas as pd

# Creating a list of dictionaries with reasons and their related frequent words
reasons_words = [
    {"Reason": "Voluntary Change", "Frequent Words": ['change', 'flight', 'book', 'date']},
    {"Reason": "Upgrade", "Frequent Words": ['upgrade', 'seat', 'reserve']},
    {"Reason": "Baggage", "Frequent Words": ['baggage', 'lost', 'claim']},
    {"Reason": "Seating", "Frequent Words": ['seat', 'window', 'aisle']},
    {"Reason": "Booking", "Frequent Words": ['book', 'reserve', 'confirm']},
    {"Reason": "Mileage Plus", "Frequent Words": ['mileage', 'points', 'reward']}
]

# Convert the list of dictionaries into a DataFrame
df_reasons_words = pd.DataFrame(reasons_words)

# Display the DataFrame
print(df_reasons_words)
new_data.head(1)
# Deliverable 3:
cleaned_data = pd.read_csv('cleaned_data.csv')
cleaned_data.head(2)
cleaned_data['cleaned_transcript'].iloc[1]
Bar Chart of Call Reason Frequency:
#This chart will show you which call reasons are most frequently reported by customers.
Higher bars indicate more common reasons, which can help prioritize which issues to address in customer service or self-service options.
import matplotlib.pyplot as plt
import seaborn as sns

call_reason_counts = cleaned_data['primary_call_reason'].value_counts()

plt.figure(figsize=(12,10))
sns.barplot(x= call_reason_counts.values , y=call_reason_counts.index, orient='h', palette='viridis')
plt.title('Frequency of Primary Call Reasons')
plt.xlabel('Primary Call Reasons')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()
call_reason_counts.head(6)
Word Cloud of Call Transcripts:
#The word cloud visualizes the most common words spoken during calls, indicating recurring themes or issues.
Larger words represent more frequently used terms, which can guide your understanding of customer concerns and pain points.
from wordcloud import WordCloud

# Generate a word cloud from the cleaned transcript
text = ' '.join(new_data['cleaned_transcript'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Call Transcripts')
plt.show()
Count Plot of Call Reasons by elite_level_code
#This count plot will reveal if elite customers have different concerns compared to non-elite customers.
Understanding the distribution can help tailor services or self-service options for specific customer segments.
plt.figure(figsize=(17, 9))
sns.countplot(data=cleaned_data, x='primary_call_reason', hue='elite_level_code', palette='Set2')
plt.title('Count of Call Reasons by Elite Level Code')
plt.xlabel('Primary Call Reasons')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Elite Level Code')
plt.show()
Heatmap of Correlations:
#The heatmap illustrates the relationships between different numeric features, indicating where there may be strong correlations.
For example, a strong positive correlation between AHT_seconds and AST_seconds could suggest that calls taking longer to be answered also tend to take longer to resolve.
import plotly.express as px

# Select only the specified columns
selected_columns = ['AHT_seconds', 'AST_seconds', 'average_sentiment', 'elite_level_code', 'silence_percent_average']

numeric_features = new_data[selected_columns]

corr_matrix = numeric_features.corr()

fig = px.imshow(corr_matrix, 
                text_auto=True, 
                aspect="auto", 
                color_continuous_scale='RdBu_r', 
                title='Correlation Heatmap of Selected Features')
fig.show()
Box Plot of AHT by Call Reason:
#This box plot provides a visual representation of the distribution of AHT for each call reason.
It can help identify which reasons lead to longer handling times, suggesting that these issues may need more focused resolution strategies.

plt.figure(figsize=(12, 11))
sns.boxplot(data=new_data, y='primary_call_reason', x='AHT_seconds', palette='coolwarm')
plt.title('Box Plot of AHT by Primary Call Reason (Horizontal)')
plt.xlabel('Average Handling Time (AHT in seconds)')
plt.ylabel('Primary Call Reasons')
plt.show()
# Clustering Analysis
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming cleaned_data is your cleaned DataFrame
# Example: cleaned_data = pd.read_csv('cleaned_data.csv')

# Step 1: Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_data['cleaned_transcript'])
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 2: Use the elbow method to determine the optimal number of clusters
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plotting the elbow graph
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal K')
plt.grid()
plt.show()
# Step 3: Apply K-Means with the chosen K
k = 3  # Replace with the optimal K determined from the elbow method
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X)

# Add the cluster labels to your cleaned_data DataFrame
cleaned_data['cluster'] = clusters
import seaborn as sns

# Step 4: Visualize the distribution of call reasons across clusters
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', data=cleaned_data)
plt.title('Count of Call Reasons by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.grid()
plt.show()
