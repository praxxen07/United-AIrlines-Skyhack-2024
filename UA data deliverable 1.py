import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
calls= pd.read_csv('callsf0d4f5a.csv')
calls.head()
sentiment= pd.read_csv('sentiment_statisticscc1e57a.csv')
sentiment.head()
reasons= pd.read_csv('reason18315ff.csv')
reasons.head()
customer= pd.read_csv('customers2afd6ea.csv')
customer.head()
# How are AHT and AST Calculated?
# AHT (Average Handle Time):
# Time from when the agent picks up the call to when they hang up.

# AST (Average Speed to Answer):
# Time spent by the customer in queue till the agent answers the call
# Formula:
# AST = Total Waiting Time / Total Number of Calls

# Deliverables:
# Long average handle time (AHT) affects both efficiency and customer satisfaction. Explore the factors 
# contributing to extended call durations, such as agent performance, call types, and sentiment. Identify
# key drivers of long AHT and AST, especially during high volume call periods. Additionally, could 
# you quantify the percentage difference between the average handling time for the most frequent and
# least frequent call reasons?
reasons.sample(8)
reasons['primary_call_reason'].value_counts()
calls.columns 
reasons.columns
sentiment.columns
customer.columns
# Preprocessing Missing Values:
print("Before preprocessing 'reasons' dataset missing values:")
print(reasons.isnull().sum())                           
print("Before preprocessing 'sentiment' dataset missing values:")
print(sentiment.isnull().sum())
print("After preprocessing 'sentiment' dataset missing values:")
sentiment['agent_tone'].fillna('neutral', inplace=True)
sentiment['average_sentiment'].fillna(sentiment['average_sentiment'].mean(), inplace=True)
print(sentiment.isnull().sum())
print("Before preprocessing 'customer' dataset missing values:")
print(customer.isnull().sum())
print("After preprocessing 'customer' dataset missing values:")
customer['elite_level_code'].fillna(0 ,inplace=True)
print(sentiment.isnull().sum())
# Explore the factors 
# contributing to extended call durations, such as agent performance, call types, and sentiment.
# Data Preprocessing
call_reasons= calls.merge(reasons, on='call_id', how='left')
call_reasons.columns
call_sent= call_reasons.merge(sentiment, on= 'call_id', how='left')
call_sent.columns
full_data = call_sent.merge(customer, on='customer_id', how='left')
full_data.columns
#Calculating AHT in minutes :
full_data['call_start_datetime'] = pd.to_datetime(full_data['call_start_datetime'])
full_data['call_end_datetime'] =  pd.to_datetime(full_data['call_end_datetime'])
full_data['AHT']= (full_data['call_end_datetime']-full_data['call_start_datetime'])
full_data.sample(5)
full_data = pd.get_dummies(full_data, columns=['primary_call_reason', 'agent_tone', 'customer_tone', 'elite_level_code'], drop_first=True)
type(full_data['AHT'])
full_data.head()
# Convert all boolean columns to integers in the entire DataFrame
full_data[full_data.select_dtypes(include=['bool']).columns] = full_data.select_dtypes(include=['bool']).astype(int)
full_data.head(3)
# Extracting datetime components
full_data['call_start_hour'] = full_data['call_start_datetime'].dt.hour
full_data['call_start_minute'] = full_data['call_start_datetime'].dt.minute


full_data['call_end_hour'] = full_data['call_end_datetime'].dt.hour
full_data['call_end_minute'] = full_data['call_end_datetime'].dt.minute
#Now agent_assigned column is not in right format so we will first convert it into datetime format 
full_data['agent_assigned_datetime'] = pd.to_datetime(full_data['agent_assigned_datetime'], errors='coerce')
# Same extraction for other datetime columns
full_data['agent_assigned_hour'] = full_data['agent_assigned_datetime'].dt.hour
full_data['agent_assigned_minute'] = full_data['agent_assigned_datetime'].dt.minute
# Assuming full_data['AHT'] is in timedelta64[ns]
# Convert AHT (timedelta) to total seconds and create a new column
full_data['AHT_seconds'] = full_data['AHT'].dt.total_seconds()
full_data.head(2)
# Convert AHT (timedelta) to total seconds and create a new column
full_data['AHT_seconds'] = full_data['AHT'].dt.total_seconds()
from sklearn.model_selection import train_test_split
x= full_data.drop(columns=['AHT_seconds', 'call_id', 'customer_id', 'agent_id_x','agent_id_y','call_end_minute','call_start_hour','call_start_minute','agent_assigned_hour','agent_assigned_minute','customer_name','call_end_hour','call_start_datetime','agent_assigned_datetime','call_end_datetime','call_transcript','AHT'])
y= full_data['AHT_seconds']
x.sample(9)
x_train, x_test, y_train, y_test= train_test_split(x,y, random_state=42, test_size= 0.2)
print(x.shape)
print(x_train.shape, x_test.shape)
full_data.columns

# Model making

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
model = CatBoostRegressor(verbose=0, random_seed=42)
#Train the model
model.fit(x_train, y_train)
# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model using MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
# Print evaluation metrics
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"{r2=}")
# Get feature importances
feature_importances = model.get_feature_importance()
feature_names = x_train.columns

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the most important feature
print("\nMost important feature to predict AHT_seconds:")
print(feature_importance_df)
print(feature_importance_df.max())
important_features = feature_importance_df[feature_importance_df['Importance'].abs() > 0]
print(important_features)
import matplotlib.pyplot as plt
import seaborn as sns

# Display the top 10 most important features
top_n = 10 
print(feature_importance_df.head(top_n))


plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n))
plt.title('Top Important Features Affecting AHT')
plt.show()
#  Heat Map (Co-relation between features and AHT )
call_reasons1= calls.merge(reasons, on='call_id', how='left')
call_sent1= call_reasons1.merge(sentiment, on= 'call_id', how='left')
new_data= call_sent1.merge(customer, on='customer_id', how='left')
new_data['call_start_datetime']= pd.to_datetime(new_data['call_start_datetime'])
new_data['call_end_datetime']= pd.to_datetime(new_data['call_end_datetime'])
new_data['AHT_seconds']= (new_data['call_end_datetime']- new_data['call_start_datetime']).dt.total_seconds()
new_data.head(3)
relevant_columns = [
    'agent_tone',
    'primary_call_reason',
    'customer_tone',
    'average_sentiment',
    'silence_percent_average',
    'elite_level_code',
    'AHT_seconds' 
]


heatmap_data = new_data[relevant_columns]            # Trying to show how the factors present in relevant columns affects AHT:


heatmap_data = pd.get_dummies(heatmap_data, drop_first=True)            # Convert categorical columns to numerical:
correlation_matrix = heatmap_data.corr()
plt.figure(figsize=(12, 10))

# Draw the heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})

plt.title('Correlation Heatmap of Factors Contributing to AHT')
plt.xlabel('Factors')
plt.ylabel('Factors')

plt.show()
#Dynamic Visualization using plotly
import plotly.express as px

relevant_columns = [
    'agent_tone',
    'primary_call_reason',
    'customer_tone',
    'average_sentiment',
    'silence_percent_average',
    'elite_level_code',
    'AHT_seconds'     
]
heatmap_data = new_data[relevant_columns]


heatmap_data = pd.get_dummies(heatmap_data, drop_first=True)                   # Categorical encoding:

correlation_matrix = heatmap_data.corr()


correlation_matrix = correlation_matrix.reset_index().melt(id_vars='index')     # Reset index to create a tidy format for Plotly

correlation_matrix.columns = ['Feature1', 'Feature2', 'Correlation']   # Rename the columns for clarity
fig = px.imshow(
    correlation_matrix.pivot(index='Feature1', columns='Feature2', values='Correlation'),           #Plotting
    color_continuous_scale='RdBu',
    labels=dict(x="Features", y="Features", color="Correlation"),
    title='Correlation Heatmap of Factors Contributing to AHT',
)

fig.update_layout(
    width=1000, 
    height=800,   
    title_font=dict(size=24),  
    xaxis_title_font=dict(size=18),  
    yaxis_title_font=dict(size=18), 
    xaxis=dict(tickfont=dict(size=12)), 
    yaxis=dict(tickfont=dict(size=12)),  
)
fig.show()
# Percentage difference between the average handling time for the most frequent and least frequent call reasons!!
new_data.columns
# Removing Outliers
Q1= new_data['AHT_seconds'].quantile(0.25)
Q3 = new_data['AHT_seconds'].quantile(0.75)
IQR = (Q3-Q1)
upper_limit = (Q3+1.5*IQR)
lower_limit = (Q1-1.5*IQR)
print("upper_limit: ", upper_limit)
print("lower_limit: ", lower_limit)
filtered_data= new_data[(new_data['AHT_seconds']<= upper_limit) & (new_data['AHT_seconds'] >= lower_limit)]
sns.boxplot( filtered_data['AHT_seconds'])
aht_reason= filtered_data.groupby('primary_call_reason')['AHT_seconds'].mean().reset_index()
# Here i calculated average AHT of each reason
aht_reason.sample(5)
frequency_by_reason = new_data['primary_call_reason'].value_counts().reset_index()
frequency_by_reason.columns = ['primary_call_reason', 'Frequency']
frequency_by_reason.head()
#Now I have calculate most frequent AHT call reasons

merged_data = pd.merge(aht_reason, frequency_by_reason, on='primary_call_reason')
merged_data = pd.merge(aht_reason, frequency_by_reason, on='primary_call_reason')
sorted_data = merged_data.sort_values(by='Frequency', ascending=False)
sorted_data.head()
most_frequent = merged_data.loc[merged_data['Frequency'].idxmax()]           #Identify the most and least frequent call reasons
least_frequent = merged_data.loc[merged_data['Frequency'].idxmin()]
most_frequent
least_frequent
most_frequent_aht = most_frequent['AHT_seconds']
least_frequent_aht = least_frequent['AHT_seconds']

percentage_difference = ((most_frequent_aht - least_frequent_aht) / least_frequent_aht) * 100
print(f'Most Frequent Call Reason:- {most_frequent["primary_call_reason"]} | Average AHT:- {most_frequent_aht} seconds')
print(f'Least Frequent Call Reason:- {least_frequent["primary_call_reason"]} | Average AHT:- {least_frequent_aht} seconds')
print(f'Percentage Difference in AHT:- {percentage_difference:.2f}%')

new_data.columns
# Key drivers of long AHT and AST, especially during high volume call periods
#AST calculation:
new_data['call_start_datetime']= pd.to_datetime(new_data['call_start_datetime'])
new_data['agent_assigned_datetime'] = pd.to_datetime(new_data['agent_assigned_datetime'])
new_data['AST']= (new_data['agent_assigned_datetime']- new_data['call_start_datetime'])
new_data.head(1)
new_data['AST_seconds'] = new_data['AST'].dt.total_seconds()                #Converting AST into seconds for analysis
new_data.head(2)
new_data['call_date']= new_data['call_start_datetime'].dt.date
call_counts= new_data.groupby(new_data['call_date']).size().reset_index(name='call_count')
#Call counts per day
# Setting up a  threshold for high-volume (you can adjust this based on the data)
high_volume_threshold = call_counts['call_count'].quantile(0.90)                           # Top 10% of high-volume days
high_volume_days = call_counts[call_counts['call_count'] >= high_volume_threshold]['call_date']
high_volume_data = new_data[new_data['call_date'].isin(high_volume_days)]
high_volume_data.shape
#Visualization :
features = ['AHT_seconds', 'AST_seconds', 'primary_call_reason', 'agent_tone', 'customer_tone', 
            'average_sentiment', 'silence_percent_average', 'elite_level_code']
high_volume_data_encoded = pd.get_dummies(high_volume_data[features], drop_first=True)    #Converting Categorical into numerical
corr_matrix = high_volume_data_encoded.corr()

plt.figure(figsize=(12, 8))                         # Plot heatmap for AHT and AST correlations
sns.heatmap(corr_matrix[['AHT_seconds', 'AST_seconds']], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation of Features with AHT and AST')
plt.show()
#Key factors shown by Feature Importance:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Independent features and target variables for AHT and AST
X = high_volume_data_encoded.drop(columns=['AHT_seconds', 'AST_seconds'])
y_aht = high_volume_data_encoded['AHT_seconds']
y_ast = high_volume_data_encoded['AST_seconds']
X_train_aht, X_test_aht, y_train_aht, y_test_aht = train_test_split(X, y_aht, test_size=0.2, random_state=42)
X_train_ast, X_test_ast, y_train_ast, y_test_ast = train_test_split(X, y_ast, test_size=0.2, random_state=42)


rf_aht = RandomForestRegressor(n_estimators=100, random_state=42)       # Model training Random Forest Regressor for AHT
rf_aht.fit(X_train_aht, y_train_aht)

rf_ast = RandomForestRegressor(n_estimators=100, random_state=42)       # Model training Random Forest Regressor for AST   
rf_ast.fit(X_train_ast, y_train_ast)
# Get feature importance for AHT
importance_aht = rf_aht.feature_importances_
feature_importance_aht = pd.DataFrame({'Feature': X.columns, 'Importance': importance_aht})
feature_importance_aht.sort_values(by='Importance', ascending=False, inplace=True)

# Get feature importance for AST
importance_ast = rf_ast.feature_importances_
feature_importance_ast = pd.DataFrame({'Feature': X.columns, 'Importance': importance_ast})
feature_importance_ast.sort_values(by='Importance', ascending=False, inplace=True)


print("Top features driving AHT:")
print(feature_importance_aht.head())

print("\nTop features driving AST:")
print(feature_importance_ast.head())
#Visualising feature Importance using Plotly
# Visualize AHT feature importance:
fig_aht = px.bar(feature_importance_aht.head(10), x='Feature', y='Importance', title='Top Features Driving AHT')
fig_aht.show()

# Visualize AST feature importance:
fig_ast = px.bar(feature_importance_ast.head(10), x='Feature', y='Importance', title='Top Features Driving AST')
fig_ast.show()
