import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Load the user data from the CSV file
@st.cache  # Add caching to improve app performance
def load_data():
    df = pd.read_csv('user_data.csv')
    return df

df = load_data()

# Define the target user
target_user_id = 1

# Filter out the target user from the DataFrame
target_user = df[df['User_ID'] == target_user_id].iloc[0]

# One-hot encode non-numeric columns
non_numeric_columns = ['Social']
df_encoded = pd.get_dummies(df.drop(['User_ID'], axis=1), columns=non_numeric_columns)

# One-hot encode the target user
target_user_encoded = pd.get_dummies(target_user.drop(['User_ID']), columns=non_numeric_columns)

# Realign columns to ensure the same set of columns in both DataFrames
df_encoded, target_user_encoded = df_encoded.align(target_user_encoded, join='outer', axis=1)

# Fill missing values with zeros
df_encoded = df_encoded.fillna(0)
target_user_encoded = target_user_encoded.fillna(0)

# Calculate user similarity based on website usage metrics
user_similarity = cosine_similarity(df_encoded, target_user_encoded)

# Get indices of users most similar to the target user
similar_user_indices = user_similarity.argsort()[0][-5:][::-1]

# Get the top recommendations from similar users
top_recommendations = df.loc[similar_user_indices, ['User_ID', 'Number_of_Sessions', 'Session_Duration', 'Page_Views', 'CTR', 'Bounce_Rate', 'Social_Media_Shares', 'Conversion_Rate', 'Time_on_Site', 'Active_Months', 'User_Interactions', 'Age', 'Access_Time', 'Social']]

# Implement content-based filtering using user demographic information
age_range = target_user['Age'] // 10 * 10
content_based_recommendations = df[df['Age'] // 10 * 10 == age_range].sample(5)

# Combine collaborative filtering and content-based filtering recommendations
recommendations = pd.concat([top_recommendations, content_based_recommendations]).drop_duplicates()

# Create the Streamlit app
def main():
    st.title('Recommendation System')
    
    st.subheader('Top Recommendations for User ID 1:')
    st.table(recommendations)

if __name__ == '__main__':
    main()
