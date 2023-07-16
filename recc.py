import pandas as pd
import streamlit as st
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics.pairwise import cosine_similarity

# Load the user data from the CSV file
@st.cache  # Add caching to improve app performance
def load_data():
    df = pd.read_csv('user_data.csv')
    return df

df = load_data()

# Filter out non-numeric columns
numeric_columns = df.select_dtypes(include='number').columns
df_numeric = df[numeric_columns]

# Perform feature selection using SelectKBest
X = df_numeric.drop(['User_ID'], axis=1)
y = df_numeric['User_ID']
k = 10  # Number of top features to select

selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(X, y)

# Get the indices of the selected features
feature_indices = selector.get_support(indices=True)

# Get the names of the selected features
selected_features = X.columns[feature_indices]

# Use the selected features for user similarity calculation and recommendations
df_selected = df[['User_ID'] + list(selected_features)]

# Define the target user
target_user_id = 1

# Filter out the target user from the DataFrame
target_user = df_selected[df_selected['User_ID'] == target_user_id].iloc[0]

# Reshape the target user data to have a single sample
target_user_reshaped = target_user.drop(['User_ID']).values.reshape(1, -1)

# Calculate user similarity based on website usage metrics
user_similarity = cosine_similarity(df_selected.drop(['User_ID'], axis=1), target_user_reshaped)

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


if __name__ == '__main__':
    main()

