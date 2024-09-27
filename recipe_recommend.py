import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load your dataset and limit it to the first 1000 rows for testing
data = pd.read_csv('recipes.csv')  # Adjust the filename accordingly
data = data.head(1000)  # Limit to first 1000 rows

# Handle missing values
data['ingredients'] = data['ingredients'].fillna('')
data['instructions'] = data['instructions'].fillna('')
9
# Combine ingredients and instructions into one feature
data['combined'] = data['ingredients'] + " " + data['instructions']

# Create TF-IDF features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(ingredients_input, cosine_sim=cosine_sim):
    # Create a TF-IDF vector for the user input
    input_tfidf = tfidf.transform([ingredients_input])
    # Calculate similarity with existing recipes
    sim_scores = cosine_similarity(input_tfidf, tfidf_matrix)
    sim_scores = sim_scores.flatten()  # Flatten to 1D array
    sim_scores_indices = sim_scores.argsort()[-5:][::-1]  # Get indices of top 5 scores
    return data.iloc[sim_scores_indices]

# Streamlit UI
st.title("Recipe Generator")

# Create a text input for ingredients
ingredients_input = st.text_area("Enter ingredients (comma separated):")

# Get recommendations when the button is pressed
if st.button("Generate Recipe") and ingredients_input:
    recommendations = get_recommendations(ingredients_input)

    if not recommendations.empty:
        st.subheader("Recommended Recipes")
        for index, row in recommendations.iterrows():
            st.write(f"**{row['title']}**")
            st.write(f"**Ingredients:** {row['ingredients']}")
            st.write(f"**Instructions:** {row['instructions']}")
            
            st.write("---")
    else:
        st.write("No recipes found that match the ingredients.")
else:
    st.write("Please enter ingredients to get recommendations.")
