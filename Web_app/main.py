from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your dataset
df = pd.read_csv('output.csv')

def recommend_products_content_based(df, user_id_encoded):
    tfidf = TfidfVectorizer(stop_words='english')
    df['combined_features'] = df['combined_features'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    user_history = df[df['user_id'] == user_id_encoded]

    if not user_history.empty:
        indices = user_history.index.tolist()
        cosine_sim_user = cosine_similarity(tfidf_matrix[indices], tfidf_matrix)
        flat_cosine_sim = cosine_sim_user.flatten()
        top_indices = sorted(((i, sim) for i, sim in enumerate(flat_cosine_sim) if i not in indices),
                             key=lambda x: x[1], reverse=True)
        top_products = top_indices[:5]
        recommended_products = df.iloc[[i[0] for i in top_products]]
        results_df = pd.DataFrame({
            'Product_ID': recommended_products['product_id'].tolist(),
            'Recommended_Product': recommended_products['product_name'].tolist(),
            'Score_Recommendation': [i[1] for i in top_products]
        })
        return results_df
    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    error = None

    if request.method == 'POST':
        user_id = request.form.get('user_id', type=int)
        if user_id is None:
            error = "User ID is required"
        else:
            recommendations = recommend_products_content_based(df, user_id)
            if recommendations is None:
                error = "No recommendations found"

    return render_template('index.html', recommendations=recommendations, error=error)

if __name__ == '__main__':
    app.run(debug=True)
