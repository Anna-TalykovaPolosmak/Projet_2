import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def load_css(css_file):
    """
    Charge et applique un fichier CSS externe pour styliser l'application Streamlit.
    Args:
        css_file (str): Chemin du fichier CSS.
    """
    if os.path.exists(css_file):
        with open(css_file, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.error(f"Fichier CSS non trouvé: {css_file}")

def load_files(files='films'):
    """
    Charge les fichiers de données depuis des fichiers CSV.
    Cette fonction permet de charger un ou plusieurs fichiers CSV contenant des données nécessaires à l'application.
    Elle gère les erreurs si le fichier est introuvable ou si le chemin est incorrect.
    Args:
        files (str ou list):
            - Si c'est une chaîne de caractères (str), elle correspond au nom de base du fichier CSV à charger.
            - Si c'est une liste (list), elle contient plusieurs noms de base de fichiers CSV à charger.
    Returns:
        pd.DataFrame ou list:
            - Si un seul fichier est chargé, retourne un DataFrame Pandas.
            - Si plusieurs fichiers sont chargés, retourne une liste de DataFrames Pandas.
    Raises:
        FileNotFoundError: Si un ou plusieurs fichiers sont introuvables.
        ValueError: Si l'argument `files` n'est ni une chaîne ni une liste.
    """
    if isinstance(files, str):
        try:
            path = f'csv/{files}_def.csv'
            return pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Le fichier '{path}' est introuvable. Vérifiez le chemin.")
        except Exception as e:
            raise RuntimeError(f"Une erreur inattendue est survenue lors du chargement du fichier '{path}': {e}")
    elif isinstance(files, list):
        dataframes = []
        for file in files:
            try:
                path = f'csv/{file}_def.csv'
                df = pd.read_csv(path)
                dataframes.append(df)
            except FileNotFoundError:
                raise FileNotFoundError(f"Le fichier '{path}' est introuvable. Vérifiez le chemin.")
            except Exception as e:
                raise RuntimeError(f"Une erreur inattendue est survenue lors du chargement du fichier '{path}': {e}")
        return dataframes
    else:
        raise ValueError("L'argument 'files' doit être une chaîne (str) ou une liste de chaînes (list).")

def prepare_features(df):
    """
    Prépare les caractéristiques nécessaires au système de recommandation de films.
    Étapes du traitement :
    - Normalisation des colonnes numériques à l'aide de RobustScaler.
    - Encodage des genres de films sous forme de colonnes binaires.
    Args:
        df (pd.DataFrame): Le DataFrame des films d'origine.
    Returns:
        pd.DataFrame: DataFrame contenant les caractéristiques finales prêtes pour l'algorithme de recommandation.
    """
    X = df.copy()
    X['year'] = pd.to_datetime(X['release_date']).dt.year  # Extraction de l'année à partir de la date de sortie
    numeric_features = ['averageRating', 'popularity', 'year']
    # Normalisation des caractéristiques numériques
    scaler = RobustScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    # Encodage des genres en colonnes binaires
    genres_split = X['genres'].str.split(',').apply(lambda x: [genre.strip() for genre in x])
    genres_dummies = pd.get_dummies(genres_split.explode()).groupby(level=0).sum()
    # Concaténation des colonnes numériques et des genres encodés
    final_features = pd.concat([
        X[numeric_features],
        genres_dummies
    ], axis=1)
    return final_features

def get_recommendations(title, df, features_df, n_recommendations=5, genre_weight=10):
    try:
        movie_index = df[df['title'] == title].index[0]
        genre_columns = [col for col in features_df.columns if col not in ['averageRating', 'popularity', 'year']]
        
        weighted_features = features_df.copy()
        for genre in genre_columns:
            if features_df.iloc[movie_index][genre] == 1:
                weighted_features[genre] = weighted_features[genre] * genre_weight
        
        model = NearestNeighbors(n_neighbors=n_recommendations + 1, metric='euclidean')
        model.fit(weighted_features)
        
        distances, indices = model.kneighbors(weighted_features.iloc[movie_index:movie_index+1])
        
        return df.iloc[indices[0][1:]]
    except Exception as e:
        st.error(f"Erreur dans get_recommendations: {e}")
        return pd.DataFrame()


def format_movie_info(movie):
    try:
        return f"""### 🎬 {movie['title']}
**⭐ Note:** {movie['averageRating']}/10
**📅 Année:** {movie['release_date'][:4] if pd.notna(movie['release_date']) else 'Non disponible'}
**🎭 Genre:** {movie['genres'] if pd.notna(movie['genres']) else 'Non spécifié'}

#### 📝 Synopsis
{movie.get('overview', 'Synopsis non disponible')}
"""
    except Exception as e:
        st.error(f"Erreur dans format_movie_info: {e}")
        return "Information non disponible"

@st.cache_data(ttl=600)  # Mise en cache des résultats de recherche pendant 10 minutes
def search_movies(query, films_df, intervenants_df, lien_df, n_recommendations=5):
    """
    Fonction optimisée de recherche de films par mot-clé ou nom d'acteur.
    
    Args:
        query (str): Requête de recherche
        films_df (pd.DataFrame): DataFrame avec les films
        intervenants_df (pd.DataFrame): DataFrame avec les acteurs et réalisateurs
        lien_df (pd.DataFrame): DataFrame reliant les films et les personnes
        n_recommendations (int): Nombre de résultats à retourner
    
    Returns:
        pd.DataFrame: Films trouvés
    """
    try:
        # Vérification préliminaire d'une requête vide
        if not query or query.strip() == "":
            return films_df.head(n_recommendations)

        query = query.lower()  # Conversion de la requête en minuscules une seule fois
        
        # Recherche rapide par acteurs - utilise l'indexation pour la vitesse
        actor_matches = intervenants_df[intervenants_df['primaryName'].str.lower().str.contains(query, na=False)]
        
        if not actor_matches.empty:
            # Prenons seulement le premier acteur trouvé
            actor_nconst = actor_matches.iloc[0]['nconst']
            
            # Récupération des films avec cet acteur
            actor_movies = lien_df[lien_df['nconst'] == actor_nconst]
            matched_films = films_df[films_df['tconst'].isin(actor_movies['tconst'])]
            
            if not matched_films.empty:
                return matched_films.head(n_recommendations)
        
        # Création optimisée du masque pour la recherche
        # Recherche d'abord uniquement dans les champs les plus importants pour accélérer
        essential_mask = (
            films_df['title'].str.lower().str.contains(query, na=False) |
            films_df['genres'].str.lower().str.contains(query, na=False)
        )
        
        essential_matches = films_df[essential_mask]
        
        if not essential_matches.empty:
            return essential_matches.head(n_recommendations)
        
        # Recherche étendue seulement si nécessaire
        extended_mask = (
            films_df['overview'].str.lower().str.contains(query, na=False) |
            films_df['keywords'].str.lower().str.contains(query, na=False) |
            films_df['tagline'].str.lower().str.contains(query, na=False) |
            films_df['origin_country'].str.lower().str.contains(query, na=False)
        )
        
        extended_matches = films_df[extended_mask]
        
        if not extended_matches.empty:
            return extended_matches.head(n_recommendations)
            
        # TF-IDF utilisé uniquement en dernier recours, lorsque la recherche directe n'a donné aucun résultat
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Création du texte de recherche uniquement pour un sous-ensemble de champs importants
        # et avec une multiplication moindre pour overview
        films_df['search_text'] = (
            films_df['title'].fillna('') + ' ' +
            films_df['overview'].fillna('') + ' ' +
            films_df['keywords'].fillna('') + ' ' +
            films_df['genres'].fillna('')
        )
        
        # Nombre réduit de max_features pour accélérer
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=2000,  # Réduit de 5000
            ngram_range=(1, 1)  # Utilisation uniquement des unigrammes pour la vitesse
        )
        
        tfidf_matrix = tfidf.fit_transform(films_df['search_text'])
        query_vec = tfidf.transform([query])
        similarity = cosine_similarity(query_vec, tfidf_matrix)
        top_indices = similarity[0].argsort()[-n_recommendations:][::-1]
        
        return films_df.iloc[top_indices]
        
    except Exception as e:
        st.error(f"Erreur dans search_movies: {str(e)}")
        return films_df.head(n_recommendations)  # Retour des films populaires en cas d'erreur
    
from supabase import create_client, Client

url = "https://ztihcbkzolqvmmiylcnn.supabase.co"
key= "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp0aWhjYmt6b2xxdm1taXlsY25uIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQ5Njk1NjAsImV4cCI6MjA1MDU0NTU2MH0.z0lQr5IBN6-wMCJROEYLfQOHvn0WVPzsPZut4IWO-Sw"
supabase: Client = create_client(url, key)


def check_authentication():
    """Vérifie si l'utilisateur est connecté."""
    return "user" in st.session_state

def handle_signup(email, password):
    """Gestion de l'inscription."""
    if not email or not password:
        st.error("Veuillez remplir tous les champs pour vous inscrire.")
        return
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        if response.user:
            user_data = {"id": response.user.id, "email": email}
            try:
                supabase.table("users").insert(user_data).execute()
                st.success("Inscription réussie ! Vous pouvez maintenant vous connecter.")
            except Exception as e:
                st.error(f"Erreur lors de l'ajout à la base de données : {str(e)}")
        else:
            st.error("Erreur lors de l'inscription. Veuillez réessayer.")
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue : {str(e)}")

def handle_login(email, password):
    """Gestion de la connexion."""
    if not email or not password:
        st.error("Veuillez remplir tous les champs pour vous connecter.")
        return
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if response.user:
            st.session_state["user"] = {"id": response.user.id, "email": response.user.email}
            st.success(f"Bienvenue {response.user.email} !")
        else:
            st.error("Erreur lors de la connexion. Vérifiez vos identifiants.")
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue : {str(e)}")

def handle_logout():
    """Déconnexion de l'utilisateur."""
    if "user" in st.session_state:
        del st.session_state["user"]
    st.rerun()