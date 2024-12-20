import streamlit as st
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
import os
from utils.utils import load_css, prepare_features, get_recommendations, load_files

# 📘 Configuration de la page Streamlit
st.set_page_config(
   page_title="Cinevasion - Recommandations",  # Titre de la page affiché dans l'onglet du navigateur
   page_icon="🎬",  # Icône de la page
   layout="wide"  # Disposition de la page en mode "large"
)

# 📘 Initialisation de l'état de session pour le suivi des actions de l'utilisateur
if "button_state" not in st.session_state:
    st.session_state.button_state = None  # État par défaut du bouton



# 📘 Application du CSS
load_css('css/style.css')



# 📘 Chargement des données
films_def = load_files()

# 📘 Navigation de la page
st.page_link("home_page.py", label="Retour à l'accueil")



# 📘 Interface utilisateur
st.title("Recommandations de films")

# 📘 Préparation des caractéristiques des films
features_matrix = prepare_features(films_def)

# 📘 Sélection du film par l'utilisateur
if "search_film" in st.session_state:
    film_choice = st.session_state["search_film"]
else:
    film_choice = st.selectbox("Choisissez un film :", films_def['title'].unique())

try:
    # 📘 Obtenir les recommandations de films
    recommended_films = get_recommendations(
        film_choice, 
        films_def, 
        features_matrix,
        n_recommendations=5
    )
    
    # 📘 Affichage des informations du film sélectionné
    selected_film = films_def[films_def['title'] == film_choice].iloc[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(selected_film['poster_path'], use_container_width=True)

    with col2:
        st.markdown(f"""
        <div class="neo-container">
            <h2>{selected_film['title']}</h2>
            <p><strong>Année :</strong> {pd.to_datetime(selected_film['release_date']).year}</p>
            <p><strong>Genres :</strong> {selected_film['genres']}</p>
            <p><strong>Note moyenne :</strong> {selected_film['averageRating']:.1f}/10</p>
            <p><strong>Description :</strong> {selected_film.get('overview', 'Description non disponible')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommendations-container">
        <h2>Films recommandés</h2>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(5)
    
    for i, (_, film) in enumerate(recommended_films.iterrows()):
        with cols[i]:
            with st.container():
                st.image(film['poster_path'], use_container_width=True)             
                if st.button("Détails", key=f"details_{i}_{film['tconst']}"):
                    st.session_state.button_state = "action"
                    st.session_state['selected_film_tconst'] = film['tconst']
                    st.switch_page("pages/details_page.py")

except Exception as exc:
    st.error(f"Erreur lors de la génération des recommandations : {str(exc)}")