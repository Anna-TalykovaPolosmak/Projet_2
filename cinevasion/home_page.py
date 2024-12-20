import pandas as pd
import random
import streamlit as st
from streamlit_authenticator import Authenticate
from streamlit_option_menu import option_menu
import numpy as np
import os
from utils.utils import load_css

# 📘 Configuration de la page principale Streamlit
st.set_page_config(
    page_title="CINEVASION",  # Titre de la page affiché dans l'onglet du navigateur
    page_icon="🎬",  # Icône affichée dans l'onglet du navigateur
    layout="wide"  # Disposition de la page en mode "large"
)


# 📘 Chargement du fichier CSS
load_css('css/style.css')

# 📘 Lecture des données du fichier CSV contenant la base de données des films
films = pd.read_csv("csv/films_def.csv")
films = films.sort_values(by='popularity', ascending=False)

# 📘 Calcul de la décennie de sortie des films
films['decade'] = (films['release_date'].str[:4].astype(int) // 10) * 10
min_decade = (films['decade'].min() // 10) * 10  # Décennie la plus ancienne
max_decade = (films['decade'].max() // 10) * 10  # Décennie la plus récente

# 📘 Arrondi de la note moyenne des films au nombre entier le plus proche
films['averageRating_rounded'] = films['averageRating'].round()

# 📘 Extraction des genres de films uniques
unique_genres = set()
for genres in films['genres'].dropna():  # Parcours des genres de films (en ignorant les valeurs manquantes)
    for genre in genres.split(','):  # Séparation des genres multiples par ","
        unique_genres.add(genre.strip())  # Ajout du genre à l'ensemble des genres uniques
unique_genres = sorted(list(unique_genres))  # Tri des genres par ordre alphabétique

# 📘 Barre latérale (sidebar) pour la connexion utilisateur
with st.sidebar:
    st.markdown('<div class="neo-container">', unsafe_allow_html=True)
    st.title("Inscription utilisateur")  # Titre de la section de connexion
    login = st.text_input("Nom d'utilisateur", placeholder="LOGIN")  # Champ de saisie du nom d'utilisateur
    password = st.text_input("Mot de passe", type="password", placeholder="PASSWORD")  # Champ de saisie du mot de passe
    st.markdown('</div>', unsafe_allow_html=True)

# 📘 Titre de la page principale
st.title("CINEVASION")

# 📘 Section de recherche de films
with st.container():
    film_input = st.text_input("Entrez le nom d'un film", placeholder="Rechercher un film...")  # Champ de recherche de film
    if st.button("Rechercher un film"):  # Bouton de recherche de film
        if film_input:  # Vérifie si l'utilisateur a saisi un nom de film
            # Recherche du film correspondant dans le DataFrame
            selected_film = films[films['title'].str.contains(film_input, case=False, na=False)]
            
            if not selected_film.empty:  # Si un ou plusieurs films correspondent
                selected_film_row = selected_film.iloc[0]  # Prend le premier film correspondant
                st.session_state['selected_film_tconst'] = selected_film_row['tconst']  # Stocke le code du film dans la session
                st.session_state['go_to_details'] = True  # Active la navigation vers la page de détails
                st.switch_page("pages/details_page.py")  # Redirige vers la page de détails du film
            else:
                st.warning("Aucun film ne correspond à votre recherche.")  # Message si aucun film n'est trouvé
        else:
            st.warning("Veuillez entrer le nom d'un film.")  # Message d'erreur si l'utilisateur n'a pas saisi de film

# 📘 Filtres de recherche pour les films
st.subheader("Filtres")  # Sous-titre de la section des filtres
col1, col2, col3, col4 = st.columns(4)  # Création de 4 colonnes pour afficher les filtres

# 📘 Filtre par décennie
with col1:
    decades = range(min_decade, max_decade + 10, 10)  # Liste des décennies disponibles
    decade_options = [f"{decade}s" for decade in decades]  # Formatage des décennies (ex: "1990s")
    selected_decade = st.selectbox("Décennie", [''] + decade_options)  # Menu déroulant des décennies
    selected_decade_start = int(selected_decade[:-1]) if selected_decade else None  # Extraction de la décennie choisie

# 📘 Filtre par genre
with col2:
    genre = st.selectbox("Genre", [''] + unique_genres)  # Menu déroulant des genres

# 📘 Filtre par pays d'origine
with col3:
    country = st.selectbox("Pays d'origine", [''] + list(films['origin_country'].unique()))  # Liste des pays d'origine des films

# 📘 Filtre par note
with col4:
    rating_options = list(range(5, 11))  # Liste des notes de 5 à 10
    selected_rating = st.selectbox("Note", [''] + rating_options)  # Menu déroulant des notes

# 📘 Application des filtres de recherche
try:
    filtered_films = films.copy()  # Copie du DataFrame d'origine
    if selected_decade:  # Filtre par décennie
        filtered_films = filtered_films[filtered_films['decade'] == selected_decade_start]
    if genre:  # Filtre par genre
        filtered_films = filtered_films[filtered_films['genres'].str.contains(genre, na=False)]
    if country:  # Filtre par pays d'origine
        filtered_films = filtered_films[filtered_films['origin_country'] == country]
    if selected_rating:  # Filtre par note
        filtered_films = filtered_films[filtered_films['averageRating_rounded'] == selected_rating]

    # 📘 Affichage des résultats de la recherche

    if filtered_films.empty:  # Si aucun film ne correspond aux filtres
        st.warning("Aucun film ne correspond aux filtres sélectionnés.")
    else:
        first_films = filtered_films.head(5)  # Sélectionne les 5 premiers films du DataFrame filtré
        
        cols = st.columns(len(first_films))  # Création d'autant de colonnes que de films sélectionnés
        for i, (p, row) in enumerate(first_films.iterrows()):  # Parcours des films sélectionnés
            with cols[i]:  # Affiche le film dans la colonne correspondante
                movie_container = st.container()
                with movie_container:
                    st.image(row['poster_path'], use_container_width=True)  # Affiche l'affiche du film
                if st.button("Détails", key=f"btn_details_{i}_"):  # Bouton "Détails" pour chaque film
                    st.session_state['selected_film_tconst'] = row['tconst']  # Stocke le code du film dans la session
                    st.session_state['go_to_details'] = True  # Active la navigation vers la page de détails
                    st.switch_page("pages/details_page.py")  # Redirige vers la page de détails du film

except Exception as e:
    # 📘 Gestion des erreurs
    st.error(f"Erreur lors de l'application des filtres : {str(e)}")  # Affiche un message d'erreur en cas de problème