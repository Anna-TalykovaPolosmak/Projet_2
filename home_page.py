import streamlit as st
# Configuration de la page Streamlit avec un titre et une icône
st.set_page_config(
    page_title="CINEVASION",
    page_icon="🎬",
    layout="wide"
)

import pandas as pd
from utils.utils import load_css, search_movies, check_authentication, handle_signup, handle_login, handle_logout
from chatbot.chatbot import MovieChatbot


# Chargement du fichier CSS pour le style de l'application
load_css('css/style.css')

@st.cache_data(ttl=3600)  # Mise en cache pour 1 heure
def load_data():
    try:
        films = pd.read_csv("csv/films_def.csv")
        intervenants = pd.read_csv("csv/intervenants_def.csv")
        lien = pd.read_csv("csv/lien_def.csv")
        
        # Prétraitement des données pour accélération
        films = films.sort_values(by='popularity', ascending=False)
        films['decade'] = pd.to_datetime(films['release_date']).dt.year // 10 * 10
        films['averageRating_rounded'] = films['averageRating'].round()
        
        return films, intervenants, lien
    except Exception as e:
        st.error(f"Erreur de chargement des données: {str(e)}")
        st.stop()

# Chargement et prétraitement des données
films, intervenants, lien = load_data()

# Obtention des genres uniques - mise en cache pour accélération
@st.cache_data(ttl=3600)
def get_unique_genres(films_df):
    return sorted(list({
        genre.strip() 
        for genres in films_df['genres'].dropna() 
        for genre in genres.split(',')
    }))

# Obtention de la liste des décennies - mise en cache pour accélération
@st.cache_data(ttl=3600)
def get_decades(films_df):
    min_decade = films_df['decade'].min() // 10 * 10
    max_decade = films_df['decade'].max() // 10 * 10
    return range(min_decade, max_decade + 10, 10)

# Obtention de toutes les valeurs uniques
unique_genres = get_unique_genres(films)
decades = get_decades(films)
decade_options = [f"{decade}s" for decade in decades]

# Barre latérale pour l'authentification
with st.sidebar:
    st.title("Utilisateur")
    st.info("L'authentification est désactivée dans cette version de l'application.")

# Titre de l'application
st.markdown(
    '<div class="title-container"><h1 class="neon-title">CINEVASION 🎬</h1></div>', 
    unsafe_allow_html=True
)

# CSS supplémentaire pour centrer la recherche
st.markdown("""
<style>
.search-container {
    max-width: 600px;
    margin: 0 auto 30px auto;
    text-align: center;
}
.search-container p {
    text-align: center;
    font-size: 1.2em;
    margin-bottom: 15px;
    color: white;
}
.search-container button {
    margin: 15px auto 0 auto;
    display: block !important;
    max-width: 200px;
}
</style>
""", unsafe_allow_html=True)

# Recherche - centrée à l'aide du CSS
st.markdown('<div class="search-container">', unsafe_allow_html=True)
st.markdown('<p class="search-label">Rechercher par mot-clé ou nom d\'acteur 🔍</p>', unsafe_allow_html=True)
keyword_input = st.text_input(
    "",
    placeholder="Entrez un mot-clé ou nom d'acteur...",
    key="keyword",
    label_visibility="collapsed"
)
search_button = st.button("✨ Rechercher", key="search_keyword_btn")
st.markdown('</div>', unsafe_allow_html=True)

# Traitement de la recherche
if search_button:
    if keyword_input:
        with st.spinner('Recherche en cours...'):  # Afficher l'indicateur de chargement
            search_results = search_movies(keyword_input, films, intervenants, lien)
            if not search_results.empty:
                st.session_state['search_results'] = search_results
                st.rerun()
            else:
                st.warning("Aucun film trouvé pour cette recherche.")

# Affichage des résultats de recherche
if 'search_results' in st.session_state:
    with st.container():
        st.markdown("### Résultats de la recherche")
        cols = st.columns(5)
        search_results = st.session_state['search_results']
        for i in range(min(5, len(search_results))):  # Protection contre le dépassement de liste
            with cols[i]:
                film = search_results.iloc[i]
                st.image(film['poster_path'], use_container_width=True)
                if st.button("✨ Détails", key=f"search_{film['tconst']}"):
                    st.session_state['selected_film_tconst'] = film['tconst']
                    st.session_state['go_to_details'] = True
                    st.rerun()

st.markdown("---")

# Filtres pour affiner la recherche
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<p class="filter-label">Décennie 📅</p>', unsafe_allow_html=True)
    selected_decade = st.selectbox(
        "",
        [''] + decade_options,
        label_visibility="collapsed",
        key="decade_filter"
    )
    selected_decade_start = int(selected_decade[:-1]) if selected_decade else None

with col2:
    st.markdown('<p class="filter-label">Genre 🎭</p>', unsafe_allow_html=True)
    genre = st.selectbox(
        "",
        [''] + unique_genres,
        label_visibility="collapsed",
        key="genre_filter"
    )

with col3:
    st.markdown('<p class="filter-label">Pays d\'origine 🌍</p>', unsafe_allow_html=True)
    country = st.selectbox(
        "",
        [''] + list(films['origin_country'].unique()),
        label_visibility="collapsed",
        key="country_filter"
    )

with col4:
    st.markdown('<p class="filter-label">Note ⭐</p>', unsafe_allow_html=True)
    rating_options = list(range(5, 11))
    selected_rating = st.selectbox(
        "",
        [''] + rating_options,
        label_visibility="collapsed",
        key="rating_filter"
    )

# Filtrage optimisé
@st.cache_data(ttl=300, max_entries=20)  # Cache pour les résultats de filtrage pendant 5 minutes
def filter_movies(films_df, decade=None, genre=None, country=None, rating=None):
    filtered = films_df.copy()
    
    if decade:
        filtered = filtered[filtered['decade'] == decade]
    if genre:
        filtered = filtered[filtered['genres'].str.contains(genre, na=False, case=False)]
    if country:
        filtered = filtered[filtered['origin_country'] == country]
    if rating:
        filtered = filtered[filtered['averageRating_rounded'] == rating]
        
    return filtered.head(4)  # Ne retourner que les 4 premiers films

# Application des filtres
try:
    if selected_decade_start or genre or country or selected_rating:
        with st.spinner('Filtration en cours...'):
            filtered_films = filter_movies(
                films, 
                decade=selected_decade_start, 
                genre=genre, 
                country=country, 
                rating=selected_rating
            )
            
            if filtered_films.empty:
                st.warning("Aucun film ne correspond aux filtres sélectionnés.")
            else:
                rows = [filtered_films.iloc[i:i + 2] for i in range(0, len(filtered_films), 2)]
                
                for row in rows:
                    col1, col2 = st.columns(2)
                    for idx, film in row.iterrows():
                        with col1 if idx % 2 == 0 else col2:
                            with st.container():
                                poster_col, info_col = st.columns([1, 2])
                                
                                with poster_col:
                                    if pd.notna(film['poster_path']):
                                        st.image(film['poster_path'], width=150)
                                
                                with info_col:
                                    st.markdown(f"""
                                        <div class="movie-info">
                                            <h2 style='color: white; font-size: 24px; margin-bottom: 10px;'>{film['title']}</h2>
                                            <p style='color: #ff69b4; font-size: 18px;'>Année: {pd.to_datetime(film['release_date']).year}</p>
                                            <p style='color: #e0e0e0;'>Genre: {film['genres']}</p>
                                            <p style='color: #ffd700;'>⭐ {film['averageRating']:.1f}/10</p>
                                            <p style='color: #cccccc; margin-top: 10px;'>{film['overview'][:200] + '...' if pd.notna(film['overview']) and len(film['overview']) > 200 else film.get('overview', '')}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if st.button("✨ Détails", key=f"details_{film['tconst']}"):
                                        st.session_state['selected_film_tconst'] = film['tconst']
                                        st.session_state['go_to_details'] = True
                                        st.rerun()
                    
                    st.markdown("<hr style='margin: 30px 0; border: none; height: 1px; background-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)
    else:
        # Affichage des films populaires si aucun filtre n'est sélectionné
        popular_films = films.head(4)
        rows = [popular_films.iloc[i:i + 2] for i in range(0, len(popular_films), 2)]
        
        for row in rows:
            col1, col2 = st.columns(2)
            for idx, film in row.iterrows():
                with col1 if idx % 2 == 0 else col2:
                    with st.container():
                        poster_col, info_col = st.columns([1, 2])
                        
                        with poster_col:
                            if pd.notna(film['poster_path']):
                                st.image(film['poster_path'], width=150)
                        
                        with info_col:
                            st.markdown(f"""
                                <div class="movie-info">
                                    <h2 style='color: white; font-size: 24px; margin-bottom: 10px;'>{film['title']}</h2>
                                    <p style='color: #ff69b4; font-size: 18px;'>Année: {pd.to_datetime(film['release_date']).year}</p>
                                    <p style='color: #e0e0e0;'>Genre: {film['genres']}</p>
                                    <p style='color: #ffd700;'>⭐ {film['averageRating']:.1f}/10</p>
                                    <p style='color: #cccccc; margin-top: 10px;'>{film['overview'][:200] + '...' if pd.notna(film['overview']) and len(film['overview']) > 200 else film.get('overview', '')}</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if st.button("✨ Détails", key=f"details_{film['tconst']}"):
                                st.session_state['selected_film_tconst'] = film['tconst']
                                st.session_state['go_to_details'] = True
                                st.rerun()
            
            st.markdown("<hr style='margin: 30px 0; border: none; height: 1px; background-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Erreur lors de la filtration: {str(e)}")

# Redirection vers la page de détails
if st.session_state.get('go_to_details', False):
    st.session_state['go_to_details'] = False
    st.switch_page("pages/details_page.py")

# Initialisation et affichage du chatbot
chat = MovieChatbot()
chat.display()