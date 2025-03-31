import streamlit as st
# Configuration de la page Streamlit avec un titre et une icône
st.set_page_config(
    page_title="CINEVASION",
    page_icon="🎬",
    layout="wide"
)

import pandas as pd
from utils.utils import load_css, search_movies, get_recommendations, prepare_features, check_authentication, handle_signup, handle_login, handle_logout
from chatbot.chatbot import MovieChatbot


# Chargement du fichier CSS pour le style de l'application
load_css('css/style.css')

@st.cache_data
def load_data():
    try:
        films = pd.read_csv("csv/films_def.csv")
        intervenants = pd.read_csv("csv/intervenants_def.csv")
        lien = pd.read_csv("csv/lien_def.csv")
        return films, intervenants, lien
    except Exception as e:
        st.error(f"Erreur de chargement des données: {str(e)}")
        st.stop()
films, intervenants, lien = load_data()

# Tri des films par popularité décroissante
films = films.sort_values(by='popularity', ascending=False)

# Préparation des filtres
films['decade'] = pd.to_datetime(films['release_date']).dt.year // 10 * 10
min_decade = films['decade'].min() // 10 * 10
max_decade = films['decade'].max() // 10 * 10
films['averageRating_rounded'] = films['averageRating'].round()

# Extraction des genres uniques
unique_genres = sorted(list({
    genre.strip() 
    for genres in films['genres'].dropna() 
    for genre in genres.split(',')
}))

# Barre latérale pour l'inscription utilisateur
with st.sidebar:
    st.title("Utilisateur")
    if check_authentication():
        st.write(f"Connecté : {st.session_state['user']['email']}")
        if st.button("Se déconnecter"):
            handle_logout()
    else:
        st.write("Non connecté")
        auth_mode = st.radio("Action :", ["Se connecter", "S'inscrire"])
        email = st.text_input("Email", placeholder="email@example.com")
        password = st.text_input("Mot de passe", type="password", placeholder="Mot de passe")

        if auth_mode == "S'inscrire":
            if st.button("S'inscrire"):
                handle_signup(email, password)
        elif auth_mode == "Se connecter":
            if st.button("Se connecter"):
                handle_login(email, password)

# Titre principal de l'application avec un effet néon
st.markdown(
    '<div class="title-container"><h1 class="neon-title">CINEVASION 🎬</h1></div>', 
    unsafe_allow_html=True
)

# Zone de recherche de films
search_container = st.container()
with search_container:
    col1, col2 = st.columns(2)

    with col1:
        # Recherche de films par nom
        st.markdown('<p class="search-label">Recommander par le nom d\'un film 🔍</p>', unsafe_allow_html=True)
        film_input = st.text_input(
            "",
            placeholder="Rechercher un film...",
            label_visibility="collapsed",
            key="film_search"
        )
        if st.button("✨ Recommander", key="search_film_btn"):
            if film_input:
                selected_film = films[films['title'].str.contains(film_input, case=False, na=False)]
                if not selected_film.empty:
                    features_df = prepare_features(films)
                    recommendations = get_recommendations(selected_film.iloc[0]['title'], films, features_df)
                    st.session_state['recommendations'] = recommendations
                else:
                    st.warning("Aucun film trouvé.")

    with col2:
        # Recherche de films par mot-clé ou nom d'acteur
        st.markdown('<p class="search-label">Rechercher par mot-clé ou nom d\'acteur 🔍</p>', unsafe_allow_html=True)
        keyword_input = st.text_input(
            "",
            placeholder="Entrez un mot-clé ou nom d'acteur...",
            key="keyword",
            label_visibility="collapsed"
        )
        if st.button("✨ Rechercher", key="search_keyword_btn"):
            if keyword_input:
                search_results = search_movies(keyword_input, films, intervenants, lien)
                if not search_results.empty:
                    st.session_state['search_results'] = search_results
                    st.rerun()

# Affichage des résultats de la recherche ou des recommandations
results_container = st.container()
if 'search_results' in st.session_state:
    with results_container:
        st.markdown("### Résultats de la recherche")
        cols = st.columns(5)
        search_results = st.session_state['search_results']
        for i in range(5):
            with cols[i]:
                if i < len(search_results):
                    film = search_results.iloc[i]
                    st.image(film['poster_path'], use_container_width=True)
                    if st.button("✨ Détails", key=f"search_{film['tconst']}"):
                        st.session_state['selected_film_tconst'] = film['tconst']
                        st.session_state['go_to_details'] = True
                        st.rerun()

if 'recommendations' in st.session_state:
    with results_container:
        st.markdown("### Films recommandés")
        cols = st.columns(5)
        recommendations = st.session_state['recommendations']
        for i in range(5):
            with cols[i]:
                if i < len(recommendations):
                    film = recommendations.iloc[i]
                    st.image(film['poster_path'], use_container_width=True)
                    if st.button("✨ Détails", key=f"recommendation_{film['tconst']}"):
                        st.session_state['selected_film_tconst'] = film['tconst']
                        st.session_state['go_to_details'] = True
                        st.rerun()

st.markdown("---")

# Filtres pour affiner la recherche
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<p class="filter-label">Décennie 📅</p>', unsafe_allow_html=True)
    decades = range(min_decade, max_decade + 10, 10)
    decade_options = [f"{decade}s" for decade in decades]
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

# Application des filtres sur la liste des films
try:
    filtered_films = films.copy()

    if selected_decade:
        filtered_films = filtered_films[filtered_films['decade'] == selected_decade_start]
    if genre:
        filtered_films = filtered_films[filtered_films['genres'].str.contains(genre, na=False, case=False)]
    if country:
        filtered_films = filtered_films[filtered_films['origin_country'] == country]
    if selected_rating:
        filtered_films = filtered_films[filtered_films['averageRating_rounded'] == selected_rating]

    if filtered_films.empty:
        st.warning("Aucun film ne correspond aux filtres sélectionnés.")
    else:
        first_films = filtered_films.head(4)

    rows = [first_films.iloc[i:i + 2] for i in range(0, len(first_films), 2)]

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

# Redirection vers la page des détails si un film est sélectionné
if st.session_state.get('go_to_details', False):
    st.session_state['go_to_details'] = False
    st.switch_page("pages/details_page.py")

# Initialisation et affichage du chatbot
chat = MovieChatbot()
chat.display()