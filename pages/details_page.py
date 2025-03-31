import streamlit as st
# Configuration de la page
st.set_page_config(
    page_title="Cinevasion - Détails",
    page_icon="🎬",
    layout="wide"
)
import pandas as pd
from utils.utils import load_css, get_recommendations, prepare_features, check_authentication, handle_logout, handle_signup, handle_login
from chatbot.chatbot import MovieChatbot

# Chargement du fichier CSS pour le style de l'application
load_css('css/style.css')

# Chargement des ressources
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

# Navigation
st.page_link("home_page.py", label="🏠 Retour à l'accueil")

if 'selected_film_tconst' not in st.session_state:
    st.warning("⚠️ Aucun film sélectionné")
    st.page_link("home_page.py", label="🏠 Retour à l'accueil")
    st.stop()

# Barre latérale pour l'inscription utilisateur
with st.sidebar:
    st.title("Utilisateur")
    if check_authentication():
        # Si l'utilisateur est connecté
        st.write(f"Connecté : {st.session_state['user']['email']}")
        if st.button("Se déconnecter"):
            handle_logout()
    else:
        # Si l'utilisateur n'est pas connecté
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

try:
    tconst = st.session_state['selected_film_tconst']
    film_data = films[films['tconst'] == tconst]
    
    if film_data.empty:
        st.error("❌ Film non trouvé")
        st.stop()
        
    selected_film = film_data.iloc[0]

    # Container principal des informations du film
    with st.container():
        st.markdown('<div class="neo-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if pd.notna(selected_film['poster_path']):
                st.image(selected_film['poster_path'], width=300)

        with col2:
            release_date = pd.to_datetime(selected_film['release_date']).strftime('%Y') if pd.notna(selected_film['release_date']) else 'Non disponible'
            
            st.markdown(f"""
                <div class="movie-info">
                    <h1>{selected_film['title']}</h1>
                    <p><strong>Année :</strong> {release_date}</p>
                    <p><strong>Genres :</strong> {selected_film['genres']}</p>
                    <p><strong>Note moyenne :</strong> {float(selected_film['averageRating']):.1f}/10</p>
                    <h3>Synopsis</h3><p>{selected_film['overview']}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Réalisateurs
    film_participants = lien[lien['tconst'] == selected_film['tconst']]
    directors = film_participants[film_participants['category'] == 'director']
    
    if not directors.empty:
        st.markdown('<div class="neo-container directors-section">', unsafe_allow_html=True)
        st.markdown("<h3>🎥 Réalisateurs</h3>", unsafe_allow_html=True)
        
        cols = st.columns(len(directors))
        for i, (_, director_link) in enumerate(directors.iterrows()):
            director = intervenants[intervenants['nconst'] == director_link['nconst']]
            if not director.empty:
                with cols[i]:
                    director = director.iloc[0]
                    profile_path = director['profile_path'] if pd.notna(director['profile_path']) else "https://via.placeholder.com/150"
                    st.markdown(f"""
                        <div class="person-card">
                            <img src="{profile_path}" alt="{director['primaryName']}">
                            <h4>{director['primaryName']}</h4>
                        </div>
                    """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Acteurs principaux
    actors = film_participants[film_participants['category'] == 'actor'].head(5)
    if not actors.empty:
        st.markdown('<div class="neo-container actors-section">', unsafe_allow_html=True)
        st.markdown("<h3>🎭 Acteurs principaux</h3>", unsafe_allow_html=True)
        
        cols = st.columns(len(actors))
        for i, (_, actor_link) in enumerate(actors.iterrows()):
            actor = intervenants[intervenants['nconst'] == actor_link['nconst']]
            if not actor.empty:
                with cols[i]:
                    actor = actor.iloc[0]
                    profile_path = actor['profile_path'] if pd.notna(actor['profile_path']) else "https://via.placeholder.com/150"
                    st.markdown(f"""
                        <div class="person-card">
                            <img src="{profile_path}" alt="{actor['primaryName']}">
                            <h4>{actor['primaryName']}</h4>
                        </div>
                    """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Bande annonce
    if pd.notna(selected_film.get('trailer_link')):
        st.markdown('<div class="neo-container trailer-section">', unsafe_allow_html=True)
        st.markdown("<h3>🎬 Bande annonce</h3>", unsafe_allow_html=True)
        st.video(selected_film['trailer_link'])
        st.markdown('</div>', unsafe_allow_html=True)
 
    # Films similaires
    st.markdown('<div class="neo-container similar-movies">', unsafe_allow_html=True)
    st.markdown("### 🎬 Films similaires")
    features_matrix = prepare_features(films)
    film_choice = selected_film['title']

    try:
        recommended_films = get_recommendations(
            film_choice, 
            films, 
            features_matrix,
            n_recommendations=5
        )

        cols = st.columns(5)
        for i, (_, film) in enumerate(recommended_films.iterrows()):
            with cols[i]:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                st.image(film['poster_path'], use_container_width=True)
                if st.button("✨ Détails", key=f"details_{i}_{film['tconst']}"):
                    st.session_state['selected_film_tconst'] = film['tconst']
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    except Exception as exc:
        st.error(f"❌ Erreur de recommandations: {str(exc)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Une erreur est survenue: {str(e)}")
    st.session_state['go_to_details'] = False
    st.switch_page("home_page.py")

# Chatbot
chat = MovieChatbot()
chat.display()