{"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"provenance":[],"authorship_tag":"ABX9TyNdDJnD2r/tIv+zx9zcVa9n"},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},"cells":[{"cell_type":"code","execution_count":null,"metadata":{"id":"T01lGD5RAOO6"},"outputs":[],"source":["import streamlit as st\n","import pandas as pd\n","\n","# Configuration de la page\n","st.set_page_config(\n","    page_title=\"Détails du Film\",\n","    page_icon=\"🎥\",\n","    layout=\"wide\"\n",")\n","\n","# Charger les données des films\n","@st.cache_data\n","def load_data():\n","    return pd.read_csv(\"cinevasion/csv/films_acteurs.csv\")\n","\n","films = load_data()\n","\n","# Récupérer le code du film sélectionné depuis la session\n","selected_film_tconst = st.session_state.get(\"selected_film_tconst\", None)\n","\n","# Vérifier si un film a été sélectionné\n","if selected_film_tconst:\n","    # Récupérer les détails du film sélectionné\n","    selected_film = films[films[\"tconst\"] == selected_film_tconst]\n","    if not selected_film.empty:\n","        selected_film = selected_film.iloc[0]  # Extraire la première ligne\n","\n","        # Afficher les détails du film\n","        st.title(selected_film[\"title\"])\n","        st.image(selected_film[\"poster_path\"], use_column_width=True)\n","        st.subheader(f\"Genres : {selected_film['genres']}\")\n","        st.write(f\"Date de sortie : {selected_film['release_date']}\")\n","        st.write(f\"Note moyenne : {selected_film['averageRating']} ({selected_film['numVotes']} votes)\")\n","        st.write(f\"Résumé : {selected_film['overview']}\")\n","\n","        # Bouton pour voir des films similaires\n","        if st.button(\"Voir des films similaires\"):\n","            st.session_state[\"similarity_film_title\"] = selected_film[\"title\"]\n","            st.experimental_set_query_params(page=\"similarity_page\")\n","    else:\n","        st.warning(\"Aucun détail disponible pour ce film.\")\n","else:\n","    st.warning(\"Aucun film sélectionné. Retournez à la page principale pour choisir un film.\")\n","\n","# Barre de navigation latérale\n","st.sidebar.title(\"Navigation\")\n","page = st.sidebar.radio(\"Pages\", [\"Home\", \"Recommendations\", \"Similarity Page\"])\n","\n","if page == \"Home\":\n","    st.experimental_set_query_params(page=\"home_page\")\n","elif page == \"Recommendations\":\n","    st.experimental_set_query_params(page=\"recommendations_page\")\n","elif page == \"Similarity Page\":\n","    st.experimental_set_query_params(page=\"similarity_page\")"]}]}