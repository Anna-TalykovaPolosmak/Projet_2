import streamlit as st
from openai import OpenAI
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os

# Activation du support HTML et CSS pour le style
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    span[style*="color: pink"] {
        color: pink !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

class MovieChatbot:
    @staticmethod
    def initialize_session_state():
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "system",
                    "content": """Tu es CineBot, un assistant cinéma passionné 🎬. 
                    IMPORTANT: Tu DOIS formatter chaque titre de film en utilisant cette syntaxe Markdown: 
                    **<span style='color: pink'>TITRE DU FILM</span>**
                    
                    Tu donnes des réponses détaillées sur les films jusqu'à l'année 2000, acteurs et réalisateurs.
                    Tu as accès à une base de données de films, d'acteurs et de réalisateurs.
                    
                    Tu peux fournir des informations sur:
                    - Les films (titre, année, genre, note, synopsis)
                    - Les acteurs et réalisateurs (nom, films dans lesquels ils ont joué/qu'ils ont réalisés)
                    - Les recommandations de films similaires
                    
                    Pour chaque film mentionné, tu DOIS inclure:
                    ⭐ Note (si disponible)
                    📅 Année
                    🎭 Genre
                    📝 Synopsis
                    🎬 Acteurs principaux
                    🎥 Lien vers la bande-annonce (si disponible)
                    
                    Tu parles uniquement en français et utilises beaucoup des émojis appropriés."""
                },
                {
                    "role": "assistant",
                    "content": "Bonjour! 🎬 Je suis CineBot, votre assistant cinéma passionné! Comment puis-je vous aider aujourd'hui? 🍿"
                }
            ]

    def __init__(self):
        try:
            # Initialisation de l'état de session en premier
            self.initialize_session_state()
            
            # Initialisation d'OpenAI et des embeddings
            self.client = OpenAI(api_key=st.secrets["OpenAI_key"])
            self.embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OpenAI_key"])
            
            # Chargement des données principales
            self.films = pd.read_csv("csv/films_def.csv")
            self.intervenants = pd.read_csv("csv/intervenants_def.csv")
            self.lien = pd.read_csv("csv/lien_def.csv")
            
            # Création ou chargement du vectorstore
            self.vectorstore = self._create_or_load_vectorstore()
                
        except Exception as e:
            st.error(f"Erreur d'initialisation du chatbot: {str(e)}")

    def _create_or_load_vectorstore(self):
        persist_directory = "./chroma_db"
        
        # Vérification de l'existence du répertoire
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            
        try:
            # Tentative de chargement de la base existante
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            # Vérification du bon fonctionnement
            vectorstore.similarity_search("test", k=1)
            return vectorstore
            
        except Exception:
            # Création d'une nouvelle base si le chargement a échoué
            documents = self._prepare_movie_documents()
            vectorstore = Chroma.from_texts(
                texts=[doc["content"] for doc in documents],
                metadatas=[doc["metadata"] for doc in documents],
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            vectorstore.persist()
            return vectorstore
    
    def _prepare_movie_documents(self):
        documents = []
        for _, movie in self.films.iterrows():
            try:
                # Vérification de l'année du film
                year = int(movie['release_date'][:4]) if pd.notna(movie['release_date']) else 0
                
                # Ignore les films après 2000
                if year > 2000:
                    continue
                    
                # Formatage du titre du film
                title = f"**<span style='color: pink'>{movie['title']}</span>**"
                
                content = f"Titre: {title}\n"
                content += f"📅 Année: {year if year != 0 else 'Non disponible'}\n"
                content += f"🎭 Genre: {movie['genres'] if pd.notna(movie['genres']) else 'Non spécifié'}\n"
                content += f"⭐ Note: {movie['averageRating'] if pd.notna(movie['averageRating']) else 'Non disponible'}/10\n"
                content += f"📝 Synopsis: {movie['overview'] if pd.notna(movie['overview']) else 'Non disponible'}\n"
                
                # Récupération des acteurs pour le film
                movie_actors = self.lien[
                    (self.lien['tconst'] == movie['tconst']) & 
                    (self.lien['category'] == 'actor')
                ]
                actors = self.intervenants[
                    self.intervenants['nconst'].isin(movie_actors['nconst'])
                ]
                content += f"🎬 Acteurs: {', '.join(actors['primaryName'])}\n"
                
                # Ajout du lien vers la bande-annonce, s'il existe
                if pd.notna(movie['trailer_link']):
                    content += f"🎥 Bande-annonce: [Regarder le trailer]({movie['trailer_link']})\n"
                    # Ajout d'informations sur la langue de la bande-annonce, si disponible
                    if pd.notna(movie['langue_trailer']):
                        content += f"🌍 Langue du trailer: {movie['langue_trailer']}\n"
                
                # Ajout du tagline, s'il existe
                if pd.notna(movie['tagline']) and movie['tagline'] != '':
                    content += f"💫 Tagline: {movie['tagline']}\n"
                
                documents.append({
                    "content": content,
                    "metadata": {
                        "tconst": movie['tconst'],
                        "year": year,
                        "title": movie['title'],
                        "genres": movie['genres'] if pd.notna(movie['genres']) else '',
                        "rating": float(movie['averageRating']) if pd.notna(movie['averageRating']) else 0.0,
                        "trailer_link": movie['trailer_link'] if pd.notna(movie['trailer_link']) else '',
                        "langue_trailer": movie['langue_trailer'] if pd.notna(movie['langue_trailer']) else ''
                    }
                })
            except Exception as e:
                st.warning(f"Erreur lors de la préparation du document pour {movie['title']}: {str(e)}")
                continue
                
        return documents

    def get_response(self, user_input: str) -> str:
        try:
            # Obtention de l'embedding pour l'entrée utilisateur
            embedding_response = self.embeddings.embed_documents([user_input])[0]
            
            # Augmentation du nombre de films similaires pour de meilleurs résultats
            similar_movies = self.vectorstore.similarity_search_by_vector(
                embedding_response, 
                k=10,
                filter={"year": {"$lte": 2000}}  # Filtrage uniquement des films jusqu'à 2000
            )
            
            # Amélioration du format du contexte
            context = "Information sur les films disponibles:\n"
            for movie in similar_movies:
                context += f"\n---\n{movie.page_content}\n"
            
            # Mise à jour du prompt système
            system_prompt = """Tu es CineBot, un assistant cinéma passionné 🎬. 
            
            RÈGLES IMPORTANTES:
            1. Tu ne dois parler QUE des films d'avant 2000!
            2. Chaque titre de film DOIT être formaté ainsi: **<span style='color: pink'>TITRE DU FILM</span>**
            3. Pour CHAQUE film mentionné, tu DOIS inclure:
               - 📅 Année
               - 🎭 Genre
               - ⭐ Note /10
               - 📝 Synopsis
               - 🎬 Acteurs
               - 🎥 Lien de la bande-annonce (si disponible)
            4. Utilise TOUJOURS des émojis appropriés
            5. Si le film a un lien vers la bande-annonce, ajoute "🎥 Cliquez ici pour voir la bande-annonce!"
            6. Sois enthousiaste et passionné!
            
            Contexte des films disponibles:
            {context}
            """
            
            messages = [
                {"role": "system", "content": system_prompt.format(context=context)},
                {"role": "user", "content": user_input}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content

        except Exception as e:
            error_message = f"Désolé, une erreur s'est produite: {str(e)}"
            st.error(error_message)
            return "Je suis désolé, je ne peux pas répondre pour le moment. 😔"

    def display(self):
        try:
            with st.container():
                # Affichage de l'historique des messages
                for message in st.session_state.messages:
                    if message["role"] != "system":
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"], unsafe_allow_html=True)

                # Traitement de la nouvelle entrée utilisateur
                if prompt := st.chat_input("Posez votre question sur les films..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        response = self.get_response(prompt)
                        st.markdown(response, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
        except Exception as e:
            st.error(f"Erreur d'affichage du chat: {str(e)}")

# Création d'une instance du chatbot et lancement de l'application
if __name__ == "__main__":
    st.title("🎬 CineBot - Votre Assistant Cinéma")
    bot = MovieChatbot()
    bot.display()