import streamlit as st
from openai import OpenAI
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os

# Включаем поддержку HTML и CSS для стилизации
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
            # Инициализируем состояние сессии первым
            self.initialize_session_state()
            
            # Инициализация OpenAI и embeddings
            self.client = OpenAI(api_key=st.secrets["OpenAI_key"])
            self.embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OpenAI_key"])
            
            # Загружаем основные данные
            self.films = pd.read_csv("csv/films_def.csv")
            self.intervenants = pd.read_csv("csv/intervenants_def.csv")
            self.lien = pd.read_csv("csv/lien_def.csv")
            
            # Создание или загрузка vectorstore
            self.vectorstore = self._create_or_load_vectorstore()
                
        except Exception as e:
            st.error(f"Erreur d'initialisation du chatbot: {str(e)}")
            st.error(f"Type d'erreur: {type(e).__name__}")
            if hasattr(e, 'args'):
                st.error(f"Arguments de l'erreur: {e.args}")

    def _create_or_load_vectorstore(self):
        persist_directory = "./new_chroma_db"
        
        # Проверяем существование директории
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            
        try:
            # Пробуем загрузить существующую базу
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            # Проверяем работоспособность
            vectorstore.similarity_search("test", k=1)
            return vectorstore
            
        except Exception as e:
            st.warning(f"La base existante n'a pas pu être chargée, création d'une nouvelle base. Erreur: {str(e)}")
            # Создаем новую базу, если загрузка не удалась
            documents = self._prepare_movie_documents()
            vectorstore = Chroma.from_texts(
                texts=[doc["content"] for doc in documents],
                metadatas=[doc["metadata"] for doc in documents],
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            vectorstore.persist()
            return vectorstore
    
    def _prepare_movie_documents(self):
        documents = []
        for _, movie in self.films.iterrows():
            try:
                # Проверяем год фильма
                year = int(movie['release_date'][:4]) if pd.notna(movie['release_date']) else 0
                
                # Пропускаем фильмы после 2000 года
                if year > 2000:
                    continue
                    
                # Форматируем название фильма
                title = f"**<span style='color: pink'>{movie['title']}</span>**"
                
                content = f"Titre: {title}\n"
                content += f"📅 Année: {year if year != 0 else 'Non disponible'}\n"
                content += f"🎭 Genre: {movie['genres'] if pd.notna(movie['genres']) else 'Non spécifié'}\n"
                content += f"⭐ Note: {movie['averageRating'] if pd.notna(movie['averageRating']) else 'Non disponible'}/10\n"
                content += f"📝 Synopsis: {movie['overview'] if pd.notna(movie['overview']) else 'Non disponible'}\n"
                
                # Получаем актеров для фильма
                movie_actors = self.lien[
                    (self.lien['tconst'] == movie['tconst']) & 
                    (self.lien['category'] == 'actor')
                ]
                actors = self.intervenants[
                    self.intervenants['nconst'].isin(movie_actors['nconst'])
                ]
                content += f"🎬 Acteurs: {', '.join(actors['primaryName'])}\n"
                
                # Добавляем ссылку на трейлер, если она есть
                if pd.notna(movie['trailer_link']):
                    content += f"🎥 Bande-annonce: [Regarder le trailer]({movie['trailer_link']})\n"
                    # Добавляем информацию о языке трейлера, если она есть
                    if pd.notna(movie['langue_trailer']):
                        content += f"🌍 Langue du trailer: {movie['langue_trailer']}\n"
                
                # Добавляем тэглайн, если он есть
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
            # Получаем embedding для пользовательского ввода
            embedding_response = self.embeddings.embed_documents([user_input])[0]
            
            # Увеличиваем количество похожих фильмов для лучших результатов
            similar_movies = self.vectorstore.similarity_search_by_vector(
                embedding_response, 
                k=10,
                filter={"year": {"$lte": 2000}}  # Фильтруем только фильмы до 2000 года
            )
            
            # Улучшаем формат контекста
            context = "Information sur les films disponibles:\n"
            for movie in similar_movies:
                context += f"\n---\n{movie.page_content}\n"
            
            # Обновляем системный промпт
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
                # Отображаем историю сообщений
                for message in st.session_state.messages:
                    if message["role"] != "system":
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"], unsafe_allow_html=True)

                # Обработка нового ввода пользователя
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

# Создание экземпляра чатбота и запуск приложения
if __name__ == "__main__":
    st.title("🎬 CineBot - Votre Assistant Cinéma")
    bot = MovieChatbot()
    bot.display()