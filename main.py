import re
import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("output.csv")


st.markdown("""
            ### Mini Project - *Spotify Song Recommendation*
            #### Project by  A47 - Samyak Jasani, A53 - Sarang Kulkarni
            ***
            """)


page = st.sidebar.radio(
    'Navigation', ['Exploratory Data Analysis', 'Recommendation '])

if page == 'Exploratory Data Analysis':
    st.title("Exploratory Data Analysis")

    st.subheader("Sample data from dataset")
    st.dataframe(df.sample(5))

    st.markdown("""
                ***
                """)

    st.subheader("Basic Details")
    st.write(f'Number of Songs - {df["Track"].count()}')
    st.write(f'Number of Artists - {df["Artist"].nunique()}')
    st.write(f'Number of Album - {df["Album"].nunique()}')
    st.write(
        f'Average Duration of Songs - {round(float(df["Duration_ms"].mean()) / (60000),2)} minutes')

    st.markdown("""
                ***
                """)

    st.subheader("Heatmap")
    df1 = df[['Danceability', 'Energy', 'Key', 'Loudness',
              'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',
              'Valence', 'Tempo', 'Duration_ms']]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df1.corr(), cmap='crest')
    st.pyplot(fig)

    st.markdown("""
                ***
                """)

    st.subheader("Data by Artist")
    df2 = df[['Artist', 'Danceability', 'Energy', 'Key', 'Loudness',
              'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',
              'Valence', 'Tempo', 'Duration_ms']]
    df2 = df2.groupby('Artist').mean()
    st.dataframe(df2.sample(5))

    st.markdown("""
                ***
                """)

    st.subheader("Data by Album")
    df2 = df[['Album', 'Danceability', 'Energy', 'Key', 'Loudness',
              'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',
              'Valence', 'Tempo', 'Duration_ms']]
    df2 = df2.groupby('Album').mean()
    st.dataframe(df2.sample(5))

    st.markdown("""
                ***
                """)

    st.subheader("Explore Artist")

    elements = list(df['Artist'].unique())
    artist = st.text_input("Enter Artist Name")
    search_query = artist+".*"

    matching_elements = [
        element for element in elements if re.match(search_query, element)]

    st.text(matching_elements)

    # artist = st.text_input("Enter Name of Artist")
    if st.button("Explore"):
        try:
            soloartist = df[df['Artist'] == artist]
            li = []
            for i in soloartist.columns[3:]:
                temp = str(soloartist[soloartist[i] ==
                           max(soloartist[i])]['Track'])
                temp = temp.split(" ")
                temp = temp[1:]
                temp = (" ").join(temp)
                temp = temp.split("\n")
                temp = temp[0]
                li.append([i, temp])
            # st.write(li)
            for i in li:
                st.text(f'{i[0]} -> {i[1].strip()}')
        except:
            st.write(f"No Artist Found named {artist}")

else:
    st.title("Recommendation Model")

    elements = list(df['Track'].unique())
    song = st.text_input("Enter Song Name")
    search_query = song+".*"

    matching_elements = [
        element for element in elements if re.match(search_query, element)]

    st.text(matching_elements)

    if st.button("Enter"):
        numerical_features = ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness',
                              'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms']
        for feature in numerical_features:
            df[feature] = df[feature].fillna(0)
        cs = cosine_similarity(df[numerical_features])

        indices = pd.Series(df.index, index=df['Track']).drop_duplicates()

        def get_recommendations(track_name, cosine_sim=cs):
            idx = indices[track_name]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            track_indices = []
            sim_score = []
            for i in sim_scores:
                track_indices.append(i[0])
                sim_score.append(i[1] * 100)
            df_sim = pd.DataFrame(sim_score, index = track_indices, columns =['Similarity (%)'])
            track_indices = [i[0] for i in sim_scores]
            # return df.iloc[track_indices][['Track', 'Artist', 'Album']]
            return pd.concat([df.iloc[track_indices][['Track', 'Artist', 'Album']], df_sim], axis=1)
        
        st.dataframe(get_recommendations(song), hide_index=True)
