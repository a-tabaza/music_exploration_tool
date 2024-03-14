import streamlit as st

st.set_page_config(
    page_title="Music Exploration Tool",
    page_icon="🎵",
)

import requests
import json
import numpy as np
import faiss

from urllib.parse import quote
st.title('Music Exploration Tool')
st.subheader('Explore your music visually and sonically.')
st.write("By: [Abdulrahman Tabaza](https://www.github.com/a-tabaza)")
st.write("This tool allows you to visually and sonically explore your music, its based on embeddings. To explore visually, headover [here](https://atlas.nomic.ai/data/tyqnology/likes-dump-mean-pooled-normalized-vggish/map). To learn more about how this works, an explanation appears once the page loads near the end of it.")

likes_dump = json.loads(open('likes_dump.json').read())

api_key = st.secrets["api_key"]

EMBEDDING_TYPE = "quantized"

if EMBEDDING_TYPE == "float32":
    embeddings_path = "embeddings_unquantized.npy"
    index_path = "likes_index_unquantized.faiss"

if EMBEDDING_TYPE == "quantized":
    embeddings_path = "embeddings_quantized.npy"
    index_path = "likes_index_quantized.faiss"

def query_lastfm(artist_name, track_name):
    res = requests.get(url = f"https://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={api_key}&artist={quote(artist_name.lower())}&track={quote(track_name.lower())}&format=json")
    return res.json()


embeddings = np.load(embeddings_path)
index = faiss.read_index(index_path)

with st.expander("Search"):
    st.write("Songs available:", len(likes_dump))
    query = st.selectbox("Select artist", sorted(list(set([song['artist_name'] for song in likes_dump]))))
    if st.button("Search"):
        song_idx = [i for i, song in enumerate(likes_dump) if song['artist_name'] == query]
        st.write("Songs found:", len(song_idx))
        for idx in song_idx:
            st.write(f"# {likes_dump[idx]['track_name']}")
            col1, col2 = st.columns(2)
            metadata = query_lastfm(likes_dump[idx]['artist_name'], likes_dump[idx]['track_name'])
            
            with col1:
                st.write("**Track:**", likes_dump[idx]['track_name'])
                st.write("**Artist:**", likes_dump[idx]['artist_name'])
                st.write("**Album:**", likes_dump[idx]['album_name'])
                D, I = index.search(embeddings[idx].reshape(1,-1), 8)
            with col2:
                try:
                    st.image(metadata["track"]["album"]["image"][-1]["#text"])
                except Exception as e:

                    st.write("No album art found")
            st.audio(likes_dump[idx]['preview_url'])
            st.write("## **Similar Tracks:**")

            for d, i in zip(D[0], I[0]):
                col3, col4 = st.columns(2)
                if i != idx:
                    sim_metadata = query_lastfm(likes_dump[int(i)]['artist_name'], likes_dump[int(i)]['track_name'])
                    with col3:
                        st.write("**Track:**", likes_dump[int(i)]['track_name'])
                        st.write("**Artist:**", likes_dump[int(i)]['artist_name'])
                        st.write("**Album:**", likes_dump[int(i)]['album_name'])
                    with col4:
                        try:
                            st.image(sim_metadata["track"]["album"]["image"][-1]["#text"])
                        except Exception as e:

                            st.write("No album art found")
                    st.audio(likes_dump[int(i)]['preview_url'])
            st.write('---')

with st.expander("Load Random Songs"):
    song_idx = np.random.choice(len(likes_dump), 3)
    songs = [likes_dump[i] for i in song_idx]
    if st.button("Load Random Songs"):
        song_idx = np.random.choice(len(likes_dump), 3)
        songs = [likes_dump[i] for i in song_idx]
    for idx, song in enumerate(songs):
        st.write(f"# {likes_dump[song_idx[idx]]['track_name']}")
        col3, col4 = st.columns(2)
        metadata = query_lastfm(song['artist_name'], song['track_name'])
        with col3:
            st.write("**Track:**", song['track_name'])
            st.write("**Artist:**", song['artist_name'])
            st.write("**Album:**", song['album_name'])

            D, I = index.search(embeddings[song_idx[idx]].reshape(1,-1), 8)
            
        with col4:
            try:
                st.image(metadata["track"]["album"]["image"][-1]["#text"])
            except Exception as e:

                st.write("No album art found")
        st.audio(song['preview_url'])
        st.write("## **Similar Tracks:**")

        for d, i in zip(D[0], I[0]):
            col5, col6 = st.columns(2)
            if i != song_idx[idx]:
                sim_metadata = query_lastfm(likes_dump[int(i)]['artist_name'], likes_dump[int(i)]['track_name'])
                with col5:
                    st.write("**Track:**", likes_dump[int(i)]['track_name'])
                    st.write("**Artist:**", likes_dump[int(i)]['artist_name'])
                    st.write("**Album:**", likes_dump[int(i)]['album_name'])
                with col6:
                    try:
                        st.image(sim_metadata["track"]["album"]["image"][-1]["#text"])
                    except Exception as e:

                        st.write("No album art found")
                st.audio(likes_dump[int(i)]['preview_url'])
        st.write('---')



st.write('''
## Learn More
Music is what is known as highly dimensional data, think of a song as a very long list of numbers representing the audio.
         
This is what a song looks like when you load it into a computer, it's a signal, sampled at a certain rate, and it's a list of numbers, something like this:
(1310328, 2) -> 2 channels, 1310328 samples.
         
Two means two channels, i.e. stereo, and the rest represent the samples of audio per second.
         
This is known as highly dimensional data.
         
The problem with highly dimensional data is that it's hard to visualize (think of the samples as coordinates in an N-dimensional space) and understand, and it's hard to compare two songs.
         
Think how hard would it be to measure the similarity between two songs, how would you do it?
         
One way to do it is to use embeddings, which are a way to represent the data in a lower-dimensional space, where it's easier to visualize and understand.
         
A million samples or so might give me the exaxt song, but I only need a few hundred to get a good idea of what the song is about.
         
This is an embedding. It's a lower dimensional vector, in this case 128 dimensions, that represents the song.
         
These embeddings are produced by a neural network, and they are trained to represent the song in a way that makes it easy to compare to other songs.
         
The songs available on this site are my own personal likes, and the embeddings are produced by a neural network trained on a large dataset of songs (VGGish).
         
The embeddings are then indexed, which allows for approximate neighbour search.                   
''')
