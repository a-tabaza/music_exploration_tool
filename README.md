# Music Exploration Tool
This is a very simple UI I made for a music exploration tool as part of an ongoing saga to embed every possible data modality.

# Overview
![System Design](system_design.png)

# Usage
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://musicexplorationtool.streamlit.app/)

The app is live [here.](https://musicexplorationtool.streamlit.app/)

It has my own likes in it.

The app has a companion Nomic Atlas map to visualize the embeddings, you can find it [here](https://atlas.nomic.ai/data/tyqnology/likes-dump-mean-pooled-normalized-vggish/map)

To set up locally and use your own embeddings and likes, you can use the following commands:
```bash
git clone https://github.com/a-tabaza/music_exploration_tool
cd music_exploration_tool
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install streamlit
streamlit run app.py
```

