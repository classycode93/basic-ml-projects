
# YouTube Title Generator using LSTM

This project trains an LSTM neural network to generate YouTube video titles
based on trending video datasets.

## Features
- NLP preprocessing
- Tokenization and n-gram generation
- Sequence padding
- LSTM deep learning model
- Title generation from seed text

## Dataset
YouTube Trending Videos Dataset (US, CA, GB)

Files needed:
- USvideos.csv
- CAvideos.csv
- GBvideos.csv
- US_category_id.json
- CA_category_id.json
- GB_category_id.json

Place them in the **data/** folder.

## Run

pip install -r requirements.txt
python main.py
