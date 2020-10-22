#!/bin/sh

pip install -q kaggle
mkdir ./.kaggle
cp kaggle.json ./.kaggle/
chmod 600 ./.kaggle/kaggle.json
kaggle datasets download -d andrewmvd/animal-faces
unzip -qq animal-faces.zip
mv ./afhq ./dataset
