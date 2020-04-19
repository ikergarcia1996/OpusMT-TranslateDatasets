# OpusMT: Translate Datasets
[Opus-MT](https://github.com/Helsinki-NLP/Opus-MT) provides pre-trained translation systems for a huge amount of languages. This is a very valuable resource for the research community. However OpusMT architecture is not ideal for translating large texts or datasets. This changes that allowing translating a large text file with just one command!!

# Simplified Installation
## Install Marian
Instructions: https://marian-nmt.github.io/docs/

### Install requeriments:
```
sudo apt-get install ca-certificates git wget gnupg build-essential lsb-release g++ automake autogen libtool cmake-data cmake unzip libboost-all-dev libblas-dev libopenblas-dev libz-dev libssl-dev libprotobuf17 protobuf-compiler libprotobuf-dev python3-dev python3-pip python3-setuptools python3-websocket
```

### GPU support (strongly reccomended. CUDA required):
```
git clone https://github.com/marian-nmt/marian
mkdir marian/build
cd marian/build
cmake .. -DCOMPILE_SERVER=on -DUSE_SENTENCEPIECE=on -DCOMPILE_CUDA=on -DUSE_STATIC_LIBS=on
make -j4
sudo cp marian-server /usr/local/bin/
sudo cp marian-vocab /usr/local/bin/
sudo cp marian-decoder /usr/local/bin/
sudo cp marian-scorer /usr/local/bin/
sudo cp marian-conv /usr/local/bin/
sudo cp libmarian.a  /usr/local/lib/
```

### Only CPU:

1) Install intel mkl library:

```
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update && sudo apt-get install intel-mkl-64bit-2019.5-075
rm -f GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
```
If you want to install a more recent version of the intel mkl library
```
sudo apt-get update && sudo apt-get install intel-mkl-64bit-2020.1-102 2020.1-102
```

2) Install Marian without CUDA: 

```
git clone https://github.com/marian-nmt/marian
mkdir marian/build
cd marian/build
cmake .. -DCOMPILE_SERVER=on -DUSE_SENTENCEPIECE=on -DCOMPILE_CPU=on -DCOMPILE_CUDA=off -DUSE_STATIC_LIBS=on
make -j4
sudo cp marian-server /usr/local/bin/
sudo cp marian-vocab /usr/local/bin/
sudo cp marian-decoder /usr/local/bin/
sudo cp marian-scorer /usr/local/bin/
sudo cp marian-conv /usr/local/bin/
sudo cp libmarian.a  /usr/local/lib/
```


## Install Opus-MT
Using a python virtual environment is recommended:
```
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```
1) Install Opus-MT
```
git clone https://github.com/Helsinki-NLP/Opus-MT.git
cd Opus-MT
pip install -r requirements.txt
```

2) Add my script to Opus-MT
Just put translate_dataset.py into the Opus-MT repository
```
pip install tqdm
git clone https://github.com/ikergarcia1996/OpusMT-TranslateDatasets.git
mv OpusMT-TranslateDatasets/translate_dataset.py translate_dataset.py
rm -rf OpusMT-TranslateDatasets
```

# Translate a dataset

Download the pre-trained model you want to use from https://github.com/Helsinki-NLP/Opus-MT-train/tree/master/models
Unzip the model in the directory you want (i.e model/en-es)
Put the text you want to translate in a text file. The script will translate the file line by line, a sentence per line is recommended but not necessary, if there are multiple sentences in a line Opus-MT will perform sentence segmentation, however, huge lines may cause Opus-MT server to crash. In the output file, each line will correspond to the translation of each line in the input file. 

Some models use bpe codes and other models use a sentencepiece model (take a look inside the model directory to see if it contains .bpe or .spm files), set the parameters accordingly.
English -> Spanish example (BPE): 
```
python3 translate_dataset.py --dataset_path english_sentences.txt --output_path spanish_sentences.txt --decoder_path models/en-es/decoder.yml --sourcebpe models/en-es/source.bpe --targetbpe models/en-es/target.bpe --source_lang en --target_lang es 
```

Dutch -> English example (SPM)
```
python3 translate_dataset.py --dataset_path dutch_sentences.txt --output_path english_sentences.txt --decoder_path models/nl-en/decoder.yml --sourcespm ./models/nl-en/source.spm --targetspm ./models/nl-en/target.spm --source_lang nl --target_lang en      
```

You can lowercase all the words using the --lowercase_all argument. If you want to skip lowercasing the first letter of words (i.e PETTER -> Petter) use the --lowercase_capitals argument. 

