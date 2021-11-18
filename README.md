
# SAMARAIZA 
You are in a debate or an interview where you might need the summarised 
notes of a particular person, or in a seminar or meetings where simultaneously making notes and listening
is not productive hence short notes of the speaker would really do the work...

### What is SAMARAIZA..?
![](https://github.com/RohitSinghDev/SAMARAIZA/blob/main/samarizer.png)

SAMARAIZA has a **speaker recognition** model based on **ML** which
could train itself by taking the audio files of the desired speaker.
Now, after training it takes an audio file to test, if it recognises the speaker, 
it starts recording the audio of the speaker. Later the audio recorded is converted to 
text and the text is then **summarised**. Also the summarised text is given 
in the form of **docx file** which has the major keywords
highlighted and also provides **detailed notes/definition** of the keywords.

It also has a **Chatbot**, which provides instructions/guide to use the SAMARAIZA.

## Major Features and components: 
1. **GMM ML speaker recognition model:**
Model can take the audio files from the user and can train itself. 5 audio files are taken 
of a particular person each of 10 secs to train the model.

A testing audio file of 10 sec is taken from the user to identify the speaker.
 The user can anytime choose to train or test the model from anyone's audio file after recording it.

2. **Audio to text:** 
A python package which converts identified audio file to text.

3. **Summarisation:**
It provides 3 different algorithms for the user to choose to summarise the text:
+ using python libs sumy and nltk
+ using python libs spacy
+ using python lib pytorch, pegasus tokenizer

4. **Chatbot:**
A deep learning chabot made using tensorflow, nltk, tflearn to give the user instructions about using the SAMARAIZA.



## Requirements:
#### Python Lib to be installed: 
+ wave
+ pyaudio
+ warnings
+ numpy
+ sklearn
+ python_speech_features
+ speech_recognition
+ docx
+ yake
+ spacy
+ sumy
+ nltk
+ pytorch
+ transformers
+ sentencepiece


