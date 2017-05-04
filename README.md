# Deep House Spy

## Summary
The Deep House Spy is an artist classification project based on a convolutional neural network (CNN). CNNs have proven to be really effective for image classification but can also work with audio when it's put in the right form.

The idea project came when I listened to a deep house dj set on soundcloud. I really liked the song that I was listening to, but Shazam just wouldn't recognize it. This happens very often and the reason for this is most of the time, that Shazam doesn't have the song in its database because it has not yet been released.

My approach to solve this problem is to learn the style of artists by using songs that have been released already and then identifying the respective artists of the unreleased songs.

## Data collection and features
I scraped over 10.000 songs previews of 100 artists from beatport.com. The song previews are 120 second snippets from the middle of the songs and can be downloaded for free.

I used the Librosa library for Python to read in the audio files. In the raw form, audio is in wave format. Using [fourier transform](https://en.wikipedia.org/wiki/Fourier_transform), the wave can be represented as herz frequencies. With a series of other transformation I extracted the [mel-frequency cepstral coefficients (MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) as they shown to be very effective in audio szene classification and music information retrieval.

![](https://s3.eu-central-1.amazonaws.com/deephousespy/img/features.png)

## MVP: Random Forest
For my baseline model I used a Random Forest on two and 10 artists. To train the model I split each song in 120 snippets of 1 second and tested it on a 5 second snippet that I took from the middle of the song.

![](https://s3.eu-central-1.amazonaws.com/deephousespy/img/train_test.png)

Instead of using the full MFCCs (44x20 for 1 second) I averaged them which left me with 20 features per snippet. This brought the following accuracy:

![](https://s3.eu-central-1.amazonaws.com/deephousespy/img/mvp_accuracy.png)

While this clearly showed that there is signal it was also obvious that I had to use a more complex model in order to do useful classifications of a higher number of artists.

## Convolutional Neural Network
For my convolutional neural networt I used structure used by Yoonchang Han and Kyogu Lee in their [paper on acustic scene classification](https://arxiv.org/pdf/1607.02383.pdf) as basis and amended it slightly to work with my data. (zero padding layers are not shown here):

Input layer, 1x44x20
Convolution #1, 32@3x3
Convolution #2, 32@3x3
MaxPool #1, 2x2
Dropout #1, 0.25
Convolution #3, 64@3x3
Convolution #4, 64@3x3
MaxPool #2, 3x3
Dropout #2, 0.25
Convolution #5, 128@3x3
Convolution #6, 128@3x3
MaxPool #3, 3x3
Dropout #3, 0.25
Convolution #7, 256@3x3
Convolution #8, 256@3x3
GlobalAvgPool #1, 3x3
Dense #1, 1024
Dropout #4, 0.5
Output layer (softmax), dense with 100 Nodes

To build the model I used Keras and TensorFlow. I trained the CNN on a GPU instance on AWS (p2.xlarge).

## Ensemble classification
As my model takes in 1 second snippets I also had to split up the songs that I would run the classification on. First I would classify each snippet individually and then use the majority of the classifications as final classification.

![](https://s3.eu-central-1.amazonaws.com/deephousespy/img/ensemble_prediction.png)

## Results
Unsurprisingly the accuracy droppes when the number of artists increases. As often done in multyclass classification, I decided to show the 5 most likely artists instead of showing one.

![](https://s3.eu-central-1.amazonaws.com/deephousespy/img/cnn_accuracy.png)

## Next steps
One major issue is to decide who to assign a song to. Songs usually have main artists but when they are remixed they also have remix artists. Other songs have multiple main artists and/ or multiple remix artists. Right now assigning these songs happened random: Beatport lists all songs associated to an artist under his song list. If the song has multiple artists (either as main oder remixers) it will appear also on the list of those other artists. My script will first scrape the song for every of those artists and then clean it to keep only one copy of it. One way to tackle this would be to treat this problem as a multi-label problem i.e. assigning all possible artists to a song.
=======
I love to listen to deep house dj sets on Soundcloud or Youtube. The problem is, that very there is no tracklist and Shazam doesn’t work with most of the newer deep house songs so it is almost impossible to figure out what the track is called that I am listening to right now. My goal was to identify the artist of the song to make further research possible.

I built a convolutional neural network using an EC2 GPU instance. I had to learn a lot about audio engineering (e.g. Fourier Transform, MFCC), neural networks, computer vision as well as TensorFlow and Keras. While classification of a few artists works really well, it is very hard when the number increases above 20. To overcome this, I used a n most likely prediction.
