# Android-based-Rock-Paper_Scissor-game

This is a simple Rock, Paper, scissor image recognition app.

This is an opencv based app. You will have to download opencv manager from 
app store to run this app. And you will need to specify the path of nnet_20.xml 
on your phone into the MainActivity.java in the function nnet.load(). Save the 
nnet_20.xml on your phone. Before hitting the play button put any of the three gesture 
Rock or Paper or Scissor in front of the front camera. It will capture the image and will 
try to predict the current gesture of your hand out of the three gestures.

NeuralNetwork.java is use to create the actual neural network and train it
using the sample images.

TetstNeuralNetwork.java was use to test the neural network on sample images.

