# AI playing GeoGuessr explained

Author: Pavel Roganin

## Prerequisites to read

None

## Introduction

Everyone has played [GeoGuessr](https://www.geoguessr.com/) at least once in their life. This is how the game looks like:

![Untitled](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled.png)

If you do not remember this game, I will briefly explain the rules of the game. 

This is a simple browser game that selects a random location from around the world and shows the player interactive Google Street View panoramas of that location. The player can move through the streets and look around. The player’s task is to determine where he is and point the expected location on the world map. The closer the location guessed by the player is to the real one, the more points he receives.

This game can be played alone or with other players. You can limit the game area to a specific continent or country, and you can play a number of different game modes.

At the end of the game, GeoGuessr shows your score, your guesses, and how far they are from the actual locations. The game is actually quite challenging. My results usually look like this:

![Untitled](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%201.png)

Not even once I guessed a right country, let alone being close to the real location. 

But there are players whose result looks like this:

![Untitled](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%202.png)

You know what is much more interesting than the score? This is a screenshot from the YouTube video: **[geoguessr pro guesses in 0.1 seconds](https://www.youtube.com/watch?v=ff6E4mrUkBY)** …

## Neural Networks

In this video, the person doesn't even look around but can guess the country from a picture shown for just a fraction of a second.

Clearly, he doesn't have time to look into details and think. This means his brain is so well-trained that he doesn’t need to consciously think. He just makes a guess based on his enormous experience in this game.

This is very similar to how Neural Networks (NN) work. NNs are actually inspired by the structure and function of biological neural networks. Essentially, it is a network of neurons that are connected and send information to each another.

![Untitled](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%203.png)

Neural networks also learn from experience, just like the human brain. Due to this, neural network is a very powerful and widely used model in Machine Learning.

However, it's worth noting a difference between how RAINBOLT (the player from the video) learned to play at this level and how NN learns. We will come to this in a moment.

**Important:**

Because RAINBOLT proved that you do not need to look around in the game to achieve great results, we will simplify the game to just one picture upon which we are supposed to guess a country. This way it will be much easier to use a neural network.

![High level view on structure of a neural network for pictures](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%204.png)

<figcaption style="font-size: 0.8em; color: grey; font-weight: normal;">High level view on structure of a neural network for pictures</figcaption><br><br>

NN might have the following inner dialogue when playing and learning the simplified game:

**At the very beginning (zero experience):** 

“They showed me picture and said that I have to guess a country… How am I supposed to guess, I know nothing! But I have to say something… Okay, let it be Slovakia!”

Correct answer: Finland.

“How am I supposed to know?”

**Later (already having some experience):** 

“I see something (hieroglyphics) that I saw before, and the correct answer was China… So, I say China this time.”

Correct answer: Japan.

“How? There must be something else that distinguishes China and Japan… How about the streets? Last time they were wide, and this time they are very narrow… Okay, next time if I see hieroglyphics I’ll look at the streets. If they are wide, I'll guess China; if narrow, then Japan.”

![China](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%205.png)

<figcaption style="font-size: 0.8em; color: grey; font-weight: normal;">China</figcaption><br><br>

![Japan](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%206.png)

<figcaption style="font-size: 0.8em; color: grey; font-weight: normal;">Japan</figcaption><br><br>

Now, how is human learning different? Humans have more sources of information available. RAINBOLT could asked his friend who also plays GeoGuessr for some insights (for example, what to play attention to when seeing hieroglyphics). That way, he would already know some facts without needing to play many games to figure them out through trial and error.

Doing the same for a neural network would enormously complicate its design, but it is in fact not needed. Neural networks have an advantage: processing speed. Humans cannot compete with computer in processing speed. An average human finishes one game in 100 seconds, while neural networks can play 100 games per second. This is enough for NN to learn just from experience without extra help.

## Training a Neural Network

As mentioned earlier, a neural network will learn just by looking at the pictures. Therefore, we need to gather pictures from different countries featured in the game.

Fortunately, there is a [collection](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k) of these pictures (dataset) available on Kaggle.

We will use DenseNet121, a Dense Convolutional Neural Network (Dense CNN), for this task.

We'll train and use the model on Kaggle because it provides 29 GB of RAM, whereas Google Colab offers only 13 GB, which is insufficient for efficient model training.

![Training the model on Google Colab with batch size = 2 (amount of images the model trains on simultaneously)](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%207.png)

<figcaption style="font-size: 0.8em; color: grey; font-weight: normal;">Training the model on Google Colab with batch size = 2 (amount of images the model trains on simultaneously)</figcaption><br><br>

Additionally, Google Colab offers only one GPU, while Kaggle provides two GPUs, which boosts training performance by 1.5x.

**Comparison of 1 GPU vs 2 GPUs**

Utilizations of resources and training performance for **one** GPU on Kaggle (batch size = 32):

![Untitled](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%208.png)

![Untitled](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%209.png)

Utilizations of resources and training performance for **two** GPUs on Kaggle (batch size = 32):

![Untitled](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%2010.png)

![Untitled](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%2011.png)

The code used for training was originally written by [ANIMESH SINGH BASNET](https://www.kaggle.com/code/crypticsy/geo-guesser/notebook).

I have made [adjustments](https://www.kaggle.com/code/pavelroganin/geo-guesser?scriptVersionId=177970716) to the model architecture and achieved 52% accuracy, compared to the original 43%.

The model was successfully trained for 30 epochs over 3 hours using two GPUs on Kaggle.

## Explaining the model

We are going to use LIME method for explanations. LIME method provides insights into the decision-making process of complex models by approximating them with simpler, more interpretable models.

Here is an example of how the explanations look:

![Untitled](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%2012.png)

Our model works on images with a resolution of 224x224. This resolution is convenient for the model because it contains enough information for pattern recognition while greatly reducing computation times.

The first picture shows the original image, its actual class (the country depicted), and the class predicted by our model. The second picture highlights what the model focused on when making its prediction.

As you can see, the grass made the model think that the country is Argentina. You might wonder, how does that make sense? This is the same type of question you could ask RAINBOLT: how does he guess the right country by looking at a random picture in the middle of nowhere?

Let's look at another explanation:

![Untitled](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%2013.png)

This time, the model focused on the pavement. Yet this was sufficient for the model to guess correctly.

Let's look at a couple more explanations:

![Untitled](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%2014.png)

![Untitled](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%2015.png)

![Untitled](https://exfai.xyz/docs/groups/AI%20playing%20GeoGuessr%20explained/Untitled%2016.png)

However, the model is not perfect and sometimes makes incorrect guesses. This might be due to cases like the one in image 5 above. In this case, the model took into consideration the map, which never changes in any of the pictures and doesn’t give any useful information. This suggests the model needs more training to recognize that focusing on the map is not helpful.

## Conclusion

Neural networks utilize a very powerful mechanism designed by nature to be perfect, making them powerful tools for solving a wide range of tasks. Despite their complexity, there are methods that allow us to peak into neural networks’ mind. One of them is LIME. 

To demonstrate the method, we formulated a problem of guessing a country by looking at an image inspired by the game GeoGuessr. We explored the human capabilities in solving this problem, considered the advantages and disadvantages humans have over neural networks. Next, we trained a neural network on Kaggle and used LIME to understand model’s decision making and faults in its decision making.
