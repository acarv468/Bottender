# Bottender

## The Game
Bottender is a chatbot based game where the user acts as a bartender for robots. The user must chat with the bot to earn tips. Each "character" will have a subject that they are particularly interested in. The more the user talks about this topic, the higher the tip will be.

## Motivation
This game is intended to be a proof of concept for the use of natural language understanding (NLU) in games. In most modern video games, players can only interact with non-player characters (NPCs) through pre-scripted dialog choices the player can select. I believe that allowing the player to use natural language to interact with NPCs could add an interesting layer of gameplay to story-based games.

## Features
The project features heavy use of python libraries such as pandas, numpy, scikit-learn, and flask.

The project currently relies on 2 different classification models for 2 different purposes:
1) intent classification: predicting what the user means by his or her entry
2) toxic classification: this project will also recognize toxic language entered by the user and the characters will respond accordingly.

## Installation
This project is equipped with a Dockerfile. Clone this repository and follow the guide to spin up a Docker Container
https://docs.docker.com/get-started/part2/

## API Reference
You can access the predictive features of this project via API. Simply post the string you want to predict as "user_submission" and the API will return the predicted intent, predicted intent probability, predicted toxic, and predicted toxic probability. Here is an example:

![alt text](https://github.com/acarv468/Bottender/blob/master/bottender_api_example.jpg)

## Planned Additions
- Continue to improve model with more data
- Host via Heroku
- Create user profiles
- Improve UI
- Add more responses
- Add more characters
- Using natural language generation (NLG) to create responses

## Credits
In process...

MIT © Andrew Carver
