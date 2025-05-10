# Blackjack using Epsilon-Greedy Monte Carlo

A simple implementation of a Reinforcement Learning environment from scratch for the game of Blackjack, based on the rules given in Sutton & Barto (Reinforcement Learning, 2nd edition).

The agent follows the epsilon-greedy Monte Carlo algorithm for learning the game. Here is the pseudocode of the algorithm for reference: 

![image](https://github.com/user-attachments/assets/9debe692-7ba5-4a45-91fa-72fa4f480df3)

The code only uses the random library, default to python. The main parameters used are the gamma, epsilon and the number of runs. The state representation is given in the Player class, and can be freely modified there without any impact on the code. Future releases might involve simplifications to make it easier for people to experiment with the bot and learn the algorithm better. 
