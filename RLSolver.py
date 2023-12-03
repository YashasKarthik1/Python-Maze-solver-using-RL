"""
File: RLSolver.py

Author: Yashas Kartik

Input: Maze environment matrix and JSON file with Q-values

Output: Updated JSON file with Q-values

Description:
This Python script is used to solve the maze using Reinforcement Learning. Using Q-learning, the agent learns the 
best path to the end of the maze. The Q-values are stored in a JSON file for running multiple iteration.

QValue = QValue + (Reward + DiscountFactor * (MaxQValueOfNextState - QValue))
A QValue is the value of a state-action pair. The QValue is updated after every move of the agent.

We define the rewards as follows:
Reward for winning, gives an incentive for the agent to reach the end of the maze
Reward for bumping into a wall, gives a penalty for the agent to bump into a wall
Reward for each step, gives a penalty for the agent to take more steps to reach the end of the maze

Since the maze is a deterministic environment which changes dynamically, we do not need to use an epsilon greedy policy.
Although agent will choose the action with the highest Q-value 70% of the time, we use this learning rate to ensure 
that the agent explores more than exploits the environment.
"""

# importing all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import json
import keyboard
import os

# importing the GUI file
import gui

# setting the interactive mode to true
plt.ion()

# defining the constants, these values will be updated int the maze environment
EMPTY = 0
WALL = 1
END = 4
AGENT = 5

# defining the constants for the actions
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3

# defining the rewards
REWARD_FOR_WINNING = 10000
REWARD_FOR_BUMPING_INTO_WALL = -5
REWARD_FOR_EACH_STEP = -0.1

# defining the learning parameter
DISCOUNT_FACTOR = 0.4
# 70% of the time, the agent will choose the action with the highest Q-value
LEARNING_RATE = 0.7


# creating the class for the RL solver
class RLSolver:
    # initializing the class
    def __init__(self, mazeFile, JSONFile):
        self.environment = None
        self.mazeFile = mazeFile
        self.JSONFile = JSONFile

        self.actionsList = []
        self.QValueDict = {}

        # loading the maze environment from the excel file and initializing the JSON file
        self.row, self.col = self.loadMazeEnv()
        self.initJSON()

        # setting the initial conditions of the reinforcement learning agent
        self.reward = 0
        self.action = 0
        self.state = f"{self.row} {self.col}"
        self.previousState = f"{self.row} {self.col}"

    # loading the maze environment from the excel file
    def loadMazeEnv(self):
        data = pd.read_excel(self.mazeFile, header=None)
        self.environment = np.array(data)
        agentPose = np.where(self.environment == AGENT)
        return agentPose[0][0], agentPose[1][0]

    # initializing the JSON file and updating the QValue dictionary depending on previous iterations
    def initJSON(self):

        # if the JSON file exists and is not empty, load the Q-values from the JSON file to the QValue dictionary
        if os.path.exists(self.JSONFile) and os.path.getsize(self.JSONFile) > 0:
            print("JSON file exists. Loading Q-values from JSON file\n")
            with open(self.JSONFile, "r") as file:
                self.QValueDict = json.load(file)

        # if the JSON file does not exist, create the JSON file and initialize all QValues to 0
        else:
            print(
                "JSON file does not exist. Creating JSON file to save all the Q-Values\n")
            self.QValueDict = {f"{row} {col}": [0, 0, 0, 0] for row in range(self.environment.shape[0]) for col in range(
                self.environment.shape[1]) if self.environment[row, col] == EMPTY or self.environment[row, col] == AGENT}
            jsonInput = json.dumps(self.QValueDict, indent=2)
            with open(self.JSONFile, "w") as file:
                file.write(jsonInput)

    # updating the JSON file with the updated QValues after each iteration
    def updateJSONFile(self):
        jsonInput = json.dumps(self.QValueDict, indent=2)
        with open(self.JSONFile, "w") as file:
            file.write(jsonInput)

    # making the action and updating the environment based on the action
    def makeAction(self, action):

        # updating the previous state variable and the environment
        self.environment[self.row, self.col] = EMPTY
        self.previousState = f"{self.row} {self.col}"

        if action == RIGHT:
            self.col += 1
        elif action == LEFT:
            self.col -= 1
        elif action == UP:
            self.row -= 1
        elif action == DOWN:
            self.row += 1

        # updating the current state and the position of the agent after the action
        self.action = action
        self.state = f"{self.row} {self.col}"
        self.environment[self.row, self.col] = AGENT

        # updating the reward for each step
        self.reward += REWARD_FOR_EACH_STEP

    # checking for the available actions for the agent, if the agent bumps into a wall, the reward is updated
    def getAvailableActionList(self):
        self.actionsList = []

        # checking if the agent can move in the UP, DOWN, RIGHT, LEFT directions based on the value in the environment is EMPTY
        if self.row > 0 and self.environment[self.row - 1, self.col] == EMPTY:
            self.actionsList.append(UP)
        else:
            self.reward += REWARD_FOR_BUMPING_INTO_WALL

        if self.row < self.environment.shape[0] - 1 and self.environment[self.row + 1, self.col] == EMPTY:
            self.actionsList.append(DOWN)
        else:
            self.reward += REWARD_FOR_BUMPING_INTO_WALL

        if self.col > 0 and self.environment[self.row, self.col + 1] == EMPTY:
            self.actionsList.append(RIGHT)
        else:
            self.reward += REWARD_FOR_BUMPING_INTO_WALL

        if self.col < self.environment.shape[1] - 1 and self.environment[self.row, self.col - 1] == EMPTY:
            self.actionsList.append(LEFT)
        else:
            self.reward += REWARD_FOR_BUMPING_INTO_WALL

    # updating the QValue after each iteration
    # QValue(s,a) = QValue(s,a) + (Reward + DiscountFactor * (MaxQValueOfNextState - QValue(s,a)))
    def updateQValue(self):
        self.QValueDict[self.previousState][self.action] += DISCOUNT_FACTOR * (self.reward + max(self.QValueDict[self.state]) -
                                                                               self.QValueDict[self.previousState][self.action])

        # resetting the reward to 0 after each iteration
        self.reward = 0

    # getting the action based on the learning rate
    def getAction(self):
        if random.random() < LEARNING_RATE:
            availableQValues = [self.QValueDict[self.state][action]
                                for action in self.actionsList]

            # if there are multiple actions with the same QValue, choose a random action from the list of actions
            if availableQValues:
                maxQValue = max(availableQValues)
                maxIndices = [index for index, q_value in enumerate(
                    availableQValues) if q_value == maxQValue]
                bestAction = random.choice(
                    [self.actionsList[index] for index in maxIndices])
                return bestAction

        # 30% of the time, choose a random action from the list of actions
        return random.choice(self.actionsList)

    # checking if the agent has reached the end of the maze
    def checkIfWon(self):
        if self.environment[self.row, self.col+1] == END:
            self.reward += REWARD_FOR_WINNING
            return True
        else:
            return False


# main function to run the program
if __name__ == "__main__":
    solver = RLSolver("mazeEnv.xlsx", "QValues.json")
    solver.getAvailableActionList()
    print("Press ESC to save the Q-values to JSON file and exit the program\n")
    print("Learning the maze environment...\n")

    # running the program until the agent reaches the end of the maze or the user presses ESC
    while not solver.checkIfWon():
        action = solver.getAction()
        solver.makeAction(action)
        solver.getAvailableActionList()
        solver.updateQValue()
        gui.showGUI(solver.environment)
        # if the user presses ESC, save the Q-values to the JSON file and close the maze environment
        if keyboard.is_pressed('esc'):
            solver.updateJSONFile()
            print("Saving Q-values to JSON file...")
            plt.close()
            break

    # displaying the heatmap
    gui.drawHeatmap()
