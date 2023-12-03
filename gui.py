"""
File: GUI.py

Author: Yashas Kartik

Input: Maze environment matrix

Output: GUI of maze and heatmap

Description:
For organisation purposes, the GUI is in a separate file. 
This file is used to update the GUI with the current state of the environment of the maze.
The GUI is updated after every move of the agent.

In order to better understand the exploration of the agent, a heatmap is also created.
"""

# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
import pandas as pd


# Define the colors for the different states
COLORS = {
    1: "white",
    0: "black",
    4: "black",
    5: "#6efc1c",
    3: "black"
}

# Create the figure outside the function
fig, canvas = plt.subplots(figsize=(8, 8))


# Update the GUI with the current state of the environment
def showGUI(environment):
    # Define the color map
    cmap = ListedColormap([COLORS[val] for val in np.unique(environment)])
    # Clear the previous plot
    canvas.clear()
    canvas.imshow(environment, cmap=cmap, interpolation='nearest')
    canvas.set_title('MAZE RL SOLVER')
    plt.pause(0.001)
    plt.show()


# calculate the average Q-Values for each state
def calculateQValues(data):
    heatmapDF = pd.DataFrame(data).T
    averages = heatmapDF.mean(axis=1).to_dict()
    return averages


# draw the heatmap for better reference of the exploration
def drawHeatmap():
    # load the Q-Values from the JSON file and converting the "row col" key to a tuple list [row] [col]
    with open('QValues.json', 'r') as file:
        data = json.load(file)

    heatmapDF = pd.DataFrame(data).T
    averages = heatmapDF.mean(axis=1).to_dict()

    # load the Q-Values from the JSON file and converting the "row col" key to a tuple list [row] [col]
    heatmapDF = pd.DataFrame(list(averages.items()), columns=['Key', 'Avg'])
    heatmapDF[['Row', 'Col']] = heatmapDF['Key'].str.split(
        expand=True).astype(int)

    # create a pivot table to plot the heatmap
    heatmapData = heatmapDF.pivot(index='Row', columns='Col', values='Avg')

    # plot the heatmap
    fig, canvas = plt.subplots(figsize=(8, 8))
    canvas.imshow(heatmapData, cmap='viridis_r', interpolation='nearest')
    canvas.set_xticks(np.arange(0, heatmapData.shape[0], 50))
    canvas.set_yticks(np.arange(0, heatmapData.shape[1], 50))
    plt.title('Heatmap of Average Q-Values for each state')

    plt.show()
    plt.pause(5)  # display the heatmap for 5 seconds
    plt.close()
