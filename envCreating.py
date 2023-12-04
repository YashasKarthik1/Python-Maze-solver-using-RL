''' 
File name: envCreating.py

Author: Yashas Kartik

Input: Image of the maze

Output: Excel file with the environment matrix of the maze

Description:
This Python script utilizes OpenCV to process a maze image, creating a structured matrix for maze-solving tasks. 
It employs Canny edge detection to label maze borders as '1' and open spaces as '0'. 
The resulting environment matrix is saved as 'mazeEnv.xlsx'. This concise and organized script ensures a well-defined 
environment for RL to learn the best path through the maze.
'''

# importing all the required libraries
import cv2
import numpy as np
import pandas as pd
import statistics

# defining the constants, these values will be updated in the excel file
EMPTY = 0
WALL = 1
START = 3
END = 4
AGENT = 5


# Detect and label the maze border/edges
def labelMazeEdges(inputImagePath):
    # Read the input image in grayscale and apply Gaussian blur
    image = cv2.imread(inputImagePath, cv2.IMREAD_GRAYSCALE)
    blurredImage = cv2.GaussianBlur(image, (5, 5), 0)

    # Use Canny edge detection to identify edges in the blurred image
    edges = cv2.Canny(blurredImage, 50, 150)

    # Label the edges(walls of the maze) in the matrix as 1, and empty space as 0
    labelledEdges = np.where(edges != EMPTY, WALL, EMPTY)

    # Convert the labeled edges matrix to a binary matrix (1 for maze wall, 0 for empty space) to pd dataframe and save to excel
    binaryMatrix = (labelledEdges == WALL).astype(int)
    mazeDataFrame = pd.DataFrame(binaryMatrix)
    return mazeDataFrame


# Update the dataframe to remove unnecessary rows and columns and define the agents starting position
def dataframeUpdate(mazeDataFrame):

    # Identify top row of the maze
    topRow = next(
        (index for index, row in mazeDataFrame.iterrows() if row.sum() > 20), None)

    # Remove all the white spaces in the photo thta are not the maze
    mazeDataFrame = mazeDataFrame.loc[~(mazeDataFrame == 0).all(axis=1)]
    nonZeroCols = mazeDataFrame.columns[mazeDataFrame.ne(0).any()]
    mazeDataFrame = mazeDataFrame[nonZeroCols]

    # Set Start and End borders
    mazeDataFrame = setStartEnd(mazeDataFrame)

    # Calculate the middle of the start and set AGENT initial position
    agentStartPose = int(statistics.median(
        list(mazeDataFrame.loc[mazeDataFrame.iloc[:, 0] == START].index - topRow)))
    mazeDataFrame.iloc[agentStartPose, 1] = AGENT

    # Reset column indices
    mazeDataFrame.columns = range(len(mazeDataFrame.columns))

    # Save the updated DataFrame to an Excel file for better visualization
    output_file = 'mazeEnv.xlsx'
    mazeDataFrame.to_excel(output_file, index=False, header=False)

    return mazeDataFrame


# defining the function to update the start and end points
# this function sets the start from the top left corner and end from the bottom right corner
def setStartEnd(mazeDataFrame):
    mazeEnv = mazeDataFrame.copy()

    # detecting the start and updting it with the value 3 in the first column or row
    if (mazeEnv.iloc[:, 0] == EMPTY).any():
        mazeEnv.iloc[:, 0] = mazeEnv.iloc[:, 0].replace(
            EMPTY, START)

    if (mazeEnv.iloc[0, :] == EMPTY).any():
        mazeEnv.iloc[0, :] = mazeEnv.iloc[0, :].replace(
            EMPTY, START)

    # detecting the end and updating it with the value 4 in the last column or row
    if (mazeEnv.iloc[:, -1] == EMPTY).any():
        mazeEnv.iloc[:, -
                     1] = mazeEnv.iloc[:, -1].replace(EMPTY, END)

    if (mazeEnv.iloc[-1, :] == EMPTY).any():
        mazeEnv.iloc[-1,
                     :] = mazeEnv.iloc[-1, :].replace(EMPTY, END)

    return mazeEnv


# main function to run the program
if __name__ == "__main__":
    inputPath = 'maze.jpg'  # input image path
    print(f"Reading image from {inputPath}...")
    mazeDataFrame = labelMazeEdges(inputPath)
    print("detecting borders..")
    envirnoment = dataframeUpdate(mazeDataFrame)
    print("Maze environment created successfully!")
