# Dueling-Double-Q-Network
A Dueling Double Q-Network Implementation for solving RL environments in PyTorch

# Project Details

<ul>
  <li> The environment consists of agent where the task of the agent is to collect yellow bananas to increase the cummulative reward</li>
  <li> The current state of the environment is represented by 37 dimensional feature vector and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.</li>
  <li> The agent can interact with the environment using 4 actions :<br>
  <ol>
    <li>0 - move forward</li>
    <li>1 - move backward</li>
    <li>2 - turn left</li>
    <li>3 - turn right</li>
  </ol></li>
  <li> Given this information, the agent has to learn how to best select actions</li>
  <li> A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana </li>
  <li> The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes</li>
</ul>

# Technical Dependencies

<ol>
  <li> Python 3.6 :
  <li> PyTorch (0.4,CUDA 9.0) : pip3 install torch torchvision</li>
  <li> ML-agents (0.4) : Refer to <a href = "https://github.com/Unity-Technologies/ml-agents/">ml-agents</a> for installation</li>
  <li> Numpy (1.14.5) : pip3 install numpy</li>
  <li> Matplotlib (3.0.2) : pip3 install matplotlib</li>
  <li> Jupyter notebook : pip3 install jupyter </li>
  <li> Download the environment from <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip">here</a> and place it in the same folder as that of Navigation.ipynb file  </li>
</ol>

# Network details

- [x] Deep Q - Network
- [x] Double Deep Q - Network
- [x] Dueling Deep Q - Network

# Installation Instructions :
`
step 1 : Install all the dependencies
`
<br>
`
step 2 : git clone https://github.com/adithya-subramanian/Dueling-Double-Q-Network.git
`
<br>
`
step 3 : jupyter notebook
`
<br>
`
step 4 : Run all cells in the Navigation.ipynb file
`
