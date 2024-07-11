This is an implementation of a continous agent for the TD3 algorithm. The algorithm is implemented in python, and utilizes the pytorch library for neural network implementation. It also uses gymnasium environments for testing. 

![image](https://github.com/NoahRothgaber/TD3_mountaincar/assets/116089659/8fa5e2fb-d7d7-402b-8b4f-3a1b428f0dad)

Our goal is to stabilize the exploration of our model once it reaches an adequate policy range. As of now, the model is able to attain an adequate reward, but tends to explore too frequently at the higher ends. We initially attempted to train this model with the mountain car continuous environment, but our model's results have been mixed thus far. Instead, we have been using the Pendulum environment. The image above is our model after 14,000 episodes of Pendulum.  
