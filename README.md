# Use DQN to solve "Environment 1: Paddle"

The Environment is based on "Environment 1: Paddle" from https://github.com/shivaverma/Orbit. And I made some modifications.

Now the environment can be conclude to that(I bold the part I modified):

**Action space (3)**

- **`0`** - move paddle to left
- **`1`** - do nothing
- **`2`** - move paddle to right

**State space (5)**

- x position of paddle
- x and y position of ball
- x and y velocity of ball

**Reward function**

| Reward   | Description                 |
| -------- | --------------------------- |
| **+3**   | when paddle hits the ball   |
| **-3**   | when ball touchs the ground |
| **-0.1** | when paddle moves           |

**Episode termination**

- Episode ends when ball touchs the ground, **or paddle hits the ball for 10 times.**

**Other changes**

- **Some parameters have been modified to make the strike feel more real.**

- **Added randomness. Now the ball will appear on the screen with a random speed and a random position.**

  **details as follows:**

  **x position: choose from [-290,290] with equal probability(only interger).**

  **y position: fix in 100.**

  **x velocity: choose from [-6,-1]∪[1,6] with equal probability(only interger).**

  **y velocity: choose from [-5,-2] with equal probability(only interger).**

  

  compare to old setting:

  x position: fix in 0.

  y position: fix in 100.

  x velocity: inherit the last speed.

  y velocity: fix in -3.

  

My solution uses DQN, based on Parl.

## How to use

To train the model:

```
python -u train.py
```

To eval the model:

```
python -u eval.py
```

If you want to test the model you choose, you can change the model in code. Now the model I offered is the best model I ever trained.

Although I don’t know the highest score, a good model should meet two requirements:
1. Do not let the ball touch the ground.
2. Do not make extra moves.

## Result

![](gif\example1.gif)

![](gif\example2.gif)

As you can see, the ball may have different speeds.