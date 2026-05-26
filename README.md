# BBBall-RL

Let a reinforcement learning agent master the Bouncy Basketball Android game.

## Distributed Learning PPO Environment Setup

1. Install Android tools on each CPU server.
2. Install scrcpy on each CPU server.
3. Create a Python environment and install packages on both CPU and GPU servers.
4. Run the master script on the GPU server.
5. Run the worker script on the CPU servers.

## Distributed Learning Method

Using 9 CPU (worker) workstations and 1 GPU (master) workstation from NTU CSIE.

Because of the thread limit on student user accounts, running 3 Android emulators on each CPU workstation is a suitable choice.

The transmission of model weights and environment data is done through the Python zmq library.

For each round:

1. Each worker updates the current model weights from the master.
2. Each worker has 3 concurrent environments (Android emulators), and each plays 1 episode using the latest model weights.
3. Each worker sends back 3 episodes of data (~900 steps) to the master.
4. After all 9 × 3 = 27 episodes of data are collected, the master uses these data to train and update the model weights.

## Low-Latency Android Emulation and Control

Using adb screencap and adb tap is slow. The latency is very large (~1 second), which is not suitable for agents to train and play games.

We use the scrcpy library with a custom Python script as the middleman between the scrcpy server and the agent.

The total round-trip time for:

1. The Python script sending an input action
2. The Android emulator receiving the action
3. The Android emulator reacting and the screen changing
4. The Python script receiving the updated screen

averages about 90 ms, and does not exceed 140 ms in 1000 tests.

Therefore, setting the time step for our RL agent to 150 ms is suitable.

## How the Latency Round-Trip Time Is Benchmarked

1. We open a tap counter website on the Android emulator: [https://samvlu.github.io/web-04-tap-counter/](https://samvlu.github.io/web-04-tap-counter/)
2. A Python script sends tap actions to tap the counter.
3. We continuously retrieve the screen image and check whether the number on the screen has changed.
4. We measure how much time has elapsed after sending the tap action.

* We set a threshold so that noise will not be detected as a number increase.
* We only send the next tap after the previous number increases.
* We save screenshots for verification.
* We assume that our Bouncy Basketball game is at least as reactive as the Chrome browser.

## How Rewards Are Retrieved from the Game Screen

In Bouncy Basketball, the game resembles a real stadium, where a TV screen shows whether a 2-point shot, a 3-point shot, or a dunk is made.

After that, the TV shows which team gains possession of the ball.

By using image matching on the TV screen, if we first detect that a 3-point shot is made and the next possession belongs to the blue team, then we know that the red team scored a 3-point shot.

If our agent is on the blue team, then the reward will be -3.

* A dunk shown on the TV is considered a 2-point score.
* Besides 2-point shots, 3-point shots, and dunks, there is also a buzzer beater, but our environment ignores it. Therefore, our agent will not receive either a positive or negative reward if a buzzer beater happens.
* Since the TV will display 1 second after the points are scored, the reward is a little bit delayed for training.

## How Effective Our Distributed System Is

### Single CPU Workstation Scenario

On a CPU workstation, each episode takes 60 seconds, and a model update (10 epochs) takes 240 seconds.

With 3 emulators, 27 episodes plus model updates take:
9 × 60 + 240 = 780 seconds.

### 9 CPU Workstations + 1 GPU Workstation Scenario

On CPU workstations, each episode still takes 60 seconds.
On the GPU workstation, each model update takes 20 seconds.

With 9 × 3 emulators, 27 episodes plus model updates take:
60 + 20 = 80 seconds.

* We did not measure the time required to distribute data and model weights, and this should be done in the future.