# BBBall-RL
Let reinforcement learning agent master Bouncy Basketball android game.

## Distributed Learning PPO env setup
1. install android tools for each cpu server
2. install scrcpy for each cpu server
3. make python env and install packages for cpu and gpu server
4. run master script on gpu server
5. run worker script on cpu server

## Distributed Learning method
Using 9 cpu (worker) and 1 gpu (master) workstation from NTU CSIE.

Because of the thread limit of student user account, opening 3 android emulator on each cpu workstation is a suitable choice.

The transmission of model weight and env data is through python zmq library.

For each round:
1. Each worker update current model weight from master.
2. Each worker has 3 concurrent environment (android emulator), and each plays 1 episode using the latest model weight.
3. Each worker sends back 3 episode of data (~900 steps) to master.
4. After all 9 * 3 = 27 episode of data is collected, the master use these data to train and update the model weight.

## Low latency android emulation and control
Using adb screencap and adb tap is slow. The latency is so large (~1 sec) that it isn't suitable for agents to train and to play games.

We use scrcpy library, with custom python script as the middleman for scrcpy server and the agent.

The sum of the round trip time for:
1. Python script send input action
2. Android emulator received action
3. Android emulator reacts and screen changes 
5. Python script received the updated screen

is averaging about 90 ms, and no more than 140 ms in 1000 tests.

So setting the time step for our RL agent to be 150 ms is suitable.

## How latency round trip time is benchmarked
1. We open tap counter website on the android emulator: https://samvlu.github.io/web-04-tap-counter/
2. Python script to send tap action to tap the counter
3. We continue to retrive the screen image, and check whether the number on the screen has changed
4. We measure how much time has elapsed after we send the tap action

* We have set a threshold so that the noise won't be detected as number increase
* We only send the next tap after the previous number increased
* We saved screenshot to verify
* We assume our bouncy basketball game is not less reactive than the chrome browser

## How reward is retrived from the game screen
In bouncy basketball game, it is like a real stadium, there is a TV showing a 2 point ball, a 3 point ball or a dunk is made.

After that the TV will show the possesion of the ball belongs to which team.

So using these image matching on the TV, if we first detect a 3 point ball is made, and the next possesion belongs to blue team. Then we know that red team scores a 3 point ball.

If our agent is blue team, then the reward will be -3.

* Dunk showing on the TV is consider a 2 point
* Besides 2 point ball, 3 point ball and dunk, there is also a buzzer beater, but our environment will omit it. So our agent won't receive any positive our a negative reward if buzzer beater happens.

## How effective is our distributed system
### Single cpu workstations scenario
On cpu workstation, each episode takes 60 seconds, and a model update (10 epochs) takes 240 seconds.

With 3 emulators, 27 episodes + model updates takes:
9 * 60 + 240 = 780 seconds.

### 9 cpu workstations + 1 gpu workstation scenario
On cpu workstation, each episode still takes 60 seconds.
On gpu workstation, each model update takes 20 seconds.

With 9 * 3 emulators, 27 episodes + model updates takes:
60 + 20 = 80 seconds

* We didn't measure the time to distribute data and model weights, and this should be done in the future.