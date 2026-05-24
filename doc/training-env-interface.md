## Settings
584x286 screen resolution

## Screens
1. q1 game screen
2. next quarter screen
3. q2 game screen
4. pause screen

## Buttons
1. "rematch button" at pause screen: (346, 193)
2. "pause button" at q1 game screen and q2 game screen: (291, 239)
3. "next quarter button" at next quarter screen: (313, 193)

## Flows
1. In q1 game screen, after the quarter ended, it will go to next quarter screen
2. In next quarter screen, after tapped "next quarter button", it will go to q2 game screen
3. In q1 game screen and q2 game screen, if tapped "pause button", it will go to pause screen
4. In pause screen, if tapped "rematch button", it will go to q1 game screen


## Screen Detection
### **Script 1 (`filter_by_color.py`)**
detect next quarter screen
- **Rectangle:** `(240, 201)` to `(342, 203)`
- **RGB Range:**
  - **Red:** `2.5` to `3.0`
  - **Green:** `230.0` to `235.0`
  - **Blue:** `2.0` to `3.0`

### **Script 2 (`filter_by_color2.py`)**
detect pause game screen
- **Rectangle:** `(257, 159)` to `(324, 163)`
- **RGB Range:**
  - **Red:** `0.0` to `1.0`
  - **Green:** `160.0` to `170.0`
  - **Blue:** `0.0` to `1.0`


## RL training environment interface
### Load snapshot
Initially or when the environment is behaving unexpectedly, we can reset the environment by load the saved Android snapshot.

We have save the "game_ready" snapshot by:
```
export ANDROID_HOME=/tmp2/$USER/DRL_final_workspace/android-sdk
export ANDROID_SDK_ROOT=$ANDROID_HOME
export ANDROID_USER_HOME=/tmp2/$USER/DRL_final_workspace/.android
export ANDROID_AVD_HOME=/tmp2/$USER/DRL_final_workspace/.android/avd
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator
export ANDROID_EMULATOR_HOME=/tmp2/$USER/DRL_final_workspace/.android
```

```
adb emu avd snapshot save game_ready
```

And it can be loaded by 

```
adb emu avd snapshot load game_ready
```

However, you should mind that the connection of scrcpy, adb may need to be reconfigure.

After game_ready snapshot loaded, it will be on the pause screen. You could detect the pause screen to check if it is loaded correctly.

### Navigate to reset env in game
There are four quarters in this game, we will only use quarter 1 for training. And next quarter screen will be after each quarter.

1. If it is next quarter screen, tap "next quarter button"

2. If it is pause screen, tap "rematch button"

3. If not next quarter screen or pause screen, then it is q1 game screen or q2 game screen

4. If we have tapped "next quarter button", then it is q2 game screen

5. If it is q1 game screen or q2 game screen, then tap "pause button"

6. There will be two scenarios we want to reset the environment, after q1 ended, the episode ended. Or env.reset() is called