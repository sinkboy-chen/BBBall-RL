# BB-Ball RL Training Environment Interface

## Emulator Configuration & Resolution Settings
- **Emulator Command-line:** The emulator size is set via `adb shell wm size 540x1170`.
- **Scrcpy Capture:** The game forces landscape mode. Scrcpy captures this landscape feed with `--max-size 584`.
- **Effective Resolution:** `584x268` (Scrcpy dynamically scales the height to maintain the landscape aspect ratio and codec alignment).

## Known UI Coordinates (584x268 Layout)
1. **Rematch Button** (Pause screen): `(346, 193)`
2. **Pause Button** (Q1/Q2 Game screens): `(291, 239)`
3. **Next Quarter Button** (Next Quarter screen): `(313, 193)`

## Game States
1. **Q1 Game Screen:** The main training environment.
2. **Next Quarter Screen:** Appears after the 30-second game clock expires (plus animations).
3. **Q2 Game Screen:** Reached if the Next Quarter button is tapped.
4. **Pause Screen:** Appears when the pause button is tapped mid-game, or when the `game_ready` snapshot is loaded.

## Screen Detection (RGB Tolerances)
We classify the current screen based on the mean RGB values of specific rectangular crops.

### 1. Next Quarter Screen
- **Rectangle:** `(240, 201)` to `(342, 203)`
- **RGB Range:**
  - **Red:** `0.0` to `5.0`
  - **Green:** `210.0` to `250.0`
  - **Blue:** `0.0` to `5.0`

### 2. Pause Game Screen
- **Rectangle:** `(257, 159)` to `(324, 163)`
- **RGB Range:**
  - **Red:** `0.0` to `5.0`
  - **Green:** `140.0` to `190.0`
  - **Blue:** `0.0` to `5.0`

### 3. Game Screen (Q1/Q2)
- Assumed as default if the screen does not match either the Next Quarter or Pause criteria.

## Robust Validation & Reset Strategy
When interacting with the environment (e.g., executing soft resets or hard resets), the following robust verification strategies MUST be used in the Python `gym.Env` wrapper to prevent desyncs and UI glitches:

1. **2-Consecutive-Frame Rule:** To eliminate 1-frame rendering glitches or UI transitions, a target screen state is ONLY considered valid if the RGB detection matches for **at least 2 consecutive checks**.
2. **Action Delay:** Always wait **1.0 second** immediately after sending an ADB/Scrcpy tap action before attempting to verify the next state. This ensures the Android UI has finished animating.
3. **State Verification Retries:** When checking an initial expected state (e.g. verifying we are on the Pause screen before clicking Rematch), retry up to **5 times** with a **0.3-second interval** before throwing an environment error.
4. **Transition Timeouts:** When waiting for an asynchronous transition (e.g., waiting for the Next Quarter screen to naturally appear), poll at `0.1s` intervals with an appropriately long timeout (e.g., `120s`).

## Resetting the Environment (`env.reset()`)

### Hard Reset (Snapshot Load)
Used initially or when the environment enters an unrecoverable/error state.
1. Run `adb emu avd snapshot load game_ready`.
2. Wait for Android boot completion (`adb shell getprop sys.boot_completed`).
3. Re-initialize the Scrcpy/TCP connections (snapshot loading breaks existing adb sockets).
4. Verify the emulator lands on the **Pause screen**.

### Soft Reset (In-Game Navigation)
Used to quickly reset the episode without rebuilding sockets. We only use Quarter 1 (Q1) for training.

1. **If on Next Quarter Screen:** Tap "Next Quarter Button" -> wait for Q2 Game Screen -> Tap "Pause Button" -> wait for Pause Screen -> Tap "Rematch Button" -> wait for Q1 Game Screen.
2. **If on Game Screen (Q1/Q2):** Tap "Pause Button" -> wait for Pause Screen -> Tap "Rematch Button" -> wait for Q1 Game Screen.
3. **If on Pause Screen:** Tap "Rematch Button" -> wait for Q1 Game Screen.

## Event-Based Score Tracking
To track the game score in real-time with zero latency and high reliability, we utilize an event-based state machine powered by OpenCV Template Matching (`cv2.matchTemplate` using `TM_CCOEFF_NORMED`).

### 1. Template Matching
Instead of doing Optical Character Recognition (OCR) on the scoreboard digits, we detect the dynamic popups that appear when a team scores. We crop a specific Region of Interest (ROI) at `y: 18-39, x: 243-338` (in the 584x268 layout) and match it against predefined templates:
- **Event Triggers**: `2_point.png`, `3_point.png`, `dunk.png`
- **Team Colors**: `blue.png`, `red.png`

We use a strict correlation threshold of `0.85` to ensure accurate detections.

### 2. State Machine Logic
The tracker operates in two states to reliably parse the game's popup sequences:

- **`NEUTRAL` State**: Continuously scans the ROI for an event trigger (`2_point`, `3_point`, or `dunk`). When a trigger is found, it transitions to `WAITING_COLOR` and stores the pending points (2 points for `2_point` and `dunk`; 3 points for `3_point`).
- **`WAITING_COLOR` State**: Continuously scans the ROI for a team color popup.
  - If `blue` is detected, the **Red (Left)** team scored. The pending points are added to the Red team's score, and the state reverts to `NEUTRAL`.
  - If `red` is detected, the **Blue (Right)** team scored. The pending points are added to the Blue team's score, and the state reverts to `NEUTRAL`.
  - **Timeout Strategy**: If 5.0 seconds elapse in this state without seeing a `blue` or `red` template, the state machine assumes the event was a false positive or the animation was skipped. It automatically aborts the pending points and reverts to `NEUTRAL`.