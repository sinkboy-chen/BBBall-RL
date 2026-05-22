## DRL Final Project Env Setup on CSIE Workstation

1. The repo will be cloned to ~/Desktop/BBBall-RL.
2. The workspace will be /tmp2/$USER/DRL_final_workspace.

### 0. Git clone
```bash
cd ~/Desktop
git clone https://github.com/sinkboy-chen/BBBall-RL.git
```

### 1. Create directories
```bash
mkdir -p /tmp2/$USER/DRL_final_workspace/android-sdk/cmdline-tools
cd /tmp2/$USER/DRL_final_workspace/android-sdk/cmdline-tools
```

### 2. Download & unzip command line tools
Go to https://developer.android.com/studio/index.html#command-line-tools-only and find the latest Linux URL, then:
```bash
wget https://dl.google.com/android/repository/commandlinetools-linux-14742923_latest.zip
unzip commandlinetools-linux-14742923_latest.zip
mv cmdline-tools latest
```

### 3. Set environment variables
```bash
export ANDROID_HOME=/tmp2/$USER/DRL_final_workspace/android-sdk
export ANDROID_SDK_ROOT=$ANDROID_HOME
export ANDROID_USER_HOME=/tmp2/$USER/DRL_final_workspace/.android
export ANDROID_AVD_HOME=/tmp2/$USER/DRL_final_workspace/.android/avd
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator
export ANDROID_EMULATOR_HOME=/tmp2/$USER/DRL_final_workspace/.android
```

### 4. Create cache and AVD directories
```bash
mkdir -p $ANDROID_USER_HOME/cache
mkdir -p $ANDROID_AVD_HOME
```

### 5. Accept licenses
```bash
sdkmanager --licenses
```
Press `y` for all prompts.

### 6. Install SDK packages (1 mins)
```bash
sdkmanager \
  "platform-tools" \
  "platforms;android-31" \
  "system-images;android-31;google_apis;x86_64" \
  "emulator"
```

### 7. Create AVD (Pixel 5, API 31)
```bash
avdmanager create avd \
  -n pixel5_api31 \
  -k "system-images;android-31;google_apis;x86_64" \
  -d "pixel_5"
```

### 8. First launch — cold boot (one time only, takes 5 mins)
```bash
emulator -avd pixel5_api31 \
  -no-window \
  -no-audio \
  -no-boot-anim \
  -gpu swiftshader_indirect \
  -no-metrics &
```

**Run this, then open a new terminal to run the next step. It's okay if this step produces warnings or errors; ignore them.**

### 9. Install your APK

Open another terminal, run **Step3: Set environment variables** first then

```bash
until adb shell getprop sys.boot_completed 2>/dev/null | grep -q "1" && \
      adb shell service check package 2>/dev/null | grep -q "found"; do
  echo "Waiting for package manager..."; sleep 5
done
echo "Ready! Installing..."
adb install ~/Desktop/BBBall-RL/assets/Bouncy_Basketball.apk
```


### 10. Save snapshot (one time only, after APK installed)
```bash
adb emu avd snapshot save clean_boot
```
Snapshot is saved to:
`$ANDROID_AVD_HOME/pixel5_api31.avd/snapshots/clean_boot/`

Kill the emulator after saving:
```bash
adb emu kill
```

### 11. Subsequent launches — fast boot from snapshot (~10 sec)

ensure **Step3: Set environment variables** is executed

```bash
emulator -avd pixel5_api31 \
  -no-window \
  -no-audio \
  -no-boot-anim \
  -gpu swiftshader_indirect \
  -no-metrics \
  -snapshot clean_boot &
```

Wait for boot:
```bash
until adb shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; do
  echo "Waiting for boot..."; sleep 3
done
echo "Device ready!"
```