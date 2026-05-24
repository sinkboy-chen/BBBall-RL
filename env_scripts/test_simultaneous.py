#!/usr/bin/env python3
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bbball_env import BBBallEnv

def main():
    print("[*] Initializing Env 1 on emulator-5554...")
    env1 = BBBallEnv(device_serial="emulator-5554", scrcpy_port=27183)
    
    print("[*] Initializing Env 2 on emulator-5556...")
    env2 = BBBallEnv(device_serial="emulator-5556", scrcpy_port=27184)
    
    print("[*] Both environments connected! Resetting Env 1...")
    obs1, info1 = env1.reset()
    print(f"Env 1 reset success! Image shape: {obs1['image'].shape}, Last Action: {obs1['last_action']}")
    
    print("[*] Resetting Env 2...")
    obs2, info2 = env2.reset()
    print(f"Env 2 reset success! Image shape: {obs2['image'].shape}, Last Action: {obs2['last_action']}")
    
    print("[*] Running 10 parallel steps...")
    for i in range(100):
        # Step Env 1
        obs1, reward1, term1, trunc1, info1 = env1.step(1 if i % 2 == 0 else 0)
        # Step Env 2
        obs2, reward2, term2, trunc2, info2 = env2.step(0 if i % 2 == 0 else 1)
        
        print(f"Step {i+1:02d} | Env 1 Reward: {reward1} | Env 2 Reward: {reward2}")
        time.sleep(0.05)
        
    print("[*] Closing environments...")
    env1.close()
    env2.close()
    print("[*] Success! Parallel emulators work perfectly.")

if __name__ == "__main__":
    main()
