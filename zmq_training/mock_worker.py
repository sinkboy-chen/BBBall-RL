#!/usr/bin/env python3
import os
import sys
import time
import io
import argparse
import threading
import numpy as np
import zmq

def run_mock_env(workstation_id, rank, master_ip, push_port, mode, round_idx, trigger_fail=False):
    """Simulates a single virtual emulator running a rollout or evaluation."""
    device_serial = f"emulator-{5554 + 2 * rank}"
    
    zmq_context = zmq.Context()
    push_socket = zmq_context.socket(zmq.PUSH)
    push_socket.connect(f"tcp://{master_ip}:{push_port}")
    
    # Small random delay to simulate staggered completion
    time.sleep(np.random.uniform(0.5, 3.0))
    
    try:
        if trigger_fail:
            raise Exception("Mock ADB Connection Timeout Error")
            
        if mode == "evaluate":
            # Push evaluation mock result
            push_socket.send_pyobj({
                "event": "eval_report",
                "workstation": workstation_id,
                "emulator": device_serial,
                "status": "success",
                "round": round_idx,
                "reward": float(np.random.uniform(10.0, 50.0)),
                "length": int(np.random.randint(150, 450))
            })
            print(f"[{workstation_id} - {device_serial}] Pushed mock EVAL report.")
        else:
            # Generate mock observations matching spaces for exactly 1 episode
            T = int(np.random.randint(150, 450)) # dynamic episode length
            mock_images = np.random.randint(0, 256, size=(T, 150, 303, 4), dtype=np.uint8)
            mock_last_actions = np.random.randint(0, 2, size=(T,), dtype=np.int64)
            mock_actions = np.random.randint(0, 2, size=(T, 1), dtype=np.int64)
            mock_rewards = np.random.uniform(-1.0, 2.0, size=(T,)).astype(np.float32)
            
            # First transition is the start of the episode
            episode_starts = np.zeros(T, dtype=bool)
            episode_starts[0] = True
            
            push_socket.send_pyobj({
                "event": "env_report",
                "workstation": workstation_id,
                "emulator": device_serial,
                "status": "success",
                "round": round_idx,
                "data": {
                    "observations": {
                        "image": mock_images,
                        "last_action": mock_last_actions
                    },
                    "actions": mock_actions,
                    "rewards": mock_rewards,
                    "episode_starts": episode_starts,
                    "episode_reward": float(np.sum(mock_rewards))
                }
            })
            print(f"[{workstation_id} - {device_serial}] Pushed mock {T}-step EPISODE ROLLOUT.")
            
    except Exception as e:
        push_socket.send_pyobj({
            "event": "env_report" if mode == "train" else "eval_report",
            "workstation": workstation_id,
            "emulator": device_serial,
            "status": "failed",
            "round": round_idx,
            "reason": str(e)
        })
        print(f"[{workstation_id} - {device_serial}] Pushed mock FAILURE report.")
    finally:
        push_socket.close()
        zmq_context.term()

def simulate_workstation(workstation_id, master_ip, rep_port, push_port, round_idx, fail_env_idx=None):
    """Simulates a single workstation that requests weights and spawns 3 threads."""
    zmq_context = zmq.Context()
    req_socket = zmq_context.socket(zmq.REQ)
    req_socket.connect(f"tcp://{master_ip}:{rep_port}")
    
    # Request model instructions
    req_socket.send_pyobj({
        "event": "get_model",
        "workstation": workstation_id,
        "round": round_idx
    })
    
    reply = req_socket.recv_pyobj()
    req_socket.close()
    zmq_context.term()
    
    mode = reply.get("mode", "train")
    master_round = reply.get("round", round_idx)
    
    # Spawn 3 environment threads
    threads = []
    for rank in range(3):
        trigger_fail = (fail_env_idx is not None and rank == fail_env_idx)
        t = threading.Thread(
            target=run_mock_env,
            args=(workstation_id, rank, master_ip, push_port, mode, master_round, trigger_fail)
        )
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()

def main():
    parser = argparse.ArgumentParser(description="Mock Worker Node Simulator for ZMQ Distributed System")
    parser.add_argument("--master-ip", type=str, default="127.0.0.1", help="GPU master address")
    parser.add_argument("--rep-port", type=int, default=5555, help="Port of master ROUTER socket")
    parser.add_argument("--push-port", type=int, default=5556, help="Port of master PULL socket")
    parser.add_argument("--num-workers", type=int, default=9, help="Number of workstations to simulate")
    parser.add_argument("--round", type=int, default=0, help="Simulate a specific round")
    args = parser.parse_args()
    
    print(f"[*] Starting Mock Worker node pool (simulating {args.num_workers} workstations, 27 envs)...")
    
    workstation_threads = []
    for i in range(args.num_workers):
        workstation_id = f"mock-workstation-{182 + i}"
        
        # Inject deliberate simulated environment failures
        # Workstation 2 has emulator 1 fail, Workstation 5 has emulator 2 fail.
        # This will result in exactly 25 successes and 2 failures, triggering grace timers!
        fail_env_idx = None
        if i == 2:
            fail_env_idx = 1
        elif i == 5:
            fail_env_idx = 2
            
        t = threading.Thread(
            target=simulate_workstation,
            args=(workstation_id, args.master_ip, args.rep_port, args.push_port, args.round, fail_env_idx)
        )
        workstation_threads.append(t)
        t.start()
        
    for t in workstation_threads:
        t.join()
        
    print("[*] All mock worker workstation threads completed successfully.")

if __name__ == "__main__":
    main()
