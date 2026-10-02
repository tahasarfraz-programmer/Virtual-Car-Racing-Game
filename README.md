# Virtual Steering Racer

First-person cockpit racing game steered with both hands in front of your webcam
(same layout as the demo video: webcam "Virtual Steering Wheel" panel on the left, game on the right).

## Run
    pip install -r requirements.txt
    python racer.py              # camera 0
    python racer.py --camera 1   # another camera
    python racer.py --no-camera  # keyboard only

First launch with a recent mediapipe downloads a ~8 MB hand model next to racer.py (needs internet once).

## Controls
- Both hands up like you're gripping a wheel, tilt left/right -> steer (needs 2 hands visible)
- A/D or arrow keys: keyboard steering (automatic fallback when hands aren't seen)
- W: extra gas, S / Space: brake, R: restart, ESC: quit

Car auto-accelerates. Overtake traffic for +50, crashes cost -100 and slow you down.
Tips: good lighting, keep both hands in frame, tilt about 45 degrees for full lock.
Tweak ANGLE_MAX / DEADZONE at the top of racer.py to change sensitivity.
