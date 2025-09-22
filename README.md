# Workflow
<pre> ```
+---------------------+                  +----------------------+
|      CARLA Sim      |                  |    DreamerV3 Agent   |
|---------------------|                  |----------------------|
| Sensors:            |                  | World Model:         |
|  - RGB Camera(s) --->-- Observation -->|  - Encoder (CNN)     |
|  - Lidar (opt.)     |                  |  - Recurrent state   |
|  - Vehicle state    |                  |  - Latent dynamics   |
|                     |                  |                      |
| Reward function ---->-- Reward ------->| Reward predictor     |
|                     |                  |                      |
| Vehicle dynamics <--- Action ----------| Policy (actor)       |
|                     |                  | Value (critic)       |
+---------------------+                  +----------------------+

                           ^
                           |
                           v
                  (Latent imagination rollouts)

``` </pre>

# Install all necessary packages
pip install -r requirements.txt

# Test phase
1. Start Carla in sync mode (under carla/ root dir):
    ./CarlaUE4.sh -quality-level=Low -RenderOffScreen -carla-rpc-port=2000
2. Quick smoke with the test_train (under root dir of this repo, better in a virtual env):
    python dreamerv3/embodies/tests/test_train.py

# Training loop
1. python train_carla.py   -- including training and validation phase
* If jax doesn't work, unset the tags:
    unset JAX_PLATFORMS
    unset XLA_FLAGS
2. TensorBoard visualization:
    tensorboard --logdir logs_carla/tb --port 6006

3. Current checkpoint: still highly oscillating actions and rewards; there should be some improved behaviors like:
    - train/steps 1.7e4 / train/batch_size 16 / env/reward -0.51 / env/lat_err_raw 0.06 / env/head_err_raw -8.2e-4 / env/progress_raw 0.06 / env/damage_raw 0 / env/on_track_raw 1 / env/progress_m 47.67    ...    episode/length 135 / episode/mean_speed 3.84 / episode/max_speed 10.41
   But more training steps are necessary
