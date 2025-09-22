import subprocess

def handler(event):
    """
    RunPod calls this function with an `event` dict.
    You return a dict with the results.
    """
    # Example: start training with your existing training script
    try:
        subprocess.run(
            ["python", "train_carla.py"],  # <-- replace with your entry script
            check=True
        )
        return {"status": "success", "message": "Training finished"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": str(e)}
