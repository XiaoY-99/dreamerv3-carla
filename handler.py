import runpod
import subprocess

def handler(event):
    steps = event.get("input", {}).get("steps", 10)
    batch_size = event.get("input", {}).get("batch_size", 4)

    # Run your training script with arguments
    cmd = [
        "python", "train_carla.py",
        f"--steps={steps}",
        f"--batch_size={batch_size}"
    ]
    try:
        subprocess.run(cmd, check=True)
        return {"status": "success", "steps": steps, "batch_size": batch_size}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
