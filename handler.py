import runpod
import subprocess


def handler(event):
    """
    RunPod will pass an event dict with `input`.
    Example input:
    {
      "input": {
        "mode": "train",
        "steps": 1000
      }
    }
    """
    mode = event.get("input", {}).get("mode", "train")
    steps = event.get("input", {}).get("steps", 1000)

    try:
        if mode == "train":
            subprocess.run(
                ["python", "train_carla.py", f"--steps={steps}"],
                check=True
            )
            return {"status": "success", "message": f"Training ran for {steps} steps"}

        elif mode == "eval":
            subprocess.run(["python", "eval_carla.py"], check=True)
            return {"status": "success", "message": "Evaluation complete"}

        else:
            return {"status": "error", "message": f"Unknown mode {mode}"}

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
