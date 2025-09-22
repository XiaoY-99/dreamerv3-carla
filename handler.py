import runpod

# Minimal handler that just echoes input back
def handler(event):
    return {"echo": event}

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
