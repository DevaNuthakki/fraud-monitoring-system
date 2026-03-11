import os
import time
import subprocess
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DRIFT_SCRIPT = os.path.join(BASE_DIR, "monitoring", "generate_evidently_drift.py")

# ===============================
# Drift Job
# ===============================

def run_drift_monitoring():
    print("\n===============================")
    print("Running drift monitoring job")
    print("Time:", datetime.utcnow())
    print("===============================")

    try:
        subprocess.run(["python", DRIFT_SCRIPT], check=True)
        print("Drift monitoring completed successfully")

    except Exception as e:
        print("Drift monitoring failed:", str(e))


# ===============================
# Scheduler
# ===============================

scheduler = BackgroundScheduler()

scheduler.add_job(
    run_drift_monitoring,
    trigger="interval",
    hours=12,
    id="drift_monitor_job",
    replace_existing=True
)

scheduler.start()

print("Background monitoring scheduler started")
print("Drift monitoring will run every 12 hours")

# Keep process alive
try:
    while True:
        time.sleep(60)
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()