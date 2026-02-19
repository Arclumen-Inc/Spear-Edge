import time
from typing import Optional, Dict


class EdgeTaskEngine:
    """
    Decides which Tripwire events become Edge capture tasks.
    ONE active task at a time.
    """

    def __init__(self):
        self.active_task: Optional[Dict] = None
        self.last_task_end = 0.0

        # Tunables
        self.min_task_time = 3.0        # seconds task must run
        self.cooldown_time = 1.5        # seconds after task end
        self.freq_hysteresis_hz = 250e3 # treat nearby freqs as same task

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def consider_event(self, ev: Dict) -> Optional[Dict]:
        """
        Decide whether a Tripwire event should create / replace an Edge task.
        Returns a NEW task dict, or None.
        """

        now = time.time()

        # Must have a frequency
        freq_hz = ev.get("freq_hz")
        if not freq_hz:
            return None

        # Cooldown gate
        if now - self.last_task_end < self.cooldown_time:
            return None

        confidence = float(ev.get("confidence", 0.0))
        priority = int(confidence * 100)

        new_task = {
            "source_node": ev.get("node_id"),
            "freq_hz": float(freq_hz),
            "scan_plan": ev.get("scan_plan"),
            "priority": priority,
            "intent": self._infer_intent(ev),
            "created_ts": now,
            "status": "pending",
        }

        # No active task → accept
        if not self.active_task:
            return new_task

        # Too soon to replace
        age = now - self.active_task.get("created_ts", 0.0)
        if age < self.min_task_time:
            return None

        # Same frequency (within hysteresis) → ignore
        if self._is_same_frequency(freq_hz, self.active_task.get("freq_hz")):
            return None

        # Higher priority replaces
        if priority > self.active_task.get("priority", 0):
            return new_task

        return None

    def start_task(self, task: Dict) -> None:
        task["status"] = "active"
        task["start_ts"] = time.time()
        self.active_task = task

    def finish_task(self, reason: str = "completed") -> None:
        if not self.active_task:
            return

        self.active_task["status"] = reason
        self.active_task["end_ts"] = time.time()

        self.last_task_end = self.active_task["end_ts"]
        self.active_task = None

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _is_same_frequency(self, f1: float, f2: Optional[float]) -> bool:
        if not f2:
            return False
        return abs(float(f1) - float(f2)) <= self.freq_hysteresis_hz

    def _infer_intent(self, ev: Dict) -> str:
        """
        Decide what Edge should DO with this task.
        """
        scan_plan = ev.get("scan_plan", "")

        if scan_plan.startswith("fhss"):
            return "track"

        if "video" in scan_plan:
            return "capture_iq"

        if ev.get("classification") in ("rf_jammer", "rf_wideband"):
            return "capture_iq"

        return "observe"
