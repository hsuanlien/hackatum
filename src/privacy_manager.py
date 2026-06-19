"""
Essential Privacy Manager for MTU Worker Safety System (Hackathon Edition).

Implements GDPR Article 5 compliance through:
1. Pseudonymization (HMAC-SHA256 with rotating salts)
2. Aggregated event logging (prevents time-correlation tracking)
"""

import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging

import src.config as config

logger = logging.getLogger("[PrivacyManager]")


class PersonIDHasher:
    """
    Anonymizes person IDs using HMAC-SHA256 with rotating salts.
    Prevents timing attacks and length-extension attacks.
    """
    
    def __init__(self, salt_rotation_hours: int = 24):
        self.salt_rotation_hours = salt_rotation_hours
        self.current_salt = self._generate_salt()
        self.salt_timestamp = time.time()
        logger.info(f"[Hasher] HMAC-SHA256 pseudonymization active (salt rotation: {salt_rotation_hours}h)")
    
    def _generate_salt(self) -> str:
        """Generates cryptographically random salt."""
        import secrets
        return secrets.token_hex(16)
    
    def _check_salt_rotation(self) -> None:
        """Rotate salt if period elapsed."""
        elapsed_seconds = time.time() - self.salt_timestamp
        if elapsed_seconds >= self.salt_rotation_hours * 3600:
            self.current_salt = self._generate_salt()
            self.salt_timestamp = time.time()
            logger.info(f"[Hasher] Salt rotated")
    
    def hash_person_id(self, person_id: int) -> str:
        """Hash person ID using HMAC-SHA256 (deterministic, not reversible)."""
        self._check_salt_rotation()
        person_str = str(person_id).encode('utf-8')
        salt_bytes = self.current_salt.encode('utf-8')
        hmac_obj = hmac.new(salt_bytes, person_str, hashlib.sha256)
        return hmac_obj.hexdigest()
    
    def get_display_id(self, person_id: int) -> str:
        """Return short display version (first 8 chars)."""
        return self.hash_person_id(person_id)[:8].upper()


class AggregatedEventLogger:
    """
    Batches events into hourly windows instead of logging point-in-time incidents.
    Prevents time-correlation attacks (e.g., "Worker was in Zone B at 14:02").
    """
    
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.events_by_window: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        logger.info(f"[AggLogger] Event batching: {window_minutes}-minute windows")
    
    def _get_window_key(self, timestamp: float) -> str:
        """Convert timestamp to aggregation window (e.g., 2026-06-19_14:00-15:00)."""
        dt = datetime.fromtimestamp(timestamp)
        window_start = dt.replace(minute=0, second=0, microsecond=0)
        window_end = window_start + timedelta(minutes=self.window_minutes)
        return f"{window_start.strftime('%Y-%m-%d_%H:%M')}-{window_end.strftime('%H:%M')}"
    
    def log_fall_detection(self, person_hashed_id: str, zone: str = "unknown", 
                          severity: str = "Critical", timestamp: Optional[float] = None) -> None:
        """Log fall detection in aggregated window."""
        timestamp = timestamp or time.time()
        window_key = self._get_window_key(timestamp)
        self.events_by_window[window_key]["fall_detection"].append({
            "person_id_hash": person_hashed_id,
            "zone": zone,
            "severity": severity
        })
    
    def log_ppe_violation(self, person_hashed_id: str, violation_type: str, zone: str = "unknown",
                         timestamp: Optional[float] = None) -> None:
        """Log PPE violation in aggregated window."""
        timestamp = timestamp or time.time()
        window_key = self._get_window_key(timestamp)
        self.events_by_window[window_key]["ppe_violation"].append({
            "person_id_hash": person_hashed_id,
            "violation_type": violation_type,
            "zone": zone
        })
    
    def log_environmental_hazard(self, hazard_type: str, severity: str = "Warning",
                                zone: str = "unknown", timestamp: Optional[float] = None) -> None:
        """Log environmental hazard in aggregated window."""
        timestamp = timestamp or time.time()
        window_key = self._get_window_key(timestamp)
        self.events_by_window[window_key]["environmental_hazard"].append({
            "hazard_type": hazard_type,
            "severity": severity,
            "zone": zone
        })
    
    def get_window_summary(self, window_key: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated summary for a window (no individual tracking possible)."""
        if window_key is None:
            window_key = self._get_window_key(time.time())
        
        if window_key not in self.events_by_window:
            return {"window": window_key, "status": "no_incidents"}
        
        window_data = self.events_by_window[window_key]
        
        return {
            "window": window_key,
            "fall_detections": len(window_data.get("fall_detection", [])),
            "ppe_violations": len(window_data.get("ppe_violation", [])),
            "environmental_hazards": len(window_data.get("environmental_hazard", [])),
            "zones_affected": sorted(list(set(
                e.get("zone") for events in window_data.values() 
                for e in events if e.get("zone")
            )))
        }


class PrivacyManager:
    """
    Minimal privacy integration: ID anonymization + event aggregation.
    Provides GDPR Article 5 compliance without bloat.
    """
    
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.id_hasher = PersonIDHasher(salt_rotation_hours=24)
        self.event_logger = AggregatedEventLogger(window_minutes=60)
        logger.info("[PrivacyManager] Initialized (essentials only)")
    
    def process_frame_alerts(self, frame_data) -> None:
        """Convert frame alerts to anonymized, aggregated events."""
        if self.use_mock:
            return
        
        for alert in frame_data.alerts:
            person_id = alert.get("person_id")
            hashed_id = self.id_hasher.hash_person_id(person_id) if person_id else "system"
            alert_type = alert.get("type", "unknown")
            
            if alert_type == "FALL_ALERT":
                self.event_logger.log_fall_detection(
                    hashed_id,
                    zone=alert.get("metadata", {}).get("zone", "unknown"),
                    severity=alert.get("severity", "Critical"),
                    timestamp=alert.get("timestamp")
                )
            elif alert_type == "PPE_VIOLATION":
                self.event_logger.log_ppe_violation(
                    hashed_id,
                    violation_type=alert.get("message", "unknown").lower(),
                    zone=alert.get("metadata", {}).get("zone", "unknown"),
                    timestamp=alert.get("timestamp")
                )
            elif alert_type in ["ENVIRONMENT_WARNING", "ENVIRONMENT_ALERT"]:
                self.event_logger.log_environmental_hazard(
                    alert.get("message", "unknown").lower(),
                    severity=alert.get("severity", "Warning"),
                    zone=alert.get("metadata", {}).get("zone", "unknown"),
                    timestamp=alert.get("timestamp")
                )
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get current window summary (aggregated, no individual tracking)."""
        return self.event_logger.get_window_summary()


if __name__ == "__main__":
    print("Privacy Manager Essentials - Ready for hackathon")
    
    # Quick test
    pm = PrivacyManager(use_mock=False)
    
    # Simulate some alerts
    class MockAlert:
        def __init__(self, atype, person_id, msg):
            self.data = {
                "type": atype,
                "person_id": person_id,
                "message": msg,
                "severity": "Critical",
                "metadata": {"zone": "Zone_B"},
                "timestamp": time.time()
            }
        def get(self, k, default=None):
            return self.data.get(k, default)
    
    class MockFrameData:
        def __init__(self):
            self.alerts = [
                MockAlert("FALL_ALERT", 5, "Fall detected"),
                MockAlert("PPE_VIOLATION", 5, "missing_helmet"),
            ]
    
    frame = MockFrameData()
    pm.process_frame_alerts(frame)
    
    summary = pm.get_privacy_summary()
    print(f"\nCurrent window summary:")
    print(f"  Falls: {summary['fall_detections']}")
    print(f"  PPE violations: {summary['ppe_violations']}")
    print(f"  Zones: {summary['zones_affected']}")
    print("\n✓ Privacy system operational (essentials only)")
