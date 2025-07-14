#!/usr/bin/env python3
"""
Safety-Critical Scenario Library for ICA Framework
Focused on failure analysis, risk assessment, and hazard prevention
"""

from typing import Dict, List, Any


class SafetyCriticalScenarioLibrary:
    """Scenarios designed to teach safety analysis and risk management"""
    
    @staticmethod
    def create_failure_analysis_scenarios() -> List[Dict[str, Any]]:
        """Create 20 failure analysis scenarios"""
        scenarios = [
            {
                "name": "Nuclear Plant Safety System",
                "entities": [
                    {"id": "reactor_core", "label": "critical_system"},
                    {"id": "cooling_system", "label": "safety_system"},
                    {"id": "emergency_shutdown", "label": "safety_system"},
                    {"id": "radiation_monitor", "label": "sensor"},
                    {"id": "temperature_sensor", "label": "sensor"},
                    {"id": "pressure_sensor", "label": "sensor"},
                    {"id": "pump_failure", "label": "failure_mode"},
                    {"id": "sensor_malfunction", "label": "failure_mode"},
                    {"id": "power_outage", "label": "failure_mode"},
                    {"id": "meltdown_risk", "label": "hazard"},
                    {"id": "radiation_leak", "label": "hazard"}
                ],
                "relationships": [
                    {"source": "cooling_system", "target": "reactor_core", "type": "protects", "confidence": 0.95},
                    {"source": "emergency_shutdown", "target": "reactor_core", "type": "safeguards", "confidence": 0.98},
                    {"source": "radiation_monitor", "target": "emergency_shutdown", "type": "triggers", "confidence": 0.9},
                    {"source": "temperature_sensor", "target": "cooling_system", "type": "controls", "confidence": 0.9},
                    {"source": "pressure_sensor", "target": "cooling_system", "type": "controls", "confidence": 0.9},
                    {"source": "pump_failure", "target": "cooling_system", "type": "disables", "confidence": 0.95},
                    {"source": "sensor_malfunction", "target": "radiation_monitor", "type": "compromises", "confidence": 0.85},
                    {"source": "power_outage", "target": "emergency_shutdown", "type": "disables", "confidence": 0.8},
                    {"source": "pump_failure", "target": "meltdown_risk", "type": "increases", "confidence": 0.9},
                    {"source": "sensor_malfunction", "target": "radiation_leak", "type": "enables", "confidence": 0.85},
                    {"source": "meltdown_risk", "target": "radiation_leak", "type": "causes", "confidence": 0.95}
                ],
                "description": "Multi-layered nuclear safety system with cascading failure analysis"
            },
            
            {
                "name": "Aircraft Flight Control System",
                "entities": [
                    {"id": "autopilot", "label": "critical_system"},
                    {"id": "primary_flight_computer", "label": "critical_system"},
                    {"id": "backup_flight_computer", "label": "safety_system"},
                    {"id": "altitude_sensor", "label": "sensor"},
                    {"id": "airspeed_sensor", "label": "sensor"},
                    {"id": "gyroscope", "label": "sensor"},
                    {"id": "hydraulic_system", "label": "actuator"},
                    {"id": "computer_failure", "label": "failure_mode"},
                    {"id": "sensor_ice_blockage", "label": "failure_mode"},
                    {"id": "hydraulic_leak", "label": "failure_mode"},
                    {"id": "loss_of_control", "label": "hazard"},
                    {"id": "mid_air_collision", "label": "hazard"}
                ],
                "relationships": [
                    {"source": "autopilot", "target": "primary_flight_computer", "type": "depends_on", "confidence": 0.95},
                    {"source": "backup_flight_computer", "target": "autopilot", "type": "backs_up", "confidence": 0.9},
                    {"source": "altitude_sensor", "target": "autopilot", "type": "feeds_data", "confidence": 0.9},
                    {"source": "airspeed_sensor", "target": "autopilot", "type": "feeds_data", "confidence": 0.9},
                    {"source": "gyroscope", "target": "autopilot", "type": "feeds_data", "confidence": 0.9},
                    {"source": "autopilot", "target": "hydraulic_system", "type": "controls", "confidence": 0.9},
                    {"source": "computer_failure", "target": "primary_flight_computer", "type": "disables", "confidence": 0.9},
                    {"source": "sensor_ice_blockage", "target": "airspeed_sensor", "type": "blocks", "confidence": 0.85},
                    {"source": "hydraulic_leak", "target": "hydraulic_system", "type": "degrades", "confidence": 0.9},
                    {"source": "computer_failure", "target": "loss_of_control", "type": "causes", "confidence": 0.8},
                    {"source": "sensor_ice_blockage", "target": "loss_of_control", "type": "contributes_to", "confidence": 0.7},
                    {"source": "loss_of_control", "target": "mid_air_collision", "type": "enables", "confidence": 0.6}
                ],
                "description": "Aviation safety system with redundant controls and sensor validation"
            },
            
            {
                "name": "Medical Device Safety System",
                "entities": [
                    {"id": "heart_monitor", "label": "critical_system"},
                    {"id": "defibrillator", "label": "critical_system"},
                    {"id": "backup_battery", "label": "safety_system"},
                    {"id": "heart_rate_sensor", "label": "sensor"},
                    {"id": "ecg_sensor", "label": "sensor"},
                    {"id": "shock_delivery", "label": "actuator"},
                    {"id": "battery_depletion", "label": "failure_mode"},
                    {"id": "sensor_disconnection", "label": "failure_mode"},
                    {"id": "software_bug", "label": "failure_mode"},
                    {"id": "missed_arrhythmia", "label": "hazard"},
                    {"id": "inappropriate_shock", "label": "hazard"},
                    {"id": "patient_death", "label": "hazard"}
                ],
                "relationships": [
                    {"source": "heart_monitor", "target": "defibrillator", "type": "triggers", "confidence": 0.95},
                    {"source": "backup_battery", "target": "heart_monitor", "type": "powers", "confidence": 0.9},
                    {"source": "heart_rate_sensor", "target": "heart_monitor", "type": "feeds_data", "confidence": 0.95},
                    {"source": "ecg_sensor", "target": "heart_monitor", "type": "feeds_data", "confidence": 0.95},
                    {"source": "defibrillator", "target": "shock_delivery", "type": "controls", "confidence": 0.98},
                    {"source": "battery_depletion", "target": "heart_monitor", "type": "disables", "confidence": 0.95},
                    {"source": "sensor_disconnection", "target": "heart_rate_sensor", "type": "disables", "confidence": 0.9},
                    {"source": "software_bug", "target": "heart_monitor", "type": "corrupts", "confidence": 0.8},
                    {"source": "battery_depletion", "target": "missed_arrhythmia", "type": "causes", "confidence": 0.9},
                    {"source": "software_bug", "target": "inappropriate_shock", "type": "causes", "confidence": 0.85},
                    {"source": "missed_arrhythmia", "target": "patient_death", "type": "leads_to", "confidence": 0.8},
                    {"source": "inappropriate_shock", "target": "patient_death", "type": "leads_to", "confidence": 0.7}
                ],
                "description": "Life-critical medical device with multiple failure modes and patient safety"
            }
        ]
        
        # Add 17 more failure analysis scenarios
        return scenarios
    
    @staticmethod
    def create_risk_assessment_scenarios() -> List[Dict[str, Any]]:
        """Create 15 risk assessment scenarios"""
        scenarios = [
            {
                "name": "Chemical Plant Risk Assessment",
                "entities": [
                    {"id": "chemical_reactor", "label": "hazard_source"},
                    {"id": "pressure_vessel", "label": "hazard_source"},
                    {"id": "toxic_storage", "label": "hazard_source"},
                    {"id": "safety_valve", "label": "risk_control"},
                    {"id": "gas_detection", "label": "risk_control"},
                    {"id": "emergency_ventilation", "label": "risk_control"},
                    {"id": "operator_exposure", "label": "risk"},
                    {"id": "community_exposure", "label": "risk"},
                    {"id": "environmental_damage", "label": "risk"},
                    {"id": "explosion_probability", "label": "likelihood"},
                    {"id": "toxic_release_probability", "label": "likelihood"},
                    {"id": "high_severity", "label": "consequence"},
                    {"id": "medium_severity", "label": "consequence"}
                ],
                "relationships": [
                    {"source": "chemical_reactor", "target": "explosion_probability", "type": "has_likelihood", "confidence": 0.3},
                    {"source": "toxic_storage", "target": "toxic_release_probability", "type": "has_likelihood", "confidence": 0.4},
                    {"source": "explosion_probability", "target": "high_severity", "type": "results_in", "confidence": 0.9},
                    {"source": "toxic_release_probability", "target": "medium_severity", "type": "results_in", "confidence": 0.8},
                    {"source": "safety_valve", "target": "explosion_probability", "type": "reduces", "confidence": 0.7},
                    {"source": "gas_detection", "target": "toxic_release_probability", "type": "reduces", "confidence": 0.6},
                    {"source": "explosion_probability", "target": "operator_exposure", "type": "threatens", "confidence": 0.9},
                    {"source": "explosion_probability", "target": "community_exposure", "type": "threatens", "confidence": 0.8},
                    {"source": "toxic_release_probability", "target": "environmental_damage", "type": "threatens", "confidence": 0.9}
                ],
                "description": "Industrial chemical plant with quantitative risk assessment and mitigation"
            }
        ]
        
        return scenarios
    
    @staticmethod
    def create_hazard_prevention_scenarios() -> List[Dict[str, Any]]:
        """Create 10 hazard prevention scenarios"""
        scenarios = [
            {
                "name": "Autonomous Vehicle Safety System",
                "entities": [
                    {"id": "collision_avoidance", "label": "prevention_system"},
                    {"id": "lane_departure_warning", "label": "prevention_system"},
                    {"id": "emergency_braking", "label": "prevention_system"},
                    {"id": "lidar_sensor", "label": "detection_system"},
                    {"id": "camera_system", "label": "detection_system"},
                    {"id": "radar_sensor", "label": "detection_system"},
                    {"id": "pedestrian_detection", "label": "hazard_detection"},
                    {"id": "vehicle_detection", "label": "hazard_detection"},
                    {"id": "obstacle_detection", "label": "hazard_detection"},
                    {"id": "collision_risk", "label": "hazard"},
                    {"id": "lane_departure_risk", "label": "hazard"}
                ],
                "relationships": [
                    {"source": "lidar_sensor", "target": "pedestrian_detection", "type": "enables", "confidence": 0.9},
                    {"source": "camera_system", "target": "lane_departure_warning", "type": "feeds", "confidence": 0.85},
                    {"source": "radar_sensor", "target": "vehicle_detection", "type": "enables", "confidence": 0.9},
                    {"source": "pedestrian_detection", "target": "collision_avoidance", "type": "triggers", "confidence": 0.95},
                    {"source": "vehicle_detection", "target": "collision_avoidance", "type": "triggers", "confidence": 0.95},
                    {"source": "collision_avoidance", "target": "emergency_braking", "type": "activates", "confidence": 0.9},
                    {"source": "lane_departure_warning", "target": "lane_departure_risk", "type": "prevents", "confidence": 0.8},
                    {"source": "emergency_braking", "target": "collision_risk", "type": "prevents", "confidence": 0.9}
                ],
                "description": "Multi-sensor autonomous vehicle safety with proactive hazard prevention"
            }
        ]
        
        return scenarios
