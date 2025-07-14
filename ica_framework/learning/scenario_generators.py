"""
Scenario Generators for ICA Framework Learning
Enhanced scenario generation for diverse AGI training
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional


class PhysicsSimulation:
    """Enhanced physics simulation for realistic learning scenarios"""
    
    def __init__(self):
        self.entities = self._initialize_physics_entities()
        self.relations = self._initialize_physics_relations()
        self.environment_complexity = 1.0
        self.complexity_growth_rate = 0.001
        
    def _initialize_physics_entities(self) -> List[str]:
        """Initialize enhanced physics entities for complex simulations"""
        return [
            # Physical objects with properties
            'sphere_red_large', 'sphere_blue_small', 'cube_green_heavy', 'cube_yellow_light',
            'cylinder_orange_tall', 'pyramid_purple_sharp', 'platform_gray_stable',
            'box_wooden_hollow', 'ball_metal_dense', 'rod_plastic_flexible',
            
            # Physics forces and concepts
            'gravity_force', 'friction_surface', 'temperature_ambient', 'pressure_atmospheric',
            'energy_kinetic', 'energy_potential', 'momentum_linear', 'momentum_angular',
            'acceleration_downward', 'velocity_horizontal', 'mass_distribution',
            
            # Robotic and automation elements
            'robot_arm_precise', 'conveyor_belt_moving', 'sensor_array_visual',
            'human_operator_skilled', 'automated_system_fast', 'actuator_servo',
            'motor_electric', 'controller_pid', 'feedback_loop',
            
            # Environmental dynamics
            'lighting_system_adaptive', 'weather_conditions_variable', 'magnetic_field_uniform',
            'electric_field_static', 'chemical_solution_reactive', 'airflow_turbulent',
            'vibration_mechanical', 'noise_acoustic', 'radiation_thermal',
            
            # Emergent properties
            'pattern_repetitive', 'chaos_deterministic', 'equilibrium_dynamic',
            'resonance_frequency', 'phase_transition', 'symmetry_broken',
            'emergence_collective', 'feedback_positive', 'feedback_negative'
        ]
        
    def _initialize_physics_relations(self) -> List[str]:
        """Initialize enhanced physics relations"""
        return [
            # Basic causal relations
            'causes', 'affects', 'influences', 'prevents', 'enables',
            'triggers', 'inhibits', 'amplifies', 'dampens', 'modulates',
            
            # Physics-specific relations
            'accelerates', 'decelerates', 'rotates', 'vibrates', 'resonates',
            'heats', 'cools', 'conducts', 'insulates', 'reflects',
            'refracts', 'absorbs', 'emits', 'transfers', 'converts',
            
            # Mechanical relations
            'supports', 'lifts', 'compresses', 'extends', 'bends',
            'twists', 'slides', 'rolls', 'bounces', 'collides',
            
            # Systems relations
            'controls', 'monitors', 'regulates', 'stabilizes', 'optimizes',
            'coordinates', 'synchronizes', 'calibrates', 'adjusts'
        ]
    
    def generate_physics_scenario(self, complexity_level: Optional[float] = None) -> Dict[str, Any]:
        """Generate a physics-based learning scenario"""
        if complexity_level is None:
            complexity_level = self.environment_complexity
            
        # Adaptive scenario complexity
        num_entities = max(3, int(5 + complexity_level * 3))
        num_relations = max(2, int(3 + complexity_level * 2))
        
        # Select entities based on complexity
        available_entities = self.entities[:int(len(self.entities) * min(1.0, complexity_level / 2))]
        scenario_entities = np.random.choice(available_entities, 
                                           size=min(num_entities, len(available_entities)), 
                                           replace=False)
        
        # Generate physics-based relationships
        relationships = []
        for _ in range(num_relations):
            source = np.random.choice(scenario_entities)
            target = np.random.choice(scenario_entities)
            
            if source != target:
                relation = np.random.choice(self.relations)
                confidence = np.random.uniform(0.6, 0.95)  # Physics has higher confidence
                
                relationships.append({
                    "source": source,
                    "target": target,
                    "type": relation,
                    "confidence": confidence,
                    "physics_based": True,
                    "complexity_level": complexity_level
                })
        
        # Increase complexity gradually
        self.environment_complexity += self.complexity_growth_rate
        
        return {
            "entities": [{"id": entity, "label": "physics_object"} for entity in scenario_entities],
            "relationships": relationships,
            "state": np.random.normal(0, 0.1, 20),  # Enhanced state representation
            "scenario_type": "physics_simulation",
            "complexity": complexity_level
        }


class ProceduralScenarioGenerator:
    """Generate procedural learning scenarios with known patterns"""
    
    def __init__(self):
        self.scenario_types = [
            "smart_home_automation",
            "industrial_robotics", 
            "autonomous_vehicles",
            "supply_chain_optimization",
            "energy_management",
            "environmental_monitoring"
        ]
        self.generated_patterns = []
        
    def generate_smart_home_scenario(self) -> Dict[str, Any]:
        """Generate smart home automation scenarios"""
        devices = [
            'smart_thermostat', 'motion_sensor', 'door_lock', 'security_camera',
            'light_controller', 'air_purifier', 'smart_speaker', 'energy_monitor'
        ]
        
        scenario_devices = np.random.choice(devices, size=np.random.randint(3, 6), replace=False)
        
        # Create automation patterns
        relationships = []
        
        # Motion-based automations
        if 'motion_sensor' in scenario_devices:
            for device in scenario_devices:
                if device != 'motion_sensor' and device in ['light_controller', 'security_camera']:
                    relationships.append({
                        "source": "motion_sensor",
                        "target": device,
                        "type": "triggers",
                        "confidence": 0.9,
                        "automation_rule": "motion_detection"
                    })
        
        # Energy efficiency patterns
        if 'smart_thermostat' in scenario_devices and 'energy_monitor' in scenario_devices:
            relationships.append({
                "source": "energy_monitor",
                "target": "smart_thermostat",
                "type": "optimizes",
                "confidence": 0.85,
                "automation_rule": "energy_saving"
            })
        
        return {
            "entities": [{"id": device, "label": "smart_device"} for device in scenario_devices],
            "relationships": relationships,
            "state": np.random.normal(0, 0.1, 15),
            "scenario_type": "smart_home_automation",
            "automation_context": "residential"
        }
    
    def generate_industrial_scenario(self) -> Dict[str, Any]:
        """Generate industrial robotics scenarios"""
        equipment = [
            'robotic_arm', 'conveyor_system', 'quality_sensor', 'assembly_station',
            'material_handler', 'safety_system', 'production_controller', 'inventory_tracker'
        ]
        
        scenario_equipment = np.random.choice(equipment, size=np.random.randint(4, 7), replace=False)
        
        relationships = []
        
        # Production line relationships
        production_flow = ['material_handler', 'assembly_station', 'quality_sensor', 'conveyor_system']
        active_flow = [eq for eq in production_flow if eq in scenario_equipment]
        
        for i in range(len(active_flow) - 1):
            relationships.append({
                "source": active_flow[i],
                "target": active_flow[i + 1],
                "type": "feeds_into",
                "confidence": 0.95,
                "production_stage": i + 1
            })
        
        # Safety and control relationships
        if 'safety_system' in scenario_equipment:
            for eq in scenario_equipment:
                if eq != 'safety_system':
                    relationships.append({
                        "source": "safety_system",
                        "target": eq,
                        "type": "monitors",
                        "confidence": 0.9,
                        "safety_critical": True
                    })
        
        return {
            "entities": [{"id": eq, "label": "industrial_equipment"} for eq in scenario_equipment],
            "relationships": relationships,
            "state": np.random.normal(0, 0.1, 25),
            "scenario_type": "industrial_robotics",
            "production_context": "manufacturing"
        }
    
    def generate_scenario(self, scenario_type: Optional[str] = None) -> Dict[str, Any]:
        """Generate a scenario of specified or random type"""
        if scenario_type is None:
            scenario_type = np.random.choice(self.scenario_types)
        
        if scenario_type == "smart_home_automation":
            return self.generate_smart_home_scenario()
        elif scenario_type == "industrial_robotics":
            return self.generate_industrial_scenario()
        else:
            # Fallback to original IoT scenario
            return self._generate_basic_scenario()
    
    def _generate_basic_scenario(self) -> Dict[str, Any]:
        """Fallback basic scenario generation"""
        devices = ['sensor', 'actuator', 'controller', 'display', 'network']
        scenario_devices = np.random.choice(devices, size=3, replace=False)
        
        relationships = [{
            "source": scenario_devices[0],
            "target": scenario_devices[1], 
            "type": "controls",
            "confidence": 0.8
        }]
        
        return {
            "entities": [{"id": device, "label": "device"} for device in scenario_devices],
            "relationships": relationships,
            "state": np.random.normal(0, 0.1, 10),
            "scenario_type": "basic_iot"
        }
