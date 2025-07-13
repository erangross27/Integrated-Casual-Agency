"""
Continuous Learning Script for ICA Framework
This script provides continuous learning with progress tracking and resume capabilities.
"""

import numpy as np
import networkx as nx
from pathlib import Path
import json
import time
import signal
import sys
from datetime import datetime
import logging
import warnings
import os

# AGGRESSIVE logging suppression - completely silence everything
logging.disable(logging.CRITICAL)  # Disable all logging completely
os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings
warnings.filterwarnings("ignore")

# Suppress specific loggers that might still get through
for logger_name in ['urllib3', 'requests', 'neo4j', 'ica_framework', 'networkx', 'ica_framework.utils.logger']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL + 1)  # Higher than CRITICAL
    logger.disabled = True
    logger.propagate = False

# Also suppress the root logger completely
root_logger = logging.getLogger()
root_logger.setLevel(logging.CRITICAL + 1)
root_logger.disabled = True

# Set logging level to suppress ALL framework messages
logging.getLogger('ica_framework').setLevel(logging.CRITICAL + 1)
logging.getLogger('ica_framework.utils').setLevel(logging.CRITICAL + 1)
logging.getLogger('ica_framework.utils.logger').setLevel(logging.CRITICAL + 1)

# Redirect stderr and stdout temporarily to suppress any remaining output
class SuppressOutput:
    def __enter__(self):
        self._original_stderr = sys.stderr
        self._original_stdout = sys.stdout
        sys.stderr = open(os.devnull, 'w')
        sys.stdout = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stdout.close()
        sys.stderr = self._original_stderr
        sys.stdout = self._original_stdout

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Additional logging suppression right before imports
logging.disable(logging.CRITICAL)
os.environ['ICA_LOG_LEVEL'] = 'CRITICAL'  # Try to set framework log level
os.environ['PYTHONHASHSEED'] = '0'  # Suppress any hash-based randomness

from ica_framework import ICAAgent, Config
from ica_framework.sandbox import SandboxEnvironment, ProceduralDatasetGenerator, MultiDomainScenarioGenerator

# Immediately suppress ICA framework logging after import
logging.getLogger('ica_framework').disabled = True
logging.getLogger('ica_framework.utils.logger').disabled = True

# Also suppress the specific ICA logger instance
try:
    from ica_framework.utils.logger import ica_logger
    # Try to set loguru to CRITICAL+1 level to completely disable
    if hasattr(ica_logger, 'use_loguru') and ica_logger.use_loguru:
        # For loguru, remove all handlers
        try:
            from loguru import logger as loguru_logger
            loguru_logger.remove()  # Remove all handlers
            loguru_logger.disable("")  # Disable all logging
        except:
            pass
    else:
        # For standard logging, disable the logger
        if hasattr(ica_logger, 'logger'):
            ica_logger.logger.disabled = True
            ica_logger.logger.setLevel(logging.CRITICAL + 1)
except:
    pass

# Try to import enhanced knowledge graph - optional for Neo4j support
try:
    with SuppressOutput():  # Suppress any import-time logging
        from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph
    HAS_ENHANCED_KG = True
except ImportError:
    # Don't print anything - keep completely silent
    HAS_ENHANCED_KG = False


# Enhanced ContinuousLearningEnvironment capabilities
class PhysicsSimulation:
    """Enhanced physics simulation for realistic learning scenarios"""
    
    def __init__(self):
        self.entities = self._initialize_physics_entities()
        self.relations = self._initialize_physics_relations()
        self.environment_complexity = 1.0
        self.complexity_growth_rate = 0.001
        
    def _initialize_physics_entities(self):
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
        
    def _initialize_physics_relations(self):
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
    
    def generate_physics_scenario(self, complexity_level=None):
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
        
    def generate_smart_home_scenario(self):
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
    
    def generate_industrial_scenario(self):
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
    
    def generate_scenario(self, scenario_type=None):
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
    
    def _generate_basic_scenario(self):
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


class ContinuousLearning:
    def __init__(self, 
                 checkpoint_dir=None,  # Deprecated parameter
                 database_backend="memory",
                 database_config=None):
        # Database configuration
        self.database_backend = database_backend
        self.database_config = database_config or {}
        
        self.running = True
        self.agent = None
        self.session_start_time = datetime.now()
        self.session_timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        
        self.session_stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'experiments_conducted': 0,
            'confidence_progression': [],
            'learning_events': [],
            'scenarios_completed': 0,
            'session_start_time': time.time(),
            'total_learning_time': 0.0,
            'session_id': self.session_timestamp,
            'database_backend': database_backend
        }
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 Stopping... Neo4j has all data")
        self.running = False
        # Save final session state
        self.save_checkpoint()
        print("�️ All knowledge preserved in Neo4j database")
        sys.exit(0)
    
    def load_checkpoint(self):
        """Load previous learning session - check Neo4j for existing data"""
        if self.database_backend == "neo4j" and HAS_ENHANCED_KG:
            try:
                # We'll check Neo4j after the agent is created
                print("📂 Checking Neo4j for existing session data...")
                return None  # Will be handled after agent creation
            except Exception:
                pass
        
        print("📂 Starting fresh session")
        return None
    
    def save_checkpoint(self):
        """Save current learning progress - Neo4j handles all persistence"""
        # Save session metadata to Neo4j as graph properties if possible
        try:
            if (self.database_backend == "neo4j" and HAS_ENHANCED_KG and 
                hasattr(self.agent.knowledge_graph, 'set_graph_property')):
                
                # Save session metadata as graph properties
                self.agent.knowledge_graph.set_graph_property('session_id', self.session_stats['session_id'])
                self.agent.knowledge_graph.set_graph_property('scenarios_completed', self.session_stats['scenarios_completed'])
                self.agent.knowledge_graph.set_graph_property('total_learning_time', self.session_stats['total_learning_time'])
                self.agent.knowledge_graph.set_graph_property('last_updated', time.time())
                
                print(f"📊 Session metadata saved to Neo4j: {self.session_stats['scenarios_completed']} scenarios")
            else:
                print(f"📊 Session stats: {self.session_stats['scenarios_completed']} scenarios, graph data in {self.database_backend}")
            return True
        except Exception as e:
            print(f"⚠️ Session stats error: {e}")
            return False
    
    def save_incremental_data(self, scenario_count):
        """Save incremental learning data - disabled for Neo4j-only mode"""
        # No longer saving JSON snapshots - data is in Neo4j
        pass
    
    def show_progress_bar(self, scenario_num, round_num, nodes, edges, node_growth, edge_growth, confidence, learning_rate):
        """Show clean progress bar with nodes and edges"""
        # Calculate progress percentages - increased max edges for better scaling
        max_nodes = 100  # Increased to allow more node growth
        max_edges = 2000  # Increased to 2000 to match the milestone updates
        
        node_progress = min(100, (nodes / max_nodes) * 100)
        edge_progress = min(100, (edges / max_edges) * 100)
        
        # Create visual progress bars
        def make_bar(progress, width=20):
            filled = int(progress * width / 100)
            return "█" * filled + "░" * (width - filled)
        
        node_bar = make_bar(node_progress)
        edge_bar = make_bar(edge_progress)
        
        # Format growth indicators
        node_indicator = f"(+{node_growth})" if node_growth > 0 else ""
        edge_indicator = f"(+{edge_growth})" if edge_growth > 0 else ""
        
        # Clear line and show progress
        print(f"\rS{scenario_num:04d} R{round_num:02d} │ Nodes: {node_bar} {nodes:3d}/{max_nodes} {node_indicator:>5} │ Edges: {edge_bar} {edges:4d}/{max_edges} {edge_indicator:>5} │ Conf: {confidence:.3f} │ Rate: {learning_rate:.1f}/s", end="", flush=True)
    
    def debug_node_learning(self, observation, nodes_before, nodes_after, edges_before, edges_after):
        """Debug why nodes might not be updating - DISABLED for clean output"""
        # Debugging disabled to prevent console spam
        pass
    
    def should_show_detailed_update(self, edges_count, last_detailed_update):
        """Determine if we should show a detailed update based on edge milestones"""
        # Show updates every 2000 edges or every 100 scenarios, whichever comes first
        edge_milestone = (edges_count // 2000) > (last_detailed_update // 2000)
        return edge_milestone
    
    def create_learning_scenarios(self):
        """Create comprehensive learning scenarios"""
        # Extended Smart Home IoT Learning Scenarios
        entities = [
            # Sensors
            {"id": "temp_sensor", "label": "sensor"},
            {"id": "humidity_sensor", "label": "sensor"},
            {"id": "light_sensor", "label": "sensor"},
            {"id": "motion_detector", "label": "sensor"},
            {"id": "door_sensor", "label": "sensor"},
            {"id": "window_sensor", "label": "sensor"},
            {"id": "smoke_detector", "label": "sensor"},
            {"id": "air_quality_sensor", "label": "sensor"},
            
            # Controllers
            {"id": "thermostat", "label": "controller"},
            {"id": "light_controller", "label": "controller"},
            {"id": "security_system", "label": "controller"},
            {"id": "hvac_controller", "label": "controller"},
            {"id": "irrigation_controller", "label": "controller"},
            
            # Actuators
            {"id": "hvac_unit", "label": "actuator"},
            {"id": "smart_lights", "label": "actuator"},
            {"id": "door_lock", "label": "actuator"},
            {"id": "window_blinds", "label": "actuator"},
            {"id": "sprinkler_system", "label": "actuator"},
            {"id": "air_purifier", "label": "actuator"},
            
            # Environment
            {"id": "room_temperature", "label": "environment"},
            {"id": "room_humidity", "label": "environment"},
            {"id": "room_brightness", "label": "environment"},
            {"id": "occupancy_state", "label": "environment"},
            {"id": "air_quality", "label": "environment"},
            {"id": "energy_consumption", "label": "environment"}
        ]
        
        return [
            {
                "name": "System Initialization",
                "entities": entities,
                "relationships": [],
                "description": "Agent discovers all IoT system components"
            },
            {
                "name": "Temperature Sensing",
                "entities": [{"id": "temp_sensor", "label": "sensor"}, {"id": "room_temperature", "label": "environment"}],
                "relationships": [
                    {"source": "temp_sensor", "target": "room_temperature", "type": "measures", "confidence": 0.95}
                ],
                "description": "Learning temperature measurement relationships"
            },
            {
                "name": "Humidity Detection", 
                "entities": [{"id": "humidity_sensor", "label": "sensor"}, {"id": "room_humidity", "label": "environment"}],
                "relationships": [
                    {"source": "humidity_sensor", "target": "room_humidity", "type": "detects", "confidence": 0.9}
                ],
                "description": "Learning humidity sensing patterns"
            },
            {
                "name": "Temperature Control Loop",
                "entities": [{"id": "room_temperature", "label": "environment"}, {"id": "thermostat", "label": "controller"}],
                "relationships": [
                    {"source": "room_temperature", "target": "thermostat", "type": "influences", "confidence": 0.85}
                ],
                "description": "Learning feedback control mechanisms"
            },
            {
                "name": "HVAC System Integration",
                "entities": [{"id": "thermostat", "label": "controller"}, {"id": "hvac_unit", "label": "actuator"}],
                "relationships": [
                    {"source": "thermostat", "target": "hvac_unit", "type": "controls", "confidence": 0.9}
                ],
                "description": "Learning HVAC system control"
            },
            {
                "name": "Climate Response",
                "entities": [{"id": "hvac_unit", "label": "actuator"}, {"id": "room_temperature", "label": "environment"}],
                "relationships": [
                    {"source": "hvac_unit", "target": "room_temperature", "type": "modifies", "confidence": 0.8}
                ],
                "description": "Learning environmental response to HVAC"
            },
            {
                "name": "Motion-Based Automation",
                "entities": [{"id": "motion_detector", "label": "sensor"}, {"id": "occupancy_state", "label": "environment"}],
                "relationships": [
                    {"source": "motion_detector", "target": "occupancy_state", "type": "determines", "confidence": 0.9}
                ],
                "description": "Learning occupancy detection patterns"
            },
            {
                "name": "Smart Lighting Control",
                "entities": [{"id": "occupancy_state", "label": "environment"}, {"id": "light_controller", "label": "controller"}],
                "relationships": [
                    {"source": "occupancy_state", "target": "light_controller", "type": "triggers", "confidence": 0.8}
                ],
                "description": "Learning adaptive lighting responses"
            },
            {
                "name": "Light System Response",
                "entities": [{"id": "light_controller", "label": "controller"}, {"id": "smart_lights", "label": "actuator"}],
                "relationships": [
                    {"source": "light_controller", "target": "smart_lights", "type": "activates", "confidence": 0.95}
                ],
                "description": "Learning lighting actuator control"
            },
            {
                "name": "Ambient Light Integration",
                "entities": [{"id": "light_sensor", "label": "sensor"}, {"id": "room_brightness", "label": "environment"}],
                "relationships": [
                    {"source": "light_sensor", "target": "room_brightness", "type": "measures", "confidence": 0.9}
                ],
                "description": "Learning ambient light measurement"
            },
            {
                "name": "Security System Monitoring",
                "entities": [{"id": "door_sensor", "label": "sensor"}, {"id": "security_system", "label": "controller"}],
                "relationships": [
                    {"source": "door_sensor", "target": "security_system", "type": "alerts", "confidence": 0.95}
                ],
                "description": "Learning security event detection"
            },
            {
                "name": "Energy Optimization",
                "entities": [{"id": "hvac_unit", "label": "actuator"}, {"id": "energy_consumption", "label": "environment"}],
                "relationships": [
                    {"source": "hvac_unit", "target": "energy_consumption", "type": "affects", "confidence": 0.8}
                ],
                "description": "Learning energy consumption patterns"
            },
            {
                "name": "Air Quality Management",
                "entities": [{"id": "air_quality_sensor", "label": "sensor"}, {"id": "air_purifier", "label": "actuator"}],
                "relationships": [
                    {"source": "air_quality_sensor", "target": "air_purifier", "type": "triggers", "confidence": 0.85}
                ],
                "description": "Learning air quality control loops"
            },
            {
                "name": "Multi-Sensor Correlation",
                "entities": [{"id": "temp_sensor", "label": "sensor"}, {"id": "humidity_sensor", "label": "sensor"}],
                "relationships": [
                    {"source": "temp_sensor", "target": "humidity_sensor", "type": "correlates", "confidence": 0.7}
                ],
                "description": "Learning sensor correlation patterns"
            },
            {
                "name": "Complex System Dynamics",
                "entities": [{"id": "occupancy_state", "label": "environment"}, {"id": "energy_consumption", "label": "environment"}],
                "relationships": [
                    {"source": "occupancy_state", "target": "energy_consumption", "type": "influences", "confidence": 0.75}
                ],
                "description": "Learning complex system interactions"
            }
        ]
    
    def run_continuous_learning(self):
        """Main continuous learning loop with resume capability"""
        print("🧠 ICA Framework Continuous Learning")
        print("=" * 80)
        print("Press Ctrl+C to save progress and exit")
        print()
        print("📝 Enhanced Learning Behavior:")
        print("   • Enhanced Procedural Dataset: Complex motifs (control_loop, sensor_network, etc.)")
        print("   • Multi-Domain Scenarios: Smart city, healthcare, manufacturing, energy grid")
        print("   • Sandbox Experiments: Real-time ablation studies with enhanced agents")
        print("   • Physics Simulation: 40+ entities with realistic interactions") 
        print("   • Original Scenarios: IoT and procedural generation for continuity")
        print("   • Nodes represent unique entities (sensors, controllers, etc.)")
        print("   • Edges represent relationships between entities")
        print("   • Node count stabilizes once all entities are discovered")
        print("   • Edge count grows as relationships are learned and reinforced")
        print("   • Updates shown every 2000 edge milestone")
        print(f"   • Database backend: {self.database_backend}")
        print("=" * 80)
        
        # Try to load previous session
        checkpoint = self.load_checkpoint()
        start_scenario = 0
        scenario_round = 1
        last_detailed_edges = 0  # Initialize here, will be updated after Neo4j check
        
        # No checkpoint loading needed - Neo4j handles persistence
        
        # Create optimized configuration
        config = Config()
        config.abstraction.motif_min_size = 10  # Fast learning mode
        config.abstraction.motif_max_size = 20
        config.abstraction.num_clusters = 3
        
        # Create or restore agent with suppressed output
        with SuppressOutput():
            self.agent = ICAAgent(config)
        
        # Replace the standard knowledge graph with enhanced version supporting databases
        print(f"🔧 Database Backend: {self.database_backend}")
        print(f"🧩 Enhanced KG Available: {HAS_ENHANCED_KG}")
        
        if self.database_backend != "memory" and HAS_ENHANCED_KG:
            print(f"🚀 Initializing {self.database_backend} backend...")
            # Silent database initialization
            try:
                enhanced_kg = EnhancedKnowledgeGraph(
                    backend=self.database_backend,
                    config=self.database_config
                )
                
                # Test connection
                if enhanced_kg.connect():
                    print(f"✅ {self.database_backend} connection successful")
                    
                    # Migrate existing data if any
                    if hasattr(self.agent, 'knowledge_graph') and self.agent.knowledge_graph.graph.number_of_nodes() > 0:
                        print(f"📊 Migrating {self.agent.knowledge_graph.graph.number_of_nodes()} nodes...")
                        enhanced_kg.import_from_networkx(self.agent.knowledge_graph.graph)
                    
                    # Replace the knowledge graph
                    self.agent.knowledge_graph = enhanced_kg
                    print(f"🔄 Knowledge graph replaced with {self.database_backend} backend")
                else:
                    print(f"❌ {self.database_backend} connection failed, falling back to memory")
                    self.database_backend = "memory"
                
            except Exception as e:
                print(f"❌ {self.database_backend} initialization failed: {e}")
                print(f"🔄 Falling back to memory backend")
                self.database_backend = "memory"
        elif self.database_backend != "memory":
            print(f"❌ Enhanced KG not available, falling back to memory")
            self.database_backend = "memory"
        
        with SuppressOutput():
            self.agent.initialize_world_model(
                state_dim=32,
                action_dim=8,
                num_relations=10
            )
        
        # Get base learning scenarios EARLY (before any resumption logic)
        base_scenarios = self.create_learning_scenarios()
        
        # Try to load agent from Neo4j on startup and resume session
        if self.database_backend == "neo4j" and HAS_ENHANCED_KG:
            try:
                # Check if we have existing data in Neo4j
                if hasattr(self.agent.knowledge_graph, 'get_stats'):
                    db_stats = self.agent.knowledge_graph.get_stats()
                    existing_nodes = db_stats.get('nodes', 0)
                    existing_edges = db_stats.get('edges', 0)
                    
                    if existing_nodes > 0 or existing_edges > 0:
                        print(f"📊 Resuming from Neo4j: {existing_nodes} nodes, {existing_edges} edges")
                        
                        # Update session stats to reflect existing data
                        self.session_stats['total_nodes'] = existing_nodes
                        self.session_stats['total_edges'] = existing_edges
                        
                        # Initialize with estimated scenario count as fallback
                        estimated_scenarios = max(0, existing_edges // 2)
                        start_scenario = estimated_scenarios
                        self.session_stats['scenarios_completed'] = estimated_scenarios
                        print(f"🔄 Initial estimate: ~{estimated_scenarios} scenarios from {existing_edges} edges")
                        
                        # Try to load saved session metadata
                        try:
                            if hasattr(self.agent.knowledge_graph, 'get_graph_property'):
                                saved_scenarios = self.agent.knowledge_graph.get_graph_property('scenarios_completed')
                                saved_time = self.agent.knowledge_graph.get_graph_property('total_learning_time')
                                saved_session_id = self.agent.knowledge_graph.get_graph_property('session_id')
                                
                                if saved_scenarios is not None:
                                    self.session_stats['scenarios_completed'] = int(saved_scenarios)
                                    start_scenario = int(saved_scenarios)
                                    print(f"📈 Found saved metadata: {start_scenario} scenarios completed")
                                    
                                if saved_time is not None:
                                    self.session_stats['total_learning_time'] = float(saved_time)
                                    print(f"⏱️ Restored learning time: {saved_time:.1f}s")
                                    
                                if saved_session_id:
                                    print(f"🆔 Continuing session: {saved_session_id}")
                        except Exception as e:
                            print(f"⚠️ Metadata loading error: {e}")
                            print(f"📊 Using estimated scenario count: {start_scenario}")
                        
                        scenario_round = (start_scenario // len(base_scenarios)) + 1
                        print(f"🎯 Continuing from round {scenario_round}")
                        print(f"📈 Resuming scenario count from: {start_scenario}")
                        
                        # Set the last_detailed_edges to avoid immediate milestone display
                        last_detailed_edges = (existing_edges // 2000) * 2000
                    else:
                        print("🆕 No existing data found in Neo4j - starting fresh")
                else:
                    print("🆕 Starting fresh - Neo4j connection ready")
            except Exception as e:
                print(f"⚠️ Error checking Neo4j data: {e}")
                print("🆕 Continuing with fresh start")
        
        # Initialize enhanced scenario generators with proper configuration
        try:
            from ica_framework.utils.config import SandboxConfig
            sandbox_config = SandboxConfig()
            
            # Enhanced sandbox environment with complex motifs
            self.sandbox_env = SandboxEnvironment(sandbox_config)
            self.physics_sim = PhysicsSimulation()
            self.procedural_gen = ProceduralDatasetGenerator(sandbox_config)
            
            # NEW: Multi-domain scenario generator for comprehensive AGI training
            self.multi_domain_gen = MultiDomainScenarioGenerator(sandbox_config)
            
            print("🌟 Enhanced scenario generators initialized:")
            print("   • Physics Simulation (40+ entities)")
            print("   • Procedural Dataset Generator (18+ motif types)")
            print("   • Multi-Domain Scenarios (4 domains: smart_city, healthcare, manufacturing, energy)")
            print("   • Complex Motifs: control_loop, sensor_network, hierarchical_system, etc.")
            
        except Exception as e:
            print(f"⚠️ Enhanced generators initialization error: {e}")
            print("🔄 Falling back to basic scenario generation")
            # Initialize basic fallbacks
            self.sandbox_env = None
            self.physics_sim = PhysicsSimulation()
            self.procedural_gen = None
            self.multi_domain_gen = None
        
        print("=" * 80)
        print()
        
        # Show resumption status clearly
        if start_scenario > 0:
            print(f"🔄 RESUMING SESSION: Starting from scenario {start_scenario + 1}")
            print(f"   Previous scenarios completed: {start_scenario}")
            print(f"   Current round: {scenario_round}")
            print(f"   Existing knowledge: {self.session_stats['total_nodes']} nodes, {self.session_stats['total_edges']} edges")
            print()
        else:
            print("🆕 STARTING NEW SESSION")
            print()
        
        # Infinite continuous learning loop
        scenario_count = start_scenario
        last_save_time = time.time()
        # last_detailed_edges is now set above based on Neo4j resume data
        
        try:
            while self.running:
                # Calculate current round and scenario within round
                current_round = (scenario_count // len(base_scenarios)) + 1
                scenario_in_round = scenario_count % len(base_scenarios)
                
                # Add variation to scenarios each round for continuous learning
                scenario = base_scenarios[scenario_in_round].copy()
                
                # Enhanced scenario selection with new sandbox capabilities
                scenario_type = scenario_count % 10
                
                if scenario_type == 0 and self.procedural_gen:
                    # Use enhanced procedural dataset with complex motifs
                    try:
                        dataset = self.procedural_gen.generate_dataset(num_nodes=50, num_edges=100)
                        # Convert dataset to scenario format
                        entities = []
                        relationships = []
                        
                        for node in dataset['graph'].nodes():
                            node_data = dataset['graph'].nodes[node]
                            entities.append({
                                "id": node,
                                "label": node_data.get('label', 'entity'),
                                "properties": node_data.get('properties', {})
                            })
                        
                        for edge in dataset['graph'].edges(data=True):
                            relationships.append({
                                "source": edge[0],
                                "target": edge[1],
                                "type": edge[2].get('relationship', 'related'),
                                "confidence": edge[2].get('confidence', 0.8),
                                "motif_type": edge[2].get('motif_type', 'unknown')
                            })
                        
                        scenario = {
                            'name': f"Enhanced Procedural Dataset {scenario_count + 1}",
                            'entities': entities,
                            'relationships': relationships,
                            'description': f"Complex motifs: {', '.join(dataset.get('motif_types', []))}"
                        }
                    except Exception as e:
                        print(f"⚠️ Procedural generation error: {e}")
                        # Fallback to base scenario
                        pass
                
                elif scenario_type == 1 and self.multi_domain_gen:
                    # Use multi-domain scenario generation
                    try:
                        # Generate scenarios from different domains
                        domain_scenarios = self.multi_domain_gen.generate_domain_scenarios(
                            domain=np.random.choice(['smart_city', 'healthcare', 'manufacturing', 'energy_grid']),
                            count=1
                        )
                        
                        if domain_scenarios:
                            domain_scenario = domain_scenarios[0]
                            scenario = {
                                'name': f"Multi-Domain: {domain_scenario['domain']} - {domain_scenario['pattern']}",
                                'entities': domain_scenario['entities'],
                                'relationships': domain_scenario['relationships'],
                                'description': f"Domain: {domain_scenario['domain']}, Pattern: {domain_scenario['pattern']}"
                            }
                    except Exception as e:
                        print(f"⚠️ Multi-domain generation error: {e}")
                        # Fallback to base scenario
                        pass
                
                elif scenario_type == 2 and self.sandbox_env:
                    # Use sandbox environment for complex scenarios
                    try:
                        # Run enhanced experiment with complex motifs
                        splits = self.sandbox_env.setup_test_environment()
                        enhanced_results = self.sandbox_env.run_enhanced_experiment(splits)
                        
                        # Convert sandbox results to scenario format
                        entities = [{"id": f"sandbox_entity_{i}", "label": "sandbox_component"} for i in range(10)]
                        relationships = [
                            {
                                "source": f"sandbox_entity_{i}",
                                "target": f"sandbox_entity_{(i+1)%10}",
                                "type": "sandbox_relation",
                                "confidence": enhanced_results.get('global_confidence', 0.7)
                            }
                            for i in range(5)
                        ]
                        
                        scenario = {
                            'name': f"Sandbox Enhanced Experiment {scenario_count + 1}",
                            'entities': entities,
                            'relationships': relationships,
                            'description': f"Sandbox confidence: {enhanced_results.get('global_confidence', 0.0):.3f}"
                        }
                    except Exception as e:
                        print(f"⚠️ Sandbox generation error: {e}")
                        # Fallback to base scenario
                        pass
                
                # Keep original physics simulation and procedural generation for other scenarios
                elif scenario_count % 5 == 0:
                    # Physics simulation (every 5th scenario)
                    try:
                        physics_scenario = self.physics_sim.generate_physics_scenario()
                        scenario = {
                            'name': f"Physics Simulation {scenario_count + 1}",
                            'entities': physics_scenario.get('entities', []),
                            'relationships': physics_scenario.get('relationships', []),
                            'description': f"Advanced physics simulation - {physics_scenario.get('scenario_type', 'dynamic')}"
                        }
                    except Exception:
                        # Fallback to base scenario if enhanced fails
                        pass
                
                elif scenario_count % 7 == 0 and self.procedural_gen:
                    # Original procedural generation (every 7th scenario)
                    try:
                        procedural_scenario = self.procedural_gen.generate_scenario()
                        scenario = {
                            'name': f"Original Procedural Scenario {scenario_count + 1}",
                            'entities': procedural_scenario.get('entities', []),
                            'relationships': procedural_scenario.get('relationships', []),
                            'description': f"Procedural generation - {procedural_scenario.get('scenario_type', 'dynamic')}"
                        }
                    except Exception:
                        # Fallback to base scenario if enhanced fails
                        pass
                
                # Enhance scenario with round-specific variations
                if current_round > 1:
                    scenario['name'] = f"{scenario['name']} (Round {current_round})"
                    scenario['description'] = f"{scenario['description']} - Variation {current_round}"
                    
                    # Add noise and complexity as rounds progress
                    for rel in scenario.get('relationships', []):
                        # Slightly modify confidence values for learning
                        noise = np.random.normal(0, 0.05 * current_round)
                        # Ensure confidence key exists before modifying
                        if 'confidence' in rel:
                            rel['confidence'] = max(0.1, min(0.95, rel['confidence'] + noise))
                        else:
                            # Set default confidence if missing
                            rel['confidence'] = max(0.1, min(0.95, 0.8 + noise))
                
                # Create observation with enhanced patterns
                state = np.random.normal(0, 0.2, 32)
                
                # Add realistic patterns with round-based complexity
                complexity_factor = 1.0 + (current_round - 1) * 0.1
                noise_factor = 0.1 * current_round
                
                if any(word in scenario['name'].lower() for word in ['temperature', 'temp', 'climate']):
                    base_temp = 22.5 + np.random.normal(0, 2 * complexity_factor)
                    state[0:4] = [base_temp, 1.2 * complexity_factor, -0.3, 0.8]
                elif any(word in scenario['name'].lower() for word in ['motion', 'occupancy']):
                    state[4:8] = [1.0, 0.0, 0.9 + np.random.normal(0, noise_factor), 0.1]
                elif any(word in scenario['name'].lower() for word in ['light', 'brightness']):
                    state[8:12] = [0.7 + np.random.normal(0, 0.2 * complexity_factor), 0.3, 1.0, 0.2]
                elif any(word in scenario['name'].lower() for word in ['security', 'door', 'window']):
                    state[12:16] = [0.1, 0.9 + np.random.normal(0, noise_factor), 0.0, 1.0]
                elif any(word in scenario['name'].lower() for word in ['energy', 'consumption']):
                    state[16:20] = [0.6 + np.random.normal(0, 0.15 * complexity_factor), 0.4, 0.8, 0.3]
                elif any(word in scenario['name'].lower() for word in ['air', 'quality', 'humidity']):
                    state[20:24] = [0.5 + np.random.normal(0, noise_factor), 0.7, 0.3, 0.9]
                
                observation = {
                    "entities": scenario["entities"],
                    "relationships": scenario["relationships"],
                    "state": state
                }
                
                # Get state before learning
                nodes_before = self.agent.knowledge_graph.graph.number_of_nodes()
                edges_before = self.agent.knowledge_graph.graph.number_of_edges()
                
                # Record timing and run learning step with complete silence
                step_start = time.time()
                with SuppressOutput():
                    step_results = self.agent.active_learning_step(observation)
                step_time = time.time() - step_start
                self.session_stats['total_learning_time'] += step_time
                
                # Get state after learning
                nodes_after = self.agent.knowledge_graph.graph.number_of_nodes()
                edges_after = self.agent.knowledge_graph.graph.number_of_edges()
                
                # Calculate growth and learning rate
                node_growth = nodes_after - nodes_before
                edge_growth = edges_after - edges_before
                total_scenarios_completed = scenario_count + 1
                learning_rate = total_scenarios_completed / max(self.session_stats['total_learning_time'], 0.001)
                
                # Only show progress at significant milestones - no continuous output
                # Progress bar disabled for completely clean output
                
                # Debug disabled for clean output
                # if scenario_count % 50 == 0:  # Debug every 50 scenarios
                #     self.debug_node_learning(observation, nodes_before, nodes_after, edges_before, edges_after)
                
                # Show detailed update every 2000 edges
                if self.should_show_detailed_update(edges_after, last_detailed_edges):
                    total_scenarios_completed = scenario_count + 1
                    print(f"\n📊 Milestone: {edges_after} edges reached!")
                    print(f"    Nodes: {nodes_after}, Confidence: {step_results.get('global_confidence', 0):.3f}")
                    print(f"    Scenarios completed: {total_scenarios_completed}, Rate: {learning_rate:.1f}/s")
                    
                    # Show database backend status
                    if self.database_backend == "neo4j":
                        try:
                            # Test if data is actually in Neo4j
                            if hasattr(self.agent.knowledge_graph, 'get_stats'):
                                db_stats = self.agent.knowledge_graph.get_stats()
                                print(f"    🗄️ Neo4j: {db_stats.get('nodes', 'N/A')} nodes, {db_stats.get('edges', 'N/A')} edges in database")
                            else:
                                print(f"    🗄️ Neo4j: Backend active")
                        except Exception as e:
                            print(f"    ⚠️ Neo4j: Connection issue - {str(e)[:50]}...")
                    else:
                        print(f"    💾 Backend: {self.database_backend}")
                    
                    last_detailed_edges = edges_after
                
                # Update session stats
                total_scenarios_completed = scenario_count + 1
                self.session_stats['total_nodes'] = nodes_after
                self.session_stats['total_edges'] = edges_after
                self.session_stats['experiments_conducted'] += step_results['experiments_conducted']
                self.session_stats['confidence_progression'].append(step_results['global_confidence'])
                self.session_stats['scenarios_completed'] = total_scenarios_completed
                self.session_stats['learning_events'].append({
                    'scenario': scenario['name'],
                    'round': current_round,
                    'nodes_added': node_growth,
                    'edges_added': edge_growth,
                    'time': step_time,
                    'confidence': step_results['global_confidence']
                })
                
                # Auto-save session metadata every 30 seconds
                current_time = time.time()
                if current_time - last_save_time > 30:
                    # Save session metadata to Neo4j
                    self.save_checkpoint()
                    last_save_time = current_time
                
                # Save incremental data every 2000 edges or every 100 scenarios
                if edges_after % 2000 == 0 or total_scenarios_completed % 100 == 0:
                    self.save_incremental_data(total_scenarios_completed)
                
                scenario_count += 1
                
                # Brief delay between scenarios (adjust for speed)
                time.sleep(0.05)  # Reduced delay for faster processing
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        finally:
            print(f"\n💾 Session complete - saving final state to {self.database_backend}")
            # Save final session metadata
            self.save_checkpoint()
            
            # Calculate learning rate safely
            total_time = self.session_stats.get('total_learning_time', 0.001)
            total_scenarios = self.session_stats.get('scenarios_completed', 0)
            learning_rate = total_scenarios / max(total_time, 0.001)
            
            # Show final edge milestone reached
            edge_milestones = self.session_stats['total_edges'] // 2000
            print(f"✅ Session complete! {self.session_stats['scenarios_completed']} scenarios processed")
            print(f"📊 Final: {self.session_stats['total_nodes']} nodes, {self.session_stats['total_edges']} edges")
            print(f"🎯 Edge milestones reached: {edge_milestones} × 2000")
            print(f"⚡ Rate: {learning_rate:.1f} scenarios/sec")
            print(f"🗄️ All knowledge and session data persisted in {self.database_backend}")
            print(f"▶️ Resume with: python examples/learning.py")

    

def main():
    """Main function to run continuous learning with database options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ICA Framework Continuous Learning')
    parser.add_argument('--backend', choices=['memory', 'neo4j'], default='neo4j',
                       help='Database backend to use (default: neo4j)')
    parser.add_argument('--neo4j-uri', default=None,
                       help='Neo4j URI (overrides config file)')
    parser.add_argument('--neo4j-user', default=None,
                       help='Neo4j username (overrides config file)')
    parser.add_argument('--neo4j-password', default=None,
                       help='Neo4j password (overrides config file)')
    parser.add_argument('--neo4j-database', default=None,
                       help='Neo4j database name (overrides config file)')
    
    args = parser.parse_args()
    
    # Configure database
    database_config = {}
    
    if args.backend == 'neo4j':
        # Try to load from config file first
        config_file = Path("config/database/neo4j.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)['config']
                
                database_config = {
                    'uri': args.neo4j_uri or file_config.get('uri', 'neo4j://127.0.0.1:7687'),
                    'username': args.neo4j_user or file_config.get('username', 'neo4j'),
                    'password': args.neo4j_password or file_config.get('password', 'password'),
                    'database': args.neo4j_database or file_config.get('database', 'neo4j')
                }
                
                print(f"🗄️ Neo4j Configuration (from {config_file}):")
            except Exception as e:
                print(f"⚠️ Failed to load config file: {e}")
                print("Using command line arguments...")
                database_config = {
                    'uri': args.neo4j_uri or 'neo4j://127.0.0.1:7687',
                    'username': args.neo4j_user or 'neo4j',
                    'password': args.neo4j_password or 'password',
                    'database': args.neo4j_database or 'neo4j'
                }
        else:
            print(f"⚠️ Config file not found: {config_file}")
            print("Using command line arguments...")
            database_config = {
                'uri': args.neo4j_uri or 'neo4j://127.0.0.1:7687',
                'username': args.neo4j_user or 'neo4j',
                'password': args.neo4j_password or 'password',
                'database': args.neo4j_database or 'neo4j'
            }
        
        print(f"   URI: {database_config['uri']}")
        print(f"   Database: {database_config['database']}")
        print(f"   User: {database_config['username']}")
        print()
    
    # Create learning system with specified backend
    learner = ContinuousLearning(
        database_backend=args.backend,
        database_config=database_config
    )
    
    learner.run_continuous_learning()


if __name__ == "__main__":
    main()
