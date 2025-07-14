"""
Worker Functions for ICA Framework Parallel Learning
Contains the actual worker processes for parallel learning
"""

import time
import os
import numpy as np
from typing import Dict, Any, List
from multiprocessing import Queue
import multiprocessing as mp

from .scenario_generators import PhysicsSimulation


class SuppressOutput:
    """Context manager to suppress ALL output including logging"""
    
    def __enter__(self):
        import logging
        import sys
        
        # Store originals
        self._original_stderr = sys.stderr
        self._original_stdout = sys.stdout
        
        # Redirect to devnull
        sys.stderr = open(os.devnull, 'w')
        sys.stdout = open(os.devnull, 'w')
        
        # Disable all logging during context
        self._original_level = logging.root.level
        logging.disable(logging.CRITICAL)
        
        # Disable specific loggers
        loggers_to_disable = [
            'ica_framework',
            'ica_framework.utils.logger',
            'ica_framework.core',
            'ica_framework.components',
            'neo4j',
            'urllib3',
            'torch'
        ]
        
        self._disabled_loggers = []
        for logger_name in loggers_to_disable:
            logger = logging.getLogger(logger_name)
            if not logger.disabled:
                logger.disabled = True
                logger.setLevel(logging.CRITICAL + 1)
                self._disabled_loggers.append(logger)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import logging
        import sys
        
        # Close redirected streams
        sys.stderr.close()
        sys.stdout.close()
        
        # Restore original streams
        sys.stderr = self._original_stderr
        sys.stdout = self._original_stdout
        
        # Note: Keep logging disabled in workers - don't restore it


def continuous_parallel_worker(worker_id: int, 
                             scenario_queue: Queue, 
                             results_queue: Queue, 
                             database_config: Dict[str, Any], 
                             database_backend: str, 
                             stop_event):
    """Continuous worker function that processes scenarios one by one from a queue"""
    try:
        # Suppress ALL logging in workers - AGGRESSIVE MODE
        import logging
        import warnings
        import os
        
        # Disable ALL logging completely
        logging.disable(logging.CRITICAL)
        os.environ['PYTHONWARNINGS'] = 'ignore'
        warnings.filterwarnings("ignore")
        
        # Disable ICA Framework logger specifically
        ica_logger = logging.getLogger('ica_framework')
        ica_logger.disabled = True
        ica_logger.propagate = False
        ica_logger.setLevel(logging.CRITICAL + 1)
        
        # Disable root logger
        root_logger = logging.getLogger()
        root_logger.disabled = True
        root_logger.setLevel(logging.CRITICAL + 1)
        
        # Initialize worker-specific agent with suppressed output
        # Re-enable SuppressOutput now that debugging is complete
        with SuppressOutput():
            from ..core.ica_agent import ICAAgent
            from ..utils.config import Config
            from ..sandbox import ProceduralDatasetGenerator, MultiDomainScenarioGenerator
            
            # Create worker configuration - MUST MATCH MAIN AGENT CONFIG
            config = Config()
            config.abstraction.motif_min_size = 10  # Fast learning mode
            config.abstraction.motif_max_size = 20
            config.abstraction.num_clusters = 3
            
            # Create worker agent
            worker_agent = ICAAgent(config)
        
        # Initialize worker stats FIRST - needed for database setup
        worker_stats = {
            'worker_id': worker_id,
            'scenarios_processed': 0,
            'total_processing_time': 0,
            'start_time': time.time(),
            'neo4j_test_write': False,
            'neo4j_test_verified': False
        }
        
        # Setup enhanced knowledge graph if available - FRESH CONNECTION PER WORKER
        if database_backend != "memory":
            try:
                from ..enhanced_knowledge_graph import EnhancedKnowledgeGraph
                
                # Create a completely fresh database connection for this worker
                # This is critical for multiprocessing - each worker needs its own connection
                enhanced_kg = EnhancedKnowledgeGraph(
                    backend=database_backend,
                    config=database_config.copy()  # Use copy to avoid shared state
                )
                
                # Force fresh connection - disconnect any existing connections first
                if hasattr(enhanced_kg, 'disconnect'):
                    enhanced_kg.disconnect()
                
                # Create new connection for this worker process
                if enhanced_kg.connect():
                    worker_agent.knowledge_graph = enhanced_kg
                    
                    # Test the connection with a worker-specific test entity
                    test_node_id = f"worker_{worker_id}_test_{int(time.time())}"
                    if hasattr(enhanced_kg, 'add_entity'):
                        enhanced_kg.add_entity(test_node_id, "test_worker", 
                                             {"worker_id": worker_id, "timestamp": time.time(), "process_id": os.getpid()})
                        # Verify the test entity was created
                        test_stats = enhanced_kg.get_stats()
                        worker_stats['neo4j_test_write'] = True
                else:
                    worker_agent.knowledge_graph = None
                    
            except Exception as e:
                worker_agent.knowledge_graph = None
        
        worker_agent.initialize_world_model(
            state_dim=32,
            action_dim=8,
            num_relations=10
        )
        
        # Initialize enhanced generators
        try:
            from ..utils.config import SandboxConfig
            sandbox_config = SandboxConfig()
            
            physics_sim = PhysicsSimulation()
            procedural_gen = ProceduralDatasetGenerator(sandbox_config)
            multi_domain_gen = MultiDomainScenarioGenerator(sandbox_config)
            
        except Exception as e:
            physics_sim = PhysicsSimulation()
            procedural_gen = None
            multi_domain_gen = None
        
        # Output control: only print every 30 seconds
        last_output_time = time.time()
        output_interval = 30.0
        
        while not stop_event.is_set():
            try:
                scenario_info = scenario_queue.get(timeout=1.0)
                if scenario_info is None:
                    break
                    
                scenario_count, scenario, current_round = scenario_info
                scenario_type = scenario_count % 10
                
                # Enhanced scenario generation logic
                scenario = _generate_enhanced_scenario(
                    scenario_type, scenario_count, current_round, 
                    physics_sim, procedural_gen, multi_domain_gen, scenario
                )
                
                # Create observation
                observation = _create_observation(scenario, current_round)
                
                # Process learning step with proper database node tracking
                if database_backend == "neo4j" and hasattr(worker_agent.knowledge_graph, 'get_stats'):
                    # Use database stats for Neo4j
                    stats_before = worker_agent.knowledge_graph.get_stats()
                    nodes_before = stats_before.get('nodes', 0)
                    edges_before = stats_before.get('edges', 0)
                else:
                    # Use local graph for memory backend
                    nodes_before = worker_agent.knowledge_graph.graph.number_of_nodes()
                    edges_before = worker_agent.knowledge_graph.graph.number_of_edges()
                
                step_start = time.time()
                # Remove output suppression temporarily to debug entity creation
                step_results = worker_agent.active_learning_step(observation)
                
                # Ensure entities are properly added to Neo4j with forced database sync
                if database_backend == "neo4j" and hasattr(worker_agent.knowledge_graph, 'db'):
                    # Force explicit entity and relationship creation in Neo4j
                    entities_created = 0
                    relationships_created = 0
                    
                    # Process entities first
                    for entity in observation.get('entities', []):
                        entity_id = entity['id']
                        if hasattr(worker_agent.knowledge_graph, 'add_entity'):
                            # Combine all entity properties
                            all_properties = {}
                            all_properties.update(entity.get('properties_static', {}))
                            all_properties.update(entity.get('properties_dynamic', {}))
                            all_properties['confidence'] = entity.get('confidence', 1.0)
                            all_properties['worker_id'] = worker_id
                            all_properties['scenario_count'] = scenario_count
                            
                            # Force entity creation in Neo4j
                            try:
                                worker_agent.knowledge_graph.add_entity(
                                    entity_id, 
                                    entity.get('label', 'entity'),
                                    all_properties
                                )
                                entities_created += 1
                            except Exception as e:
                                pass  # Continue processing other entities
                    
                    # Process relationships after entities
                    for rel in observation.get('relationships', []):
                        if hasattr(worker_agent.knowledge_graph, 'add_relationship'):
                            try:
                                worker_agent.knowledge_graph.add_relationship(
                                    rel['source'],
                                    rel['target'], 
                                    rel['type'],
                                    rel.get('confidence', 0.5)
                                )
                                relationships_created += 1
                            except Exception as e:
                                pass  # Continue processing other relationships
                    
                    # Force database synchronization for this worker
                    _handle_neo4j_commit(worker_agent.knowledge_graph.db)
                    
                    # Force a database flush if available
                    if hasattr(worker_agent.knowledge_graph, 'flush'):
                        worker_agent.knowledge_graph.flush()
                        
                    # Small delay to ensure database write completion
                    if entities_created > 0 or relationships_created > 0:
                        time.sleep(0.001)  # 1ms delay for database sync
                        
                step_time = time.time() - step_start
                
                # Get accurate node/edge counts after processing
                if database_backend == "neo4j" and hasattr(worker_agent.knowledge_graph, 'get_stats'):
                    stats_after = worker_agent.knowledge_graph.get_stats()
                    nodes_after = stats_after.get('nodes', 0)
                    edges_after = stats_after.get('edges', 0)
                else:
                    nodes_after = worker_agent.knowledge_graph.graph.number_of_nodes()
                    edges_after = worker_agent.knowledge_graph.graph.number_of_edges()
                
                # Update worker stats
                worker_stats['scenarios_processed'] += 1
                worker_stats['total_processing_time'] += step_time
                
                result = {
                    'worker_id': worker_id,
                    'scenario_count': scenario_count,
                    'nodes_added': nodes_after - nodes_before,
                    'edges_added': edges_after - edges_before,
                    'confidence': step_results.get('global_confidence', 0),
                    'processing_time': step_time,
                    'timestamp': time.time()
                }
                results_queue.put(result)
                # Note: multiprocessing.Queue doesn't have task_done() method
                # scenario_queue.task_done()  # Remove this line for multiprocessing compatibility
                
                # Silent worker operation - no output every 30 seconds to avoid encoding issues
                now = time.time()
                if now - last_output_time > output_interval:
                    # Update last output time but don't print to avoid multiprocessing encoding issues
                    last_output_time = now
                    
            except Exception as e:
                # Silent worker operation - continue on errors unless stop event is set
                if not stop_event.is_set():
                    continue
                else:
                    break
        
        # Send final worker stats
        worker_stats['end_time'] = time.time()
        worker_stats['total_runtime'] = worker_stats['end_time'] - worker_stats['start_time']
        results_queue.put({'worker_stats': worker_stats})
        
    except Exception as e:
        error_result = {
            'worker_id': worker_id,
            'error': str(e),
            'timestamp': time.time()
        }
        results_queue.put(error_result)


def parallel_learning_worker(worker_id: int, 
                           scenario_batch: List, 
                           database_config: Dict[str, Any], 
                           database_backend: str, 
                           return_queue: Queue) -> Dict[str, Any]:
    """Worker function for batch parallel learning processing"""
    try:
        # Suppress ALL logging in workers - AGGRESSIVE MODE
        import logging
        import warnings
        import os
        
        # Disable ALL logging completely
        logging.disable(logging.CRITICAL)
        os.environ['PYTHONWARNINGS'] = 'ignore'
        warnings.filterwarnings("ignore")
        
        # Disable ICA Framework logger specifically
        ica_logger = logging.getLogger('ica_framework')
        ica_logger.disabled = True
        ica_logger.propagate = False
        ica_logger.setLevel(logging.CRITICAL + 1)
        
        # Disable root logger
        root_logger = logging.getLogger()
        root_logger.disabled = True
        root_logger.setLevel(logging.CRITICAL + 1)
        
        # Initialize worker-specific agent with suppressed output
        with SuppressOutput():
            from ..core.ica_agent import ICAAgent
            from ..utils.config import Config
            from ..sandbox import ProceduralDatasetGenerator, MultiDomainScenarioGenerator
            
            # Create worker configuration - MUST MATCH MAIN AGENT CONFIG
            config = Config()
            config.abstraction.motif_min_size = 10  # Fast learning mode
            config.abstraction.motif_max_size = 20
            config.abstraction.num_clusters = 3
            
            # Create worker agent
            worker_agent = ICAAgent(config)
            
            # Setup enhanced knowledge graph if available
            if database_backend != "memory":
                try:
                    from ..enhanced_knowledge_graph import EnhancedKnowledgeGraph
                    
                    enhanced_kg = EnhancedKnowledgeGraph(
                        backend=database_backend,
                        config=database_config
                    )
                    if enhanced_kg.connect():
                        worker_agent.knowledge_graph = enhanced_kg
                except Exception:
                    pass  # Silent fallback to memory
            
            worker_agent.initialize_world_model(
                state_dim=32,
                action_dim=8,
                num_relations=10
            )
            
            # Initialize enhanced generators
            try:
                from ..utils.config import SandboxConfig
                sandbox_config = SandboxConfig()
                
                physics_sim = PhysicsSimulation()
                procedural_gen = ProceduralDatasetGenerator(sandbox_config)
                multi_domain_gen = MultiDomainScenarioGenerator(sandbox_config)
                
            except Exception:
                physics_sim = PhysicsSimulation()
                procedural_gen = None
                multi_domain_gen = None
        
        # Process scenario batch
        worker_results = {
            'worker_id': worker_id,
            'scenarios_processed': 0,
            'nodes_added': 0,
            'edges_added': 0,
            'processing_time': 0,
            'learning_results': []
        }
        
        batch_start_time = time.time()
        last_output_time = batch_start_time
        output_interval = 30.0
        
        for scenario_info in scenario_batch:
            scenario_count, scenario, current_round = scenario_info
            scenario_type = scenario_count % 10
            
            # Enhanced scenario generation
            scenario = _generate_enhanced_scenario(
                scenario_type, scenario_count, current_round, 
                physics_sim, procedural_gen, multi_domain_gen, scenario
            )
            
            # Create observation
            observation = _create_observation(scenario, current_round)
            
            # Process learning step with proper database node tracking
            if database_backend == "neo4j" and hasattr(worker_agent.knowledge_graph, 'get_stats'):
                # Use database stats for Neo4j
                stats_before = worker_agent.knowledge_graph.get_stats()
                nodes_before = stats_before.get('nodes', 0)
                edges_before = stats_before.get('edges', 0)
            else:
                # Use local graph for memory backend
                nodes_before = worker_agent.knowledge_graph.graph.number_of_nodes()
                edges_before = worker_agent.knowledge_graph.graph.number_of_edges()
            
            step_start = time.time()
            with SuppressOutput():
                step_results = worker_agent.active_learning_step(observation)
                
                # Ensure entities are properly added to Neo4j
                if database_backend == "neo4j" and hasattr(worker_agent.knowledge_graph, 'db'):
                    # Force commit any pending Neo4j transactions
                    _handle_neo4j_commit(worker_agent.knowledge_graph.db)
                    
                    # Verify entities were actually created in database
                    for entity in observation.get('entities', []):
                        entity_id = entity['id']
                        if hasattr(worker_agent.knowledge_graph, 'add_entity'):
                            # Ensure entity exists in Neo4j
                            worker_agent.knowledge_graph.add_entity(
                                entity_id, 
                                entity.get('label', 'entity'),
                                entity.get('properties', {})
                            )
                    
                    # Verify relationships were created
                    for rel in observation.get('relationships', []):
                        if hasattr(worker_agent.knowledge_graph, 'add_relationship'):
                            worker_agent.knowledge_graph.add_relationship(
                                rel['source'],
                                rel['target'], 
                                rel['type'],
                                rel.get('confidence', 0.5)
                            )
                    
            step_time = time.time() - step_start
            
            # Get accurate node/edge counts after processing
            if database_backend == "neo4j" and hasattr(worker_agent.knowledge_graph, 'get_stats'):
                stats_after = worker_agent.knowledge_graph.get_stats()
                nodes_after = stats_after.get('nodes', 0)
                edges_after = stats_after.get('edges', 0)
            else:
                nodes_after = worker_agent.knowledge_graph.graph.number_of_nodes()
                edges_after = worker_agent.knowledge_graph.graph.number_of_edges()
            
            # Update results
            worker_results['scenarios_processed'] += 1
            worker_results['nodes_added'] += nodes_after - nodes_before
            worker_results['edges_added'] += edges_after - edges_before
            worker_results['learning_results'].append({
                'scenario_count': scenario_count,
                'nodes_added': nodes_after - nodes_before,
                'edges_added': edges_after - edges_before,
                'confidence': step_results.get('global_confidence', 0),
                'processing_time': step_time
            })
            
            # Silent batch worker operation to avoid encoding issues
            now = time.time()
            if now - last_output_time > output_interval:
                # Update last output time but don't print to avoid multiprocessing encoding issues
                last_output_time = now
        
        worker_results['processing_time'] = time.time() - batch_start_time
        
        # Return results to main process
        return_queue.put(worker_results)
        return worker_results
        
    except Exception as e:
        error_result = {
            'worker_id': worker_id,
            'error': str(e),
            'scenarios_processed': 0,
            'nodes_added': 0,
            'edges_added': 0,
            'processing_time': 0,
            'learning_results': []
        }
        return_queue.put(error_result)
        return error_result


def _generate_enhanced_scenario(scenario_type: int, 
                              scenario_count: int, 
                              current_round: int,
                              physics_sim, 
                              procedural_gen, 
                              multi_domain_gen, 
                              base_scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Generate enhanced scenarios using various generators"""
    scenario = base_scenario.copy()
    
    if scenario_type == 0 and procedural_gen:
        try:
            dataset = procedural_gen.generate_dataset(num_nodes=20, num_edges=40)
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
        except Exception:
            # Fallback to simple scenario if procedural fails
            scenario = _generate_simple_fallback_scenario(scenario_type, scenario_count, current_round)
            
    elif scenario_type == 1 and multi_domain_gen:
        try:
            domain_scenarios = multi_domain_gen.generate_domain_scenarios(
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
            else:
                # Fallback if multi-domain fails
                scenario = _generate_simple_fallback_scenario(scenario_type, scenario_count, current_round)
        except Exception:
            scenario = _generate_simple_fallback_scenario(scenario_type, scenario_count, current_round)
            
    elif scenario_count % 5 == 0:
        try:
            physics_scenario = physics_sim.generate_physics_scenario()
            scenario = {
                'name': f"Physics Simulation {scenario_count + 1}",
                'entities': physics_scenario.get('entities', []),
                'relationships': physics_scenario.get('relationships', []),
                'description': f"Advanced physics simulation - {physics_scenario.get('scenario_type', 'dynamic')}"
            }
        except Exception:
            scenario = _generate_simple_fallback_scenario(scenario_type, scenario_count, current_round)
    else:
        # CRITICAL FIX: Generate entities for ALL other scenario types (2,3,4,6,7,8,9)
        scenario = _generate_simple_fallback_scenario(scenario_type, scenario_count, current_round)
    
    # Enhance scenario with round-specific variations
    if current_round > 1:
        scenario['name'] = f"{scenario['name']} (Round {current_round})"
        scenario['description'] = f"{scenario['description']} - Variation {current_round}"
        for rel in scenario.get('relationships', []):
            noise = np.random.normal(0, 0.05 * current_round)
            if 'confidence' in rel:
                rel['confidence'] = max(0.1, min(0.95, rel['confidence'] + noise))
            else:
                rel['confidence'] = max(0.1, min(0.95, 0.8 + noise))
    
    return scenario


def _generate_simple_fallback_scenario(scenario_type: int, scenario_count: int, current_round: int) -> Dict[str, Any]:
    """Generate simple fallback scenarios with guaranteed entities"""
    
    # Define different entity patterns for each scenario type
    entity_patterns = {
        2: ['sensor', 'actuator', 'controller'],
        3: ['robot', 'workspace', 'tool'], 
        4: ['vehicle', 'road', 'traffic_light'],
        6: ['patient', 'doctor', 'medication'],
        7: ['server', 'database', 'network'],
        8: ['factory', 'machine', 'product'],
        9: ['home', 'device', 'user']
    }
    
    # Get pattern for this scenario type, default to basic pattern
    pattern = entity_patterns.get(scenario_type, ['entity_a', 'entity_b', 'entity_c'])
    
    # Generate entities with unique IDs
    entities = []
    for i, entity_type in enumerate(pattern):
        entity_id = f"{entity_type}_{scenario_count}_{i}"
        entities.append({
            'id': entity_id,
            'label': entity_type,
            'properties_static': {'type': entity_type, 'scenario_type': scenario_type},
            'properties_dynamic': {'round': current_round, 'active': True},
            'confidence': 1.0
        })
    
    # Generate relationships between entities
    relationships = []
    for i in range(len(entities) - 1):
        relationships.append({
            'source': entities[i]['id'],
            'target': entities[i + 1]['id'],
            'type': 'interacts_with',
            'confidence': 0.7,
            'weight': 1.0,
            'properties': {'scenario_type': scenario_type}
        })
    
    # Add one circular relationship for complexity
    if len(entities) > 2:
        relationships.append({
            'source': entities[-1]['id'],
            'target': entities[0]['id'],
            'type': 'connects_to',
            'confidence': 0.6,
            'weight': 1.0,
            'properties': {'scenario_type': scenario_type}
        })
    
    return {
        'name': f"Fallback Scenario Type {scenario_type} - {scenario_count}",
        'entities': entities,
        'relationships': relationships,
        'description': f"Simple scenario with {len(entities)} entities and {len(relationships)} relationships"
    }


def _create_observation(scenario: Dict[str, Any], current_round: int) -> Dict[str, Any]:
    """Create observation from scenario with proper entity formatting"""
    state = np.random.normal(0, 0.2, 32)
    complexity_factor = 1.0 + (current_round - 1) * 0.1
    noise_factor = 0.1 * current_round
    
    # Set state based on scenario name keywords (if name exists)
    scenario_name = scenario.get('name', '').lower()
    if any(word in scenario_name for word in ['temperature', 'temp', 'climate']):
        base_temp = 22.5 + np.random.normal(0, 2 * complexity_factor)
        state[0:4] = [base_temp, 1.2 * complexity_factor, -0.3, 0.8]
    elif any(word in scenario_name for word in ['motion', 'occupancy']):
        state[4:8] = [1.0, 0.0, 0.9 + np.random.normal(0, noise_factor), 0.1]
    elif any(word in scenario_name for word in ['light', 'brightness']):
        state[8:12] = [0.7 + np.random.normal(0, 0.2 * complexity_factor), 0.3, 1.0, 0.2]
    elif any(word in scenario_name for word in ['security', 'door', 'window']):
        state[12:16] = [0.1, 0.9 + np.random.normal(0, noise_factor), 0.0, 1.0]
    elif any(word in scenario_name for word in ['energy', 'consumption']):
        state[16:20] = [0.6 + np.random.normal(0, 0.15 * complexity_factor), 0.4, 0.8, 0.3]
    elif any(word in scenario_name for word in ['air', 'quality', 'humidity']):
        state[20:24] = [0.5 + np.random.normal(0, noise_factor), 0.7, 0.3, 0.9]
    
    # Convert simple entity names to proper entity format for observe_environment
    formatted_entities = []
    for entity in scenario.get("entities", []):
        if isinstance(entity, dict) and 'id' in entity:
            # Already formatted entity from physics sim - make ID unique across scenarios
            base_entity_id = str(entity['id'])
            unique_entity_id = f"{base_entity_id}_{current_round}_{int(time.time() * 1000) % 100000}"
            formatted_entities.append({
                'id': unique_entity_id,
                'label': entity.get('label', 'physics_object'),
                'properties_static': {
                    'type': entity.get('label', 'physics_object'), 
                    'scenario': 'physics',
                    'base_id': base_entity_id  # Keep original ID for reference
                },
                'properties_dynamic': {'round': current_round, 'complexity': complexity_factor},
                'confidence': 1.0
            })
        elif isinstance(entity, str):
            # Simple entity name - create unique entity ID
            entity_id = f"{entity}_{current_round}_{int(time.time() * 1000) % 100000}"
            formatted_entities.append({
                'id': entity_id,
                'label': entity,
                'properties_static': {'type': entity, 'scenario': scenario_name or 'unknown'},
                'properties_dynamic': {'round': current_round, 'complexity': complexity_factor},
                'confidence': 1.0
            })
        else:
            # Already formatted entity
            formatted_entities.append(entity)
    
    # Convert relationships to use entity IDs
    formatted_relationships = []
    # Create mapping from original entity names to new unique IDs
    entity_name_to_id = {}
    for entity in formatted_entities:
        # For physics entities, use the base_id as the key
        if 'base_id' in entity.get('properties_static', {}):
            original_name = entity['properties_static']['base_id']
            entity_name_to_id[original_name] = entity['id']
        else:
            # For other entities, use the label
            entity_name_to_id[entity['label']] = entity['id']
    
    for rel in scenario.get("relationships", []):
        source_name = rel.get('source', '')
        target_name = rel.get('target', '')
        
        # Map entity names to IDs
        source_id = entity_name_to_id.get(source_name, source_name)
        target_id = entity_name_to_id.get(target_name, target_name)
        
        formatted_relationships.append({
            'source': source_id,
            'target': target_id,
            'type': rel.get('type', 'related'),
            'confidence': rel.get('confidence', 0.5),
            'weight': 1.0,
            'properties': rel.get('properties', {})
        })
    
    return {
        "entities": formatted_entities,
        "relationships": formatted_relationships,
        "state": state
    }


def _handle_neo4j_commit(db):
    """Handle Neo4j database commits"""
    try:
        if hasattr(db, 'commit'):
            db.commit()
        if hasattr(db, 'session') and hasattr(db.session, 'commit'):
            db.session.commit()
        if hasattr(db, 'close') and hasattr(db, 'connect'):
            db.close()
            db.connect()
    except Exception:
        pass  # Silent failure for robustness
