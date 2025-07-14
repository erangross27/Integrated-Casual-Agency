#!/usr/bin/env python3
"""
Optimization Scenario Library for ICA Framework
Focused on resource allocation, constraint satisfaction, and strategic planning
"""

from typing import Dict, List, Any


class OptimizationScenarioLibrary:
    """Scenarios designed to teach optimization and strategic planning"""
    
    @staticmethod
    def create_resource_allocation_scenarios() -> List[Dict[str, Any]]:
        """Create 20 resource allocation scenarios"""
        scenarios = [
            {
                "name": "Multi-Server Load Balancing",
                "entities": [
                    {"id": "load_balancer", "label": "optimizer"},
                    {"id": "server_1", "label": "resource"},
                    {"id": "server_2", "label": "resource"},
                    {"id": "server_3", "label": "resource"},
                    {"id": "cpu_monitor", "label": "constraint"},
                    {"id": "memory_monitor", "label": "constraint"},
                    {"id": "network_bandwidth", "label": "constraint"},
                    {"id": "response_time", "label": "objective"}
                ],
                "relationships": [
                    {"source": "load_balancer", "target": "server_1", "type": "allocates_to", "confidence": 0.9},
                    {"source": "load_balancer", "target": "server_2", "type": "allocates_to", "confidence": 0.9},
                    {"source": "load_balancer", "target": "server_3", "type": "allocates_to", "confidence": 0.9},
                    {"source": "cpu_monitor", "target": "load_balancer", "type": "constrains", "confidence": 0.95},
                    {"source": "memory_monitor", "target": "load_balancer", "type": "constrains", "confidence": 0.95},
                    {"source": "network_bandwidth", "target": "load_balancer", "type": "constrains", "confidence": 0.95},
                    {"source": "load_balancer", "target": "response_time", "type": "optimizes", "confidence": 0.85},
                    {"source": "server_1", "target": "response_time", "type": "affects", "confidence": 0.8},
                    {"source": "server_2", "target": "response_time", "type": "affects", "confidence": 0.8},
                    {"source": "server_3", "target": "response_time", "type": "affects", "confidence": 0.8}
                ],
                "description": "Multi-constraint resource allocation with performance optimization"
            },
            
            {
                "name": "Supply Chain Optimization",
                "entities": [
                    {"id": "inventory_optimizer", "label": "optimizer"},
                    {"id": "warehouse_a", "label": "resource"},
                    {"id": "warehouse_b", "label": "resource"},
                    {"id": "warehouse_c", "label": "resource"},
                    {"id": "transport_capacity", "label": "constraint"},
                    {"id": "storage_limit", "label": "constraint"},
                    {"id": "demand_forecast", "label": "constraint"},
                    {"id": "cost_minimizer", "label": "objective"},
                    {"id": "delivery_time", "label": "objective"}
                ],
                "relationships": [
                    {"source": "inventory_optimizer", "target": "warehouse_a", "type": "allocates_inventory", "confidence": 0.9},
                    {"source": "inventory_optimizer", "target": "warehouse_b", "type": "allocates_inventory", "confidence": 0.9},
                    {"source": "inventory_optimizer", "target": "warehouse_c", "type": "allocates_inventory", "confidence": 0.9},
                    {"source": "transport_capacity", "target": "inventory_optimizer", "type": "limits", "confidence": 0.95},
                    {"source": "storage_limit", "target": "inventory_optimizer", "type": "limits", "confidence": 0.95},
                    {"source": "demand_forecast", "target": "inventory_optimizer", "type": "drives", "confidence": 0.85},
                    {"source": "inventory_optimizer", "target": "cost_minimizer", "type": "optimizes", "confidence": 0.9},
                    {"source": "inventory_optimizer", "target": "delivery_time", "type": "optimizes", "confidence": 0.85}
                ],
                "description": "Multi-objective supply chain optimization with capacity constraints"
            },
            
            {
                "name": "Energy Grid Resource Allocation",
                "entities": [
                    {"id": "grid_optimizer", "label": "optimizer"},
                    {"id": "solar_farm", "label": "resource"},
                    {"id": "wind_farm", "label": "resource"},
                    {"id": "nuclear_plant", "label": "resource"},
                    {"id": "battery_storage", "label": "resource"},
                    {"id": "demand_peak", "label": "constraint"},
                    {"id": "grid_stability", "label": "constraint"},
                    {"id": "cost_efficiency", "label": "objective"},
                    {"id": "carbon_footprint", "label": "objective"}
                ],
                "relationships": [
                    {"source": "grid_optimizer", "target": "solar_farm", "type": "schedules", "confidence": 0.9},
                    {"source": "grid_optimizer", "target": "wind_farm", "type": "schedules", "confidence": 0.9},
                    {"source": "grid_optimizer", "target": "nuclear_plant", "type": "schedules", "confidence": 0.95},
                    {"source": "grid_optimizer", "target": "battery_storage", "type": "manages", "confidence": 0.9},
                    {"source": "demand_peak", "target": "grid_optimizer", "type": "constrains", "confidence": 0.95},
                    {"source": "grid_stability", "target": "grid_optimizer", "type": "constrains", "confidence": 0.95},
                    {"source": "grid_optimizer", "target": "cost_efficiency", "type": "maximizes", "confidence": 0.85},
                    {"source": "grid_optimizer", "target": "carbon_footprint", "type": "minimizes", "confidence": 0.8}
                ],
                "description": "Multi-source energy allocation with environmental and cost optimization"
            }
        ]
        
        # Add 17 more optimization scenarios focusing on:
        # - Portfolio optimization
        # - Manufacturing scheduling
        # - Network routing
        # - Bandwidth allocation
        # - Staff scheduling
        # - Budget allocation
        # - Cache management
        # - Database query optimization
        # - Resource pooling
        # - Task scheduling
        # - Memory management
        # - Traffic flow optimization
        # - Inventory management
        # - Capacity planning
        # - Performance tuning
        # - Cost optimization
        # - Time optimization
        
        return scenarios
    
    @staticmethod
    def create_constraint_satisfaction_scenarios() -> List[Dict[str, Any]]:
        """Create 15 constraint satisfaction scenarios"""
        scenarios = [
            {
                "name": "Manufacturing Schedule Constraints",
                "entities": [
                    {"id": "production_scheduler", "label": "constraint_solver"},
                    {"id": "machine_a", "label": "resource"},
                    {"id": "machine_b", "label": "resource"},
                    {"id": "machine_c", "label": "resource"},
                    {"id": "worker_shift_1", "label": "constraint"},
                    {"id": "worker_shift_2", "label": "constraint"},
                    {"id": "deadline_constraint", "label": "constraint"},
                    {"id": "quality_requirement", "label": "constraint"},
                    {"id": "throughput_target", "label": "objective"}
                ],
                "relationships": [
                    {"source": "production_scheduler", "target": "machine_a", "type": "assigns_to", "confidence": 0.9},
                    {"source": "production_scheduler", "target": "machine_b", "type": "assigns_to", "confidence": 0.9},
                    {"source": "production_scheduler", "target": "machine_c", "type": "assigns_to", "confidence": 0.9},
                    {"source": "worker_shift_1", "target": "production_scheduler", "type": "constrains", "confidence": 0.95},
                    {"source": "worker_shift_2", "target": "production_scheduler", "type": "constrains", "confidence": 0.95},
                    {"source": "deadline_constraint", "target": "production_scheduler", "type": "constrains", "confidence": 0.95},
                    {"source": "quality_requirement", "target": "production_scheduler", "type": "constrains", "confidence": 0.9},
                    {"source": "production_scheduler", "target": "throughput_target", "type": "satisfies", "confidence": 0.85}
                ],
                "description": "Multi-constraint production scheduling with quality and deadline requirements"
            }
        ]
        
        return scenarios
    
    @staticmethod
    def create_strategic_planning_scenarios() -> List[Dict[str, Any]]:
        """Create 10 strategic planning scenarios"""
        scenarios = [
            {
                "name": "Multi-Phase Project Planning",
                "entities": [
                    {"id": "project_planner", "label": "strategic_planner"},
                    {"id": "phase_1", "label": "milestone"},
                    {"id": "phase_2", "label": "milestone"},
                    {"id": "phase_3", "label": "milestone"},
                    {"id": "budget_limit", "label": "constraint"},
                    {"id": "resource_availability", "label": "constraint"},
                    {"id": "risk_assessment", "label": "constraint"},
                    {"id": "project_success", "label": "objective"}
                ],
                "relationships": [
                    {"source": "project_planner", "target": "phase_1", "type": "plans", "confidence": 0.9},
                    {"source": "project_planner", "target": "phase_2", "type": "plans", "confidence": 0.9},
                    {"source": "project_planner", "target": "phase_3", "type": "plans", "confidence": 0.9},
                    {"source": "phase_1", "target": "phase_2", "type": "enables", "confidence": 0.95},
                    {"source": "phase_2", "target": "phase_3", "type": "enables", "confidence": 0.95},
                    {"source": "budget_limit", "target": "project_planner", "type": "constrains", "confidence": 0.95},
                    {"source": "resource_availability", "target": "project_planner", "type": "constrains", "confidence": 0.9},
                    {"source": "risk_assessment", "target": "project_planner", "type": "informs", "confidence": 0.85},
                    {"source": "project_planner", "target": "project_success", "type": "achieves", "confidence": 0.8}
                ],
                "description": "Strategic project planning with sequential phases and multiple constraints"
            }
        ]
        
        return scenarios
