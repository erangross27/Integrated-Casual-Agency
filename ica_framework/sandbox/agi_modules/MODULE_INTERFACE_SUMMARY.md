"""
AGI MODULES INTERFACE SUMMARY
============================

This document maps all the modules created and their public interfaces.
Use this to ensure compatibility between modules and the main AGI agent.

1. MEMORY_SYSTEM (memory_system.py)
   Public Methods:
   - store_short_term(memory)
   - store_long_term(memory) 
   - store_episodic(episode)
   - get_recent_memories(count)
   - consolidate_memory(memory)
   - search_memories(query_type)
   - get_memory_usage()
   - store_experience(experience)  [Added for AGI agent compatibility]
   - recall_by_type(memory_type)   [Added for AGI agent compatibility]
   - get_memory_summary()          [Added for AGI agent compatibility]
   - clear_memories()

2. LEARNING_PROGRESS (learning_progress.py)
   Public Methods:
   - update_concepts(count)
   - update_hypotheses_formed(count)
   - update_hypotheses_confirmed(count)
   - update_causal_relationships(count)
   - update_patterns(count)
   - process_gpu_discoveries(gpu_results, cycle_count)
   - record_learning_event(description)
   - get_progress()
   - get_progress_summary()        [Added for AGI agent compatibility]
   - update_metrics(metrics_update) [Added for AGI agent compatibility]
   - get_learning_rate()
   - reset_progress()

3. HYPOTHESIS_MANAGER (hypothesis_manager.py)
   Public Methods:
   - generate_hypothesis(question, context)
   - test_hypothesis(hypothesis_id, evidence)
   - confirm_hypothesis(hypothesis_id)
   - reject_hypothesis(hypothesis_id)
   - get_active_hypotheses()
   - get_confirmed_hypotheses()
   - get_hypothesis_summary()      [Added for AGI agent compatibility]
   - get_hypothesis_stats()

4. CAUSAL_REASONING (causal_reasoning.py)
   Public Methods:
   - observe_event(event)
   - analyze_causal_relationship(event1, event2)
   - discover_causal_patterns()
   - predict_effect(cause_event)
   - get_causal_relationships()
   - get_causal_confidence(cause, effect)

5. SENSORY_PROCESSOR (sensory_processor.py)
   Public Methods:
   - process_sensory_input(sensory_input)
   - get_object_motion_summary(obj_id)
   Private Methods:
   - _process_visual_input(visual_data)
   - _merge_physics_data(visual_objects, physics_objects)
   - _extract_object_properties(obj_data)
   - _update_object_tracker(obj_id, obj_data)
   - _detect_motion_events(objects)
   - _detect_collision_events(objects)
   - _objects_colliding(obj1, obj2)

6. ATTENTION_SYSTEM (attention_system.py)
   Public Methods:
   - process_stimuli(stimuli, current_goals)
   - focus_on_target(target_id, boost_factor)
   - ignore_target(target_id)
   - get_current_focus()
   - shift_attention(new_stimuli, trigger)
   - get_attention_summary()
   - get_attention_patterns()

7. CURIOSITY_ENGINE (curiosity_engine.py)
   Public Methods:
   - assess_curiosity(observations, current_knowledge)
   - predict_next_event(recent_events, context)
   - get_curiosity_summary()
   - get_investigation_recommendations()

8. PATTERN_LEARNER (pattern_learner.py)
   Public Methods:
   - learn_patterns(observations, context)
   - predict_next_event(recent_events, context)
   - get_pattern_summary()

9. PHYSICS_LEARNER (physics_learner.py)
   Public Methods:
   - learn_from_observation(objects, motion_events)
   - generate_physics_hypothesis(concept_name)
   - predict_behavior(obj_data)
   - get_physics_knowledge()
   - get_confidence_summary()

10. EXPLORATION_CONTROLLER (exploration_controller.py)
    Public Methods:
    - plan_exploration(current_position, environment_info, curiosity_targets)
    - add_goal(goal)
    - update_goal_progress(current_position)
    - get_exploration_summary()
    - get_recommended_action(current_position, environment_info, curiosity_targets)

COMPATIBILITY NOTES:
===================

The AGI agent expects these method calls:
- memory_system.store_experience()          ✅ Added
- memory_system.recall_by_type()            ✅ Added  
- memory_system.get_memory_summary()        ✅ Added
- learning_progress.get_progress_summary()  ✅ Added
- learning_progress.update_metrics()        ✅ Added
- learning_progress.process_gpu_discoveries() ✅ Exists
- hypothesis_manager.get_hypothesis_summary() ✅ Added
- All other expected methods exist in the modules

INPUT VALIDATION ADDED:
======================
- sensory_processor.process_sensory_input() - validates dict input
- sensory_processor._process_visual_input() - validates dict input  
- sensory_processor._extract_object_properties() - validates dict input
- physics_learner._learn_gravity_effects() - validates dict obj_data

REMAINING ISSUES:
================
The "'str' object has no attribute 'get'" error suggests that somewhere
in the system, string data is being passed where dictionary data is expected.
The validation added should catch and handle these cases gracefully.
"""
