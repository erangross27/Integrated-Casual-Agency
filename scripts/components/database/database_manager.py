#!/usr/bin/env python3
"""
Database Operations Module
Handles AGI learning data storage and retrieval from Neo4j database
"""

import time


class DatabaseManager:
    """Manages database operations for AGI learning data"""
    
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.storage_stats = {
            'concepts_stored': 0,
            'hypotheses_stored': 0,
            'relationships_stored': 0
        }
    
    def store_agi_learning(self, agi_agent):
        """Store what the AGI actually learned to Neo4j database"""
        if not agi_agent or not self.kg:
            return False
        
        # Simple database connectivity check
        if not hasattr(self.kg, 'connected') or not self.kg.connected:
            return False
        
        try:
            # Get AGI's actual learning data
            knowledge_base = agi_agent.get_knowledge_base()
            active_hypotheses = agi_agent.get_active_hypotheses()
            confirmed_hypotheses = agi_agent.get_confirmed_hypotheses()
            causal_models = agi_agent.get_causal_models()
            
            # Check if there's actually new learning to store
            total_learning_data = (
                len(knowledge_base) + 
                len(active_hypotheses) + 
                len(confirmed_hypotheses) + 
                len(causal_models)
            )
            
            if total_learning_data == 0:
                return False
                
            stored_concepts = 0
            stored_hypotheses = 0
            stored_relationships = 0
            
            # Display learning progress before storing
            progress = agi_agent.learning_progress
            if progress['concepts_learned'] > 0 or progress['hypotheses_formed'] > 0:
                print(f"💾 [DB] Writing Learning Progress to Neo4j:")
                print(f"   • Concepts to Store: {len(knowledge_base)}")
                print(f"   • Active Hypotheses: {len(active_hypotheses)}")
                print(f"   • Confirmed Hypotheses: {len(confirmed_hypotheses)}")
                print(f"   • Causal Models: {len(causal_models)}")
                print(f"   • Total Learning Progress: {progress['concepts_learned']} concepts, {progress['hypotheses_formed']} hypotheses")
            
            # Store AGI's learned concepts
            if knowledge_base and len(knowledge_base) > 0:
                for i, concept in enumerate(knowledge_base):
                    if isinstance(concept, dict):
                        concept_id = f"agi_concept_{concept.get('id', i)}"
                        concept_name = concept.get('name', f'Concept_{i}')
                        concept_confidence = concept.get('confidence', 0.5)
                    else:
                        # Handle string or other types
                        concept_id = f"agi_concept_{i}_{hash(str(concept)) % 10000}"
                        concept_name = str(concept)[:50] if len(str(concept)) > 50 else str(concept)
                        concept_confidence = 0.5
                    
                    concept_entity = {
                        'id': concept_id,
                        'type': 'agi_concept',
                        'name': concept_name,
                        'confidence': concept_confidence,
                        'learned_at': time.time(),
                        'source': 'agi_learning'
                    }
                    
                    if self.kg.add_entity(concept_entity):
                        stored_concepts += 1
            
            # Store active hypotheses
            if active_hypotheses and len(active_hypotheses) > 0:
                for i, hypothesis in enumerate(active_hypotheses):
                    hypothesis_id = f"agi_hypothesis_{i}_{int(time.time() * 1000)}"
                    hypothesis_entity = {
                        'id': hypothesis_id,
                        'type': 'agi_hypothesis',
                        'description': str(hypothesis)[:100],
                        'status': 'active',
                        'created_at': time.time(),
                        'source': 'agi_learning'
                    }
                    
                    if self.kg.add_entity(hypothesis_entity):
                        stored_hypotheses += 1
            
            # Store confirmed hypotheses
            if confirmed_hypotheses and len(confirmed_hypotheses) > 0:
                for i, hypothesis in enumerate(confirmed_hypotheses):
                    hypothesis_id = f"agi_confirmed_hypothesis_{i}_{int(time.time() * 1000)}"
                    hypothesis_entity = {
                        'id': hypothesis_id,
                        'type': 'agi_hypothesis',
                        'description': str(hypothesis)[:100],
                        'status': 'confirmed',
                        'created_at': time.time(),
                        'source': 'agi_learning'
                    }
                    
                    if self.kg.add_entity(hypothesis_entity):
                        stored_hypotheses += 1
            
            # Update statistics
            self.storage_stats['concepts_stored'] += stored_concepts
            self.storage_stats['hypotheses_stored'] += stored_hypotheses
            self.storage_stats['relationships_stored'] += stored_relationships
            
            # Log what was actually stored
            if stored_concepts > 0 or stored_hypotheses > 0 or stored_relationships > 0:
                print(f"💾 [DB] ✅ Neo4j Write Complete: {stored_concepts} concepts, {stored_hypotheses} hypotheses, {stored_relationships} relationships")
                return True
            else:
                # Show progress status when AGI is developing
                total_progress = progress['concepts_learned'] + progress['hypotheses_formed'] + progress['causal_relationships_discovered']
                if total_progress > 0:
                    print(f"📊 [DB] Learning Progress: {total_progress} total progress - Building concepts before hypothesis formation")
                    
                    # Estimate when meaningful learning will begin
                    if progress['concepts_learned'] > 0 and progress['hypotheses_formed'] == 0:
                        print(f"🎯 [DB] Next Milestone: Hypothesis formation expected after observing pattern variations")
                    elif progress['hypotheses_formed'] > 0 and progress['causal_relationships_discovered'] == 0:
                        print(f"🎯 [DB] Next Milestone: Causal discovery expected after hypothesis testing")
                
                return False
            
        except Exception as e:
            print(f"[AGI] ⚠️ Error storing AGI learning: {e}")
            return False
    
    def get_storage_stats(self):
        """Get database storage statistics"""
        return self.storage_stats.copy()
