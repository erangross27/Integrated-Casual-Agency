#!/usr/bin/env python3
"""
Concept Storage Module
Handles storage and retrieval of AGI concepts
"""

import time


class ConceptStorage:
    """Handles AGI concept storage operations"""
    
    def __init__(self, knowledge_graph, session_id):
        self.kg = knowledge_graph
        self.session_id = session_id
        self.concepts_stored = 0
    
    def store_concepts(self, knowledge_base):
        """Store AGI concepts from knowledge base"""
        if not knowledge_base or len(knowledge_base) == 0:
            return 0
        
        stored_count = 0
        
        for i, concept in enumerate(knowledge_base):
            if isinstance(concept, dict):
                concept_id = f"agi_concept_{concept.get('id', i)}_{self.session_id}"
                concept_name = concept.get('name', f'Concept_{i}')
                concept_confidence = concept.get('confidence', 0.5)
            else:
                concept_id = f"agi_concept_{i}_{hash(str(concept)) % 10000}_{self.session_id}"
                concept_name = str(concept)[:50] if len(str(concept)) > 50 else str(concept)
                concept_confidence = 0.5
            
            concept_entity = {
                'type': 'agi_concept',
                'name': concept_name,
                'confidence': concept_confidence,
                'learned_at': time.time(),
                'session_id': self.session_id,
                'source': 'agi_learning'
            }
            
            if self.kg.add_entity(concept_id, 'agi_concept', concept_entity):
                stored_count += 1
        
        self.concepts_stored += stored_count
        return stored_count
    
    def get_stored_concepts(self, session_id=None):
        """Get stored concepts from database"""
        if not self.kg:
            return []
        
        try:
            if session_id:
                query = f"MATCH (n:Entity {{type: 'agi_concept', session_id: '{session_id}'}}) RETURN n ORDER BY n.learned_at DESC"
            else:
                query = f"MATCH (n:Entity {{type: 'agi_concept'}}) RETURN n ORDER BY n.learned_at DESC"
            
            result = self.kg.execute_custom_query(query)
            
            concepts = []
            for record in result:
                concept_data = record.get('n', {})
                concepts.append({
                    'id': concept_data.get('id', 'unknown'),
                    'name': concept_data.get('name', 'Unnamed'),
                    'confidence': concept_data.get('confidence', 0.5),
                    'learned_at': concept_data.get('learned_at', 0),
                    'session_id': concept_data.get('session_id', 'unknown')
                })
            
            return concepts
            
        except Exception as e:
            print(f"[CONCEPT] ⚠️ Concept retrieval error: {e}")
            return []
    
    def restore_concepts_to_agent(self, agi_agent):
        """Restore concepts from database to AGI agent"""
        if not self.kg or not agi_agent:
            return False
        
        try:
            # Get all stored concepts from any session (for restoration)
            stored_concepts = self.get_stored_concepts()  # No session_id = get all
            
            if not stored_concepts:
                return False
            
            # Restore concepts to agent's knowledge base
            if hasattr(agi_agent, 'knowledge_base'):
                # Handle both dict and list knowledge bases
                if isinstance(agi_agent.knowledge_base, dict):
                    # If it's a dict, add concepts as key-value pairs
                    for concept in stored_concepts:
                        concept_name = concept['name']
                        concept_data = {
                            'type': 'concept',
                            'description': concept.get('description', f'Restored concept: {concept_name}'),
                            'evidence': concept.get('evidence', []),
                            'confidence': concept.get('confidence', 0.0),
                            'learned_at': concept.get('learned_at', time.time())
                        }
                        # Use concept name as key
                        agi_agent.knowledge_base[concept_name] = concept_data
                
                elif isinstance(agi_agent.knowledge_base, list):
                    # If it's a list, append concepts
                    for concept in stored_concepts:
                        concept_data = {
                            'type': 'concept',
                            'description': concept.get('description', f'Restored concept: {concept["name"]}'),
                            'evidence': concept.get('evidence', []),
                            'confidence': concept.get('confidence', 0.0),
                            'learned_at': concept.get('learned_at', time.time())
                        }
                        
                        # Check for duplicates by name
                        existing_names = [c.get('name') if isinstance(c, dict) else str(c) for c in agi_agent.knowledge_base]
                        if concept_data['name'] not in existing_names:
                            agi_agent.knowledge_base.append(concept_data)
                
                else:
                    # If it's neither dict nor list, initialize as dict
                    agi_agent.knowledge_base = {}
                    for concept in stored_concepts:
                        concept_name = concept['name']
                        concept_data = {
                            'type': 'concept',
                            'description': concept.get('description', f'Restored concept: {concept_name}'),
                            'evidence': concept.get('evidence', []),
                            'confidence': concept.get('confidence', 0.0),
                            'learned_at': concept.get('learned_at', time.time())
                        }
                        agi_agent.knowledge_base[concept_name] = concept_data
                
                print(f"💾 [CONCEPT] ✅ Restored {len(stored_concepts)} concepts to agent")
                return True
            
            return False
            
        except Exception as e:
            print(f"[CONCEPT] ⚠️ Concept restoration error: {e}")
            return False
    
    def get_stats(self):
        """Get concept storage statistics"""
        return {
            'concepts_stored': self.concepts_stored
        }
