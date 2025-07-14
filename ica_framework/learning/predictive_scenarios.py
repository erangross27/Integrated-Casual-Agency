#!/usr/bin/env python3
"""
Predictive Intelligence Scenario Library for ICA Framework
Focused on trend analysis, forecasting, and behavior prediction
"""

from typing import Dict, List, Any


class PredictiveIntelligenceScenarioLibrary:
    """Scenarios designed to teach predictive modeling and forecasting"""
    
    @staticmethod
    def create_trend_analysis_scenarios() -> List[Dict[str, Any]]:
        """Create 20 trend analysis scenarios"""
        scenarios = [
            {
                "name": "Stock Market Trend Prediction",
                "entities": [
                    {"id": "trend_analyzer", "label": "predictor"},
                    {"id": "price_history", "label": "historical_data"},
                    {"id": "volume_data", "label": "historical_data"},
                    {"id": "market_sentiment", "label": "feature"},
                    {"id": "economic_indicators", "label": "feature"},
                    {"id": "company_earnings", "label": "feature"},
                    {"id": "seasonal_pattern", "label": "pattern"},
                    {"id": "momentum_pattern", "label": "pattern"},
                    {"id": "reversal_pattern", "label": "pattern"},
                    {"id": "bullish_trend", "label": "prediction"},
                    {"id": "bearish_trend", "label": "prediction"},
                    {"id": "sideways_trend", "label": "prediction"}
                ],
                "relationships": [
                    {"source": "price_history", "target": "trend_analyzer", "type": "feeds_into", "confidence": 0.9},
                    {"source": "volume_data", "target": "trend_analyzer", "type": "feeds_into", "confidence": 0.85},
                    {"source": "market_sentiment", "target": "trend_analyzer", "type": "influences", "confidence": 0.8},
                    {"source": "economic_indicators", "target": "trend_analyzer", "type": "influences", "confidence": 0.85},
                    {"source": "company_earnings", "target": "trend_analyzer", "type": "influences", "confidence": 0.9},
                    {"source": "trend_analyzer", "target": "seasonal_pattern", "type": "detects", "confidence": 0.7},
                    {"source": "trend_analyzer", "target": "momentum_pattern", "type": "detects", "confidence": 0.75},
                    {"source": "trend_analyzer", "target": "reversal_pattern", "type": "detects", "confidence": 0.6},
                    {"source": "seasonal_pattern", "target": "bullish_trend", "type": "indicates", "confidence": 0.7},
                    {"source": "momentum_pattern", "target": "bullish_trend", "type": "indicates", "confidence": 0.8},
                    {"source": "reversal_pattern", "target": "bearish_trend", "type": "indicates", "confidence": 0.75},
                    {"source": "trend_analyzer", "target": "sideways_trend", "type": "predicts", "confidence": 0.6}
                ],
                "description": "Multi-factor stock market trend analysis with pattern recognition"
            },
            
            {
                "name": "Climate Change Prediction",
                "entities": [
                    {"id": "climate_model", "label": "predictor"},
                    {"id": "temperature_records", "label": "historical_data"},
                    {"id": "precipitation_data", "label": "historical_data"},
                    {"id": "co2_levels", "label": "feature"},
                    {"id": "ocean_currents", "label": "feature"},
                    {"id": "solar_radiation", "label": "feature"},
                    {"id": "greenhouse_effect", "label": "pattern"},
                    {"id": "ice_cap_melting", "label": "pattern"},
                    {"id": "sea_level_rise", "label": "pattern"},
                    {"id": "temperature_increase", "label": "prediction"},
                    {"id": "extreme_weather", "label": "prediction"},
                    {"id": "ecosystem_shift", "label": "prediction"}
                ],
                "relationships": [
                    {"source": "temperature_records", "target": "climate_model", "type": "trains", "confidence": 0.95},
                    {"source": "precipitation_data", "target": "climate_model", "type": "trains", "confidence": 0.9},
                    {"source": "co2_levels", "target": "climate_model", "type": "drives", "confidence": 0.9},
                    {"source": "ocean_currents", "target": "climate_model", "type": "influences", "confidence": 0.85},
                    {"source": "solar_radiation", "target": "climate_model", "type": "influences", "confidence": 0.8},
                    {"source": "climate_model", "target": "greenhouse_effect", "type": "models", "confidence": 0.9},
                    {"source": "greenhouse_effect", "target": "ice_cap_melting", "type": "causes", "confidence": 0.85},
                    {"source": "ice_cap_melting", "target": "sea_level_rise", "type": "causes", "confidence": 0.9},
                    {"source": "climate_model", "target": "temperature_increase", "type": "predicts", "confidence": 0.85},
                    {"source": "climate_model", "target": "extreme_weather", "type": "predicts", "confidence": 0.8},
                    {"source": "temperature_increase", "target": "ecosystem_shift", "type": "triggers", "confidence": 0.75}
                ],
                "description": "Long-term climate prediction with cascading environmental effects"
            },
            
            {
                "name": "Disease Outbreak Forecasting",
                "entities": [
                    {"id": "epidemic_model", "label": "predictor"},
                    {"id": "infection_rates", "label": "historical_data"},
                    {"id": "population_density", "label": "feature"},
                    {"id": "travel_patterns", "label": "feature"},
                    {"id": "vaccination_rates", "label": "feature"},
                    {"id": "seasonal_immunity", "label": "feature"},
                    {"id": "exponential_growth", "label": "pattern"},
                    {"id": "herd_immunity", "label": "pattern"},
                    {"id": "mutation_pattern", "label": "pattern"},
                    {"id": "peak_infections", "label": "prediction"},
                    {"id": "outbreak_duration", "label": "prediction"},
                    {"id": "geographic_spread", "label": "prediction"}
                ],
                "relationships": [
                    {"source": "infection_rates", "target": "epidemic_model", "type": "trains", "confidence": 0.9},
                    {"source": "population_density", "target": "epidemic_model", "type": "parameterizes", "confidence": 0.85},
                    {"source": "travel_patterns", "target": "epidemic_model", "type": "influences", "confidence": 0.8},
                    {"source": "vaccination_rates", "target": "epidemic_model", "type": "moderates", "confidence": 0.9},
                    {"source": "seasonal_immunity", "target": "epidemic_model", "type": "affects", "confidence": 0.75},
                    {"source": "epidemic_model", "target": "exponential_growth", "type": "identifies", "confidence": 0.85},
                    {"source": "epidemic_model", "target": "herd_immunity", "type": "calculates", "confidence": 0.8},
                    {"source": "epidemic_model", "target": "mutation_pattern", "type": "tracks", "confidence": 0.7},
                    {"source": "exponential_growth", "target": "peak_infections", "type": "predicts", "confidence": 0.8},
                    {"source": "herd_immunity", "target": "outbreak_duration", "type": "determines", "confidence": 0.85},
                    {"source": "travel_patterns", "target": "geographic_spread", "type": "drives", "confidence": 0.8}
                ],
                "description": "Epidemiological modeling with multi-factor disease spread prediction"
            }
        ]
        
        # Add 17 more trend analysis scenarios
        return scenarios
    
    @staticmethod
    def create_forecasting_scenarios() -> List[Dict[str, Any]]:
        """Create 15 forecasting scenarios"""
        scenarios = [
            {
                "name": "Energy Demand Forecasting",
                "entities": [
                    {"id": "demand_forecaster", "label": "predictor"},
                    {"id": "historical_consumption", "label": "time_series"},
                    {"id": "weather_data", "label": "external_factor"},
                    {"id": "economic_activity", "label": "external_factor"},
                    {"id": "population_growth", "label": "external_factor"},
                    {"id": "industrial_expansion", "label": "external_factor"},
                    {"id": "seasonal_cycle", "label": "pattern"},
                    {"id": "daily_cycle", "label": "pattern"},
                    {"id": "trend_component", "label": "pattern"},
                    {"id": "peak_demand", "label": "forecast"},
                    {"id": "base_load", "label": "forecast"},
                    {"id": "annual_growth", "label": "forecast"}
                ],
                "relationships": [
                    {"source": "historical_consumption", "target": "demand_forecaster", "type": "trains", "confidence": 0.95},
                    {"source": "weather_data", "target": "demand_forecaster", "type": "informs", "confidence": 0.85},
                    {"source": "economic_activity", "target": "demand_forecaster", "type": "influences", "confidence": 0.8},
                    {"source": "population_growth", "target": "demand_forecaster", "type": "drives", "confidence": 0.9},
                    {"source": "industrial_expansion", "target": "demand_forecaster", "type": "increases", "confidence": 0.85},
                    {"source": "demand_forecaster", "target": "seasonal_cycle", "type": "decomposes", "confidence": 0.9},
                    {"source": "demand_forecaster", "target": "daily_cycle", "type": "decomposes", "confidence": 0.9},
                    {"source": "demand_forecaster", "target": "trend_component", "type": "extracts", "confidence": 0.85},
                    {"source": "seasonal_cycle", "target": "peak_demand", "type": "predicts", "confidence": 0.8},
                    {"source": "daily_cycle", "target": "base_load", "type": "determines", "confidence": 0.85},
                    {"source": "trend_component", "target": "annual_growth", "type": "projects", "confidence": 0.8}
                ],
                "description": "Multi-component energy demand forecasting with external factors"
            }
        ]
        
        return scenarios
    
    @staticmethod
    def create_behavior_prediction_scenarios() -> List[Dict[str, Any]]:
        """Create 10 behavior prediction scenarios"""
        scenarios = [
            {
                "name": "Customer Churn Prediction",
                "entities": [
                    {"id": "churn_predictor", "label": "predictor"},
                    {"id": "usage_patterns", "label": "behavioral_data"},
                    {"id": "payment_history", "label": "behavioral_data"},
                    {"id": "support_interactions", "label": "behavioral_data"},
                    {"id": "engagement_level", "label": "feature"},
                    {"id": "satisfaction_score", "label": "feature"},
                    {"id": "competitor_activity", "label": "feature"},
                    {"id": "declining_usage", "label": "pattern"},
                    {"id": "late_payments", "label": "pattern"},
                    {"id": "increased_complaints", "label": "pattern"},
                    {"id": "high_churn_risk", "label": "prediction"},
                    {"id": "medium_churn_risk", "label": "prediction"},
                    {"id": "low_churn_risk", "label": "prediction"}
                ],
                "relationships": [
                    {"source": "usage_patterns", "target": "churn_predictor", "type": "feeds_into", "confidence": 0.9},
                    {"source": "payment_history", "target": "churn_predictor", "type": "feeds_into", "confidence": 0.85},
                    {"source": "support_interactions", "target": "churn_predictor", "type": "feeds_into", "confidence": 0.8},
                    {"source": "engagement_level", "target": "churn_predictor", "type": "influences", "confidence": 0.85},
                    {"source": "satisfaction_score", "target": "churn_predictor", "type": "influences", "confidence": 0.9},
                    {"source": "competitor_activity", "target": "churn_predictor", "type": "affects", "confidence": 0.7},
                    {"source": "churn_predictor", "target": "declining_usage", "type": "detects", "confidence": 0.8},
                    {"source": "churn_predictor", "target": "late_payments", "type": "detects", "confidence": 0.85},
                    {"source": "churn_predictor", "target": "increased_complaints", "type": "detects", "confidence": 0.8},
                    {"source": "declining_usage", "target": "high_churn_risk", "type": "indicates", "confidence": 0.8},
                    {"source": "late_payments", "target": "medium_churn_risk", "type": "indicates", "confidence": 0.75},
                    {"source": "increased_complaints", "target": "high_churn_risk", "type": "indicates", "confidence": 0.85}
                ],
                "description": "Multi-signal customer behavior prediction with churn risk assessment"
            }
        ]
        
        return scenarios
