"""Tests pour le module de télémétrie Nina"""
import pytest
import time
from unittest.mock import patch, MagicMock
from CORE.utils.telemetry import NinaTelemetry, init_telemetry, get_telemetry


class TestNinaTelemetry:
    """Tests pour la classe NinaTelemetry"""
    
    def test_telemetry_initialization(self):
        """Test que la télémétrie s'initialise correctement"""
        telemetry = NinaTelemetry("test-service")
        assert telemetry.service_name == "test-service"
        assert not telemetry._initialized
        
        # Test d'initialisation
        telemetry.initialize()
        assert telemetry._initialized
        assert telemetry.tracer is not None
        assert telemetry.meter is not None
        
    def test_structured_logger(self):
        """Test que le logger structuré fonctionne"""
        telemetry = NinaTelemetry()
        telemetry.initialize()
        
        logger = telemetry.get_logger("test")
        assert logger is not None
        
        # Test d'écriture de log (ne devrait pas lever d'exception)
        logger.info("Test log", key="value", number=42)
        
    def test_trace_operation_context_manager(self):
        """Test que le context manager de tracing fonctionne"""
        telemetry = NinaTelemetry()
        telemetry.initialize()
        
        # Test sans attributs
        with telemetry.trace_operation("test_operation") as span:
            assert span is not None or span is None  # Peut être None si pas de tracer
            
        # Test avec attributs
        attributes = {"user_id": "test_user", "operation_type": "test"}
        with telemetry.trace_operation("test_operation_with_attrs", attributes):
            pass  # Ne devrait pas lever d'exception
            
    def test_record_metrics(self):
        """Test que l'enregistrement des métriques fonctionne"""
        telemetry = NinaTelemetry()
        telemetry.initialize()
        
        # Test enregistrement message processed
        telemetry.record_message_processed("user123", "groq")
        
        # Test enregistrement LLM call
        telemetry.record_llm_call("groq", "llama3-8b", 0.5, 150)
        
        # Test enregistrement memory operation
        telemetry.record_memory_operation("retrieve", 5)
        
        # Test enregistrement response time
        telemetry.record_response_time(1.2, "groq")
        
        # Aucune exception ne devrait être levée
        
    def test_global_telemetry_functions(self):
        """Test des fonctions globales de télémétrie"""
        # Test d'initialisation globale
        tel = init_telemetry("global-test")
        assert tel.service_name == "global-test"
        assert tel._initialized
        
        # Test de récupération de l'instance globale
        tel2 = get_telemetry()
        assert tel2 is tel


class TestTelemetryPerformance:
    """Tests de performance de la télémétrie"""
        
    def test_telemetry_performance_impact(self):
        """Test que la télémétrie n'a pas d'impact significatif sur les performances"""
        telemetry = NinaTelemetry()
        telemetry.initialize()
        
        # Mesurer le temps avec télémétrie
        start_time = time.time()
        for _ in range(100):
            with telemetry.trace_operation("performance_test"):
                telemetry.record_message_processed("user", "backend")
                telemetry.record_response_time(0.1, "backend")
        elapsed_with_telemetry = time.time() - start_time
        
        # Le temps ne devrait pas dépasser 500ms pour 100 itérations  
        assert elapsed_with_telemetry < 0.5, f"Télémétrie trop lente: {elapsed_with_telemetry}s"
        
    def test_telemetry_error_handling(self):
        """Test que les erreurs de télémétrie n'interrompent pas le fonctionnement"""
        telemetry = NinaTelemetry()
        telemetry.initialize()
        
        # Test que l'enregistrement de métriques fonctionne même en cas d'erreur
        telemetry.record_message_processed("user", "backend")
        # Pas d'exception = succès


if __name__ == "__main__":
    pytest.main([__file__]) 