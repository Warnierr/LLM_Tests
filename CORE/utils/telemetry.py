"""
Module d'observabilité pour Nina avec OpenTelemetry et logs structurés
"""
import logging
import structlog
import time
from typing import Dict, Any, Optional
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
import os
from contextlib import contextmanager

class NinaTelemetry:
    """Classe de gestion de l'observabilité Nina"""
    
    def __init__(self, service_name: str = "nina-ai"):
        self.service_name = service_name
        self.tracer = None
        self.meter = None
        self._initialized = False
        
        # Compteurs et métriques
        self.metrics = {}
        
    def initialize(self):
        """Initialise OpenTelemetry"""
        if self._initialized:
            return
            
        # Configuration du tracing
        trace.set_tracer_provider(TracerProvider())
        
        # Configuration des métriques basique
        metrics.set_meter_provider(MeterProvider())
        
        # Récupération des instances
        self.tracer = trace.get_tracer(self.service_name)
        self.meter = metrics.get_meter(self.service_name)
        
        # Création des métriques de base
        self._create_metrics()
        
        # Configuration du logging structuré
        self._setup_structured_logging()
        
        self._initialized = True
        
    def _create_metrics(self):
        """Crée les métriques métier de Nina"""
        if not self.meter:
            return
            
        # Compteurs
        self.metrics['messages_processed'] = self.meter.create_counter(
            name="nina_messages_processed_total",
            description="Nombre total de messages traités",
            unit="1"
        )
        
        self.metrics['llm_calls'] = self.meter.create_counter(
            name="nina_llm_calls_total", 
            description="Nombre d'appels LLM",
            unit="1"
        )
        
        self.metrics['memory_operations'] = self.meter.create_counter(
            name="nina_memory_operations_total",
            description="Opérations mémoire",
            unit="1"
        )
        
        # Histogrammes pour latences
        self.metrics['response_duration'] = self.meter.create_histogram(
            name="nina_response_duration_seconds",
            description="Temps de réponse Nina",
            unit="s"
        )
        
        self.metrics['llm_duration'] = self.meter.create_histogram(
            name="nina_llm_duration_seconds", 
            description="Temps d'appel LLM",
            unit="s"
        )
        
        # Gauges
        self.metrics['memory_size'] = self.meter.create_up_down_counter(
            name="nina_memory_items_current",
            description="Nombre d'éléments en mémoire",
            unit="1"
        )
        
    def _setup_structured_logging(self):
        """Configure le logging structuré avec structlog"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
    def get_logger(self, name: str):
        """Retourne un logger structuré"""
        return structlog.get_logger(name)
        
    @contextmanager
    def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager pour tracer une opération"""
        if not self.tracer:
            yield
            return
            
        with self.tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            yield span
            
    def record_message_processed(self, user_id: str, backend: str):
        """Enregistre un message traité"""
        try:
            if 'messages_processed' in self.metrics:
                self.metrics['messages_processed'].add(1, {
                    "user_id": user_id,
                    "backend": backend
                })
        except Exception:
            pass  # Ignore silencieusement les erreurs de métriques
            
    def record_llm_call(self, backend: str, model: str, duration: float, tokens: int = 0):
        """Enregistre un appel LLM"""
        try:
            if 'llm_calls' in self.metrics:
                self.metrics['llm_calls'].add(1, {
                    "backend": backend,
                    "model": model
                })
            if 'llm_duration' in self.metrics:
                self.metrics['llm_duration'].record(duration, {
                    "backend": backend,
                    "model": model
                })
        except Exception:
            pass  # Ignore silencieusement les erreurs de métriques
            
    def record_memory_operation(self, operation: str, items_count: int = 0):
        """Enregistre une opération mémoire"""
        try:
            if 'memory_operations' in self.metrics:
                self.metrics['memory_operations'].add(1, {"operation": operation})
            if 'memory_size' in self.metrics and items_count > 0:
                self.metrics['memory_size'].add(items_count)
        except Exception:
            pass  # Ignore silencieusement les erreurs de métriques
            
    def record_response_time(self, duration: float, backend: str):
        """Enregistre le temps de réponse total"""
        try:
            if 'response_duration' in self.metrics:
                self.metrics['response_duration'].record(duration, {"backend": backend})
        except Exception:
            pass  # Ignore silencieusement les erreurs de métriques

# Instance globale
telemetry = NinaTelemetry()

def init_telemetry(service_name: str = "nina-ai"):
    """Initialise la télémétrie globale"""
    global telemetry
    telemetry = NinaTelemetry(service_name)
    telemetry.initialize()
    return telemetry

def get_telemetry() -> NinaTelemetry:
    """Retourne l'instance de télémétrie"""
    return telemetry 