import re
import time
from typing import List, Tuple, Any, Dict
from .schema import AuditStep, AuditTrace, ConstraintSchema
from .audit_sink_kafka import get_kafka_sink

logger = from constraint_lattice.logging_config import configure_logger
logger = configure_logger(__name__)(__name__)


class BaseEvaluator:
    """Base class for constraint evaluators"""
    
    def evaluate(self, text: str, params: Dict) -> Tuple[str, Dict]:
        """
        Evaluate constraint on text and return modified text with decision log
        
        Args:
            text: Input text to process
            params: Constraint parameters
            
        Returns:
            Tuple of (processed_text, decision_log)
        """
        raise NotImplementedError


class RegexEvaluator(BaseEvaluator):
    """Evaluator for regex-based constraints"""
    
    def evaluate(self, text: str, params: Dict) -> Tuple[str, Dict]:
        pattern = params.get("pattern")
        replacement = params.get("replacement", "")
        
        if not pattern:
            return text, {"error": "Missing pattern parameter"}
            
        try:
            # Compile regex for efficiency
            regex = re.compile(pattern)
            processed_text = regex.sub(replacement, text)
            
            return processed_text, {
                "matches": regex.findall(text),
                "replacements": len(regex.findall(text))
            }
        except re.error as e:
            return text, {"error": f"Regex error: {str(e)}"}


class TextEvaluator(BaseEvaluator):
    """Evaluator for text-based constraints"""
    
    def evaluate(self, text: str, params: Dict) -> Tuple[str, Dict]:
        target = params.get("target")
        replacement = params.get("replacement", "")
        
        if not target:
            return text, {"error": "Missing target parameter"}
            
        processed_text = text.replace(target, replacement)
        
        return processed_text, {
            "occurrences": text.count(target),
            "replacements": text.count(target)
        }


class SymbolicEvaluator(BaseEvaluator):
    """Evaluator for symbolic constraints using rule-based logic"""
    
    def evaluate(self, text: str, params: Dict) -> Tuple[str, Dict]:
        # Get constraint parameters
        rule_type = params.get("type")
        pattern = params.get("pattern")
        
        if not rule_type:
            return text, {"error": "Missing rule type"}
            
        if rule_type == "contains":
            # Check if text contains pattern
            if pattern in text:
                return text, {"match": True}
            else:
                return "", {"match": False}
        elif rule_type == "excludes":
            # Check if text does not contain pattern
            if pattern not in text:
                return text, {"match": True}
            else:
                return "", {"match": False}
        else:
            return text, {"error": f"Unknown rule type: {rule_type}"}


class SemanticEvaluator(BaseEvaluator):
    """Evaluator for semantic constraints using transformer models"""
    
    def __init__(self):
        try:
            from transformers import pipeline
            self.sentiment_analyzer = pipeline("sentiment-analysis")
        except ImportError:
            logger.warning("Transformers not installed. Semantic constraints will not work.")
            self.sentiment_analyzer = None
    
    def evaluate(self, text: str, params: Dict) -> Tuple[str, Dict]:
        # Check if transformers is available
        if not self.sentiment_analyzer:
            return text, {"error": "Transformers not installed"}
            
        # Get constraint parameters
        target_label = params.get("label", "POSITIVE")
        threshold = params.get("threshold", 0.7)
        
        # Analyze text
        result = self.sentiment_analyzer(text)[0]
        
        # Check if constraint is satisfied
        if result['label'] == target_label and result['score'] >= threshold:
            return text, {
                "label": result['label'],
                "score": result['score'],
                "satisfied": True
            }
        else:
            # Apply action - for now just block the text
            return "", {
                "label": result['label'],
                "score": result['score'],
                "satisfied": False
            }


# TODO: 


class BaseAgent:
    """Base class for constraint agents"""
    
    def __init__(self):
        self.evaluators = {
            "regex": RegexEvaluator(),
            "text": TextEvaluator(),
            "symbolic": SymbolicEvaluator(),
            "semantic": SemanticEvaluator()
        }
    
    def evaluate(self, text: str, constraints: List[Any]) -> Tuple[str, List[AuditStep]]:
        """
        Evaluate constraints on text and return modified text with audit steps
        
        Args:
            text: Input text to process
            constraints: List of constraints to apply
            
        Returns:
            Tuple of (processed_text, audit_steps)
        """
        audit_steps = []
        processed_text = text
        
        for constraint in constraints:
            # Validate constraint
            try:
                constraint_schema = ConstraintSchema(**constraint.__dict__)
            except Exception as e:
                logger.error("Invalid constraint: %s", str(e))
                continue
                
            constraint_type = constraint_schema.type
            
            evaluator = self.evaluators.get(constraint_type)
            if not evaluator:
                logger.warning("No evaluator for constraint type: %s", constraint_type)
                continue
                
            # Evaluate constraint
            start_time = time.time()
            processed_text, decision_log = evaluator.evaluate(processed_text, constraint_schema.params)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Create audit step
            step = AuditStep(
                constraint_id=constraint_schema.input_hash,
                constraint_name=constraint_schema.name,
                method=f"{constraint_type}_evaluator",
                pre_text=text,
                post_text=processed_text,
                elapsed_ms=elapsed_ms,
                match_context=decision_log.get("context"),
                action_applied=decision_log.get("action"),
                confidence_score=decision_log.get("score"),
                input_hash=constraint_schema.input_hash
            )
            
            # Publish to Kafka if enabled
            kafka_sink = get_kafka_sink()
            if kafka_sink:
                kafka_sink.publish(step.__dict__)
            
            audit_steps.append(step)
            
        return processed_text, audit_steps


class Fi2Agent(BaseAgent):
    """Agent for fi2 constraint execution"""
    # Can override evaluators or add custom ones


class GemmaAgent(BaseAgent):
    """Agent for gemma constraint execution"""
    # Can override evaluators or add custom ones
