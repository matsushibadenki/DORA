# snn_research/learning_rules/__init__.py

from typing import Dict, Any
from .base_rule import PlasticityRule
from .stdp import STDPRule
from .forward_forward import ForwardForwardRule

# エイリアス
BioLearningRule = PlasticityRule

def get_bio_learning_rule(rule_name: str = "stdp", config: Dict[str, Any] = {}, **kwargs: Any) -> PlasticityRule:
    """
    Legacy factory function for backward compatibility.
    """
    if 'name' in kwargs:
        rule_name = kwargs['name']
        
    rule_name = rule_name.lower()
    
    if "stdp" in rule_name or "causal" in rule_name:
        params = config.get("params", {})
        # 古いSTDPパラメータとの整合性を保つための変換ロジックが必要ならここに追加
        return STDPRule(**params)
        
    elif "forward" in rule_name:
        params = config.get("params", {})
        return ForwardForwardRule(**params)
        
    else:
        # デフォルトでSTDPを返す
        return STDPRule()

__all__ = [
    "PlasticityRule",
    "STDPRule",
    "ForwardForwardRule",
    "get_bio_learning_rule", # 再公開
    "BioLearningRule",
]