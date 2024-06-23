from dataclasses import dataclass
from os.path import dirname, realpath
from typing import Any
import json

@dataclass
class Explanation:
    lime: bool
    lime_plot: bool
    shap: bool
    shap_bar_plot: bool
    shap_cluster: bool
    shap_scatter: bool

    @staticmethod
    def from_dict(obj: Any) -> 'Explanation':
        _lime = bool(obj.get("lime")) if obj.get("lime") is not None else True
        _lime_plot = bool(obj.get("lime_plot")) if obj.get("lime_plot") is not None else False
        _shap = bool(obj.get("shap")) if obj.get("shap") is not None else True
        _shap_bar_plot = bool(obj.get("shap_bar_plot")) if obj.get("shap_bar_plot") is not None else False
        _shap_cluster = bool(obj.get("shap_cluster")) if obj.get("shap_cluster") is not None else False
        _shap_scatter = bool(obj.get("shap_scatter")) if obj.get("shap_scatter") is not None else False
        
        return Explanation(
                            lime=_lime, 
                            lime_plot=_lime_plot,
                            shap=_shap,
                            shap_bar_plot=_shap_bar_plot,
                            shap_cluster=_shap_cluster,
                            shap_scatter=_shap_scatter,
                        )

@dataclass 
class Config:
    explanation: Explanation
    sampling: str
    selection: str
    classification: str

    @staticmethod
    def from_dict(obj: Any) -> 'Config':
        _explanation = Explanation.from_dict(obj.get("explanation"))
        _sampling = str(obj.get("sampling")) if obj.get("sampling") is not None else 'none'
        _selection = str(obj.get("selection")) if obj.get("selection") is not None else 'none'
        _classification = str(obj.get("classification")) if obj.get("classification") is not None else 'rfc'
    
        return Config(
                        explanation=_explanation,
                        sampling=_sampling,
                        selection=_selection,
                        classification=_classification
                    ) 
        
        
def load_config():
    root_dir = dirname(realpath(__file__))
    config_file_name = root_dir + "/" + "config.json"
    with open(config_file_name,'r') as file:
        configString = file.read()
        
    configJSON = json.loads(configString)
    config = Config.from_dict(configJSON)
    return config