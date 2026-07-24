#%%
"""
    The factor have embedding sort index due to the reference,
    thus there must be a sorted factor list we need.
"""
import yaml 
import os
from collections import defaultdict, deque


CONFIG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
factor_info_path = os.path.join(CONFIG_DIR, 'factor_info.yaml')
default_info_path = os.path.join(CONFIG_DIR, 'default.yaml')

def split_factor_layers(factor_list):
    # factor_list: [[name, [deps...]], ...]
    deps_map = {}
    graph = defaultdict(list)
    indegree = defaultdict(int)

    factors = set()

    for name, deps in factor_list:
        factors.add(name)
        deps_map[name] = deps
        indegree[name] = indegree.get(name, 0)

    for name, deps in factor_list:
        for dep in deps:
            graph[dep].append(name)
            indegree[name] += 1

    layers = []
    current = [f for f in factors if indegree[f] == 0]

    while current:
        layers.append(sorted(current))
        next_current = []

        for node in current:
            for nxt in graph[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    next_current.append(nxt)

        current = next_current

    return layers

def get_feature_list():
    with open(factor_info_path, 'r') as file:
        feature_factor_info = yaml.safe_load(file)
    feature_list = feature_factor_info['feature_list'] + feature_factor_info['additional_feature_list']
    return feature_list

def get_additional_feature_list():
    with open(factor_info_path, 'r') as file:
        feature_factor_info = yaml.safe_load(file)
    additional_feature_list = feature_factor_info['additional_feature_list']
    return additional_feature_list

def get_factor_list():
    with open(factor_info_path, 'r') as file:
        feature_factor_info = yaml.safe_load(file)
    factor_list = feature_factor_info['factor_list']
    factor_list = [factor[0] for factor in factor_list]
    return factor_list

def get_sign_layer_factor_list():
    with open(factor_info_path, 'r') as file:
        feature_factor_info = yaml.safe_load(file)
    factor_list = feature_factor_info['factor_list']
    sign_layer_factor_list = split_factor_layers(factor_list)
    return sign_layer_factor_list

def get_default_params():
    with open(default_info_path, 'r') as file:
        default_params = yaml.safe_load(file)  
    return default_params 


# %%
feature_list = get_feature_list()
factor_list = get_factor_list()
additional_feature_list = get_additional_feature_list()
sign_layer_factor_list = get_sign_layer_factor_list()
default_params = get_default_params()
# %%
