# Import these to you notebook of choice
import networkx as nx
import pandas as pd


def _get_metric(metric_name, network):
    import networkx as nx
    metric_dict = {
        "degree": nx.degree_centrality(network),
        "eigen": nx.eigenvector_centrality_numpy(network),
        "close": nx.closeness_centrality(network),
        "between": nx.betweenness_centrality(network)
    }
    return metric_dict[metric_name]

def make_centrality(network, metric_name):
    metric = _get_metric(metric_name, network)
    _sort=sorted(metric.items(),
          key=lambda x:x[1], 
          reverse=True)
    metric_df = pd.DataFrame(_sort, columns=['node',
                                             f"{metric_name}_central"])
    return metric_df
def make_merge(df_list):
    from functools import reduce
    df_final = reduce(
        lambda left,right: pd.merge(left,right,on='node',
                                    how="inner"),
        df_list
    )
    return df_final