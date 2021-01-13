import graph_tool.all as gt
import numpy as np

from multilens import MultiLens

# Data from http://www.sociopatterns.org/datasets/primary-school-cumulative-networks/
g1 = gt.load_graph('./data/school_day2.xml.gz')

# DeepGL setting
multilens = MultiLens(ego_dist=3,
                      base_feat_defs=['total_degree'],
                      nbr_types=['all'])

# network representation learing with DeepGL
Y1 = multilens.fit_transform(g1)

# results
print(Y1)
print(Y1.shape)
for nth_layer_feat_def in multilens.feat_defs:
    print(nth_layer_feat_def)

# transfer/inductive learning example
g2 = gt.load_graph('./data/school_day1.xml.gz')
Y2 = multilens.transform(g2)

# results
print(Y2)
print(Y2.shape)

# example with edge proprty
g1 = gt.load_graph('./data/school_day2.xml.gz')
efilt = np.random.choice([True, False], size=g1.num_edges())

# DeepGL setting
multilens = MultiLens(ego_dist=3,
                      base_feat_defs=['total_degree', 'total_degree@sample'],
                      nbr_types=['all@sample'],
                      efilts={'sample': efilt})

# network representation learing with DeepGL
Y1 = multilens.fit_transform(g1)

# results
print(Y1)
print(Y1.shape)
for nth_layer_feat_def in multilens.feat_defs:
    print(nth_layer_feat_def)

# transfer/inductive learning example
g2 = gt.load_graph('./data/school_day1.xml.gz')
efilt2 = np.random.choice([True, False], size=g2.num_edges())
Y2 = multilens.transform(g2, efilts={'sample': efilt2})

# results
print(Y2)
print(Y2.shape)
