import graph_tool.all as gt

from multilens import MultiLens

# Data from http://www.sociopatterns.org/datasets/primary-school-cumulative-networks/
g1 = gt.load_graph('./data/school_day2.xml.gz')

# DeepGL setting
multilens = MultiLens(ego_dist=3, nbr_types=['all'])

# network representation learing with DeepGL
Y1 = multilens.fit_transform(g1)

# results
print(Y1)
print(Y1.shape)
for nth_layer_feat_def in deepgl.feat_defs:
    print(nth_layer_feat_def)

# transfer/inductive learning example
g2 = gt.load_graph('./data/school_day1.xml.gz')
Y2 = deepgl.transform(g2)

# results
print(Y2)
print(Y2.shape)
