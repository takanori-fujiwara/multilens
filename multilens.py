import graph_tool.all as gt
import numpy as np
from scipy.linalg import pinv
from sklearn import decomposition

from multilens_utils import NeighborOp, RelFeatOp

# TODO: add link type in nbr_type (IMPORTANT! This needs to be done)
# TODO: use more matrix calc (right now following the psuedocode in the deepgl paper)
# TODO: use more sparse matrix to save memory usage
# TODO: do sanity check for input of base_feats


class MultiLens():
    def __init__(
            self,
            base_feat_defs=['in_degree', 'out_degree', 'total_degree'],
            rel_feat_ops=[
                'minimum', 'maximum', 'sum', 'mean', 'variance', 'l1_norm',
                'l2_norm'
            ],
            nbr_types=['in', 'out', 'all'],
            ego_dist=3,
            n_hist_bins=5,
            mat_fact_method=decomposition.PCA,  # decomposition.NMF
            n_components=30):
        self.base_feat_defs = base_feat_defs
        self.rel_feat_ops = rel_feat_ops
        self.nbr_types = nbr_types
        self.ego_dist = ego_dist
        self.n_hist_bins = n_hist_bins
        self.factorizer = mat_fact_method()
        self.factorizer.n_components = n_components

        self.feat_defs = None
        self.S = None

    def fit(self, g):
        '''
        Apply fit (i.e., process learning procedures).

        Parameters
        ----------
        g: graph-tool graph object
            A graph to be extracted features.
        Return
        ----------
        self
        '''
        X = self._prepare_base_feats(g)
        X = self._search_rel_func_space(g=g, X=X)
        H = self._gen_hist_context_matrix(g=g,
                                          X=X,
                                          nbr_types=self.nbr_types,
                                          n_bins=self.n_hist_bins)

        self.factorizer.fit(H)
        self.S = self.factorizer.components_

        return self

    def fit_transform(self, g):
        '''
        Apply fit (i.e., process learning procedures) and then return self.X.

        Parameters
        ----------
        g: graph-tool graph object
            A graph to be extracted features.
        Return
        ----------
        self.X: array_like, shape(n_nodes, n_features)
            Learned feature matrix after applying fit().
        '''
        X = self._prepare_base_feats(g)
        X = self._search_rel_func_space(g=g, X=X)
        H = self._gen_hist_context_matrix(g=g,
                                          X=X,
                                          nbr_types=self.nbr_types,
                                          n_bins=self.n_hist_bins)

        Y = self.factorizer.fit_transform(H)
        self.S = self.factorizer.components_

        return Y

    def transform(self, g, feat_defs=None, nbr_types=None, n_hist_bins=None):
        '''
        Apply transform based on the learned feature definitions (i.e.,
        applying transfer learning to a different input graph).

        Parameters
        ----------
        g: graph-tool graph object
            A graph to be extracted features.
        feat_defs: list of strings, optional, (default=None)
            Feature definitions to be used for producing features. If None,
            self.feat_defs learned by fitting is used.
        Return
        ----------
        X: array_like, shape(n_nodes, n_features)
            Feature matrix.
        '''
        if feat_defs is None:
            feat_defs = self.get_feat_defs(flatten=True)

        feat_defs_computed = []
        for feat_def in feat_defs:
            feat_comps = self._feat_def_to_feat_comps(feat_def)

            tmp_feat_def = ''
            for feat_comp in reversed(feat_comps):
                prev_tmp_feat_def = tmp_feat_def
                feat_op = feat_comp[0]
                nbr_type = None
                if len(feat_comp) == 2:
                    nbr_type = feat_comp[1]

                # base feature
                if nbr_type is None:
                    tmp_feat_def = feat_op
                    if not tmp_feat_def in feat_defs_computed:
                        self._comp_base_feat(g, feat_op)
                # rel operators
                else:
                    tmp_feat_def = self._gen_feat_def(feat_op, nbr_type,
                                                      prev_tmp_feat_def)
                    if not tmp_feat_def in feat_defs_computed:
                        self._comp_rel_op_feat(g, feat_op, nbr_type,
                                               prev_tmp_feat_def)

                feat_defs_computed.append(tmp_feat_def)

        X = np.zeros((g.num_vertices(), len(feat_defs)))
        for i, feat_def in enumerate(feat_defs):
            X[:, i] = g.vertex_properties[feat_def].a

        if nbr_types is None:
            nbr_types = self.nbr_types
        if n_hist_bins is None:
            n_hist_bins = self.n_hist_bins

        H = self._gen_hist_context_matrix(g=g,
                                          X=X,
                                          nbr_types=nbr_types,
                                          n_bins=n_hist_bins)

        Y = H @ pinv(self.S)

        return Y

    def get_feat_defs(self, flatten=True):
        '''
        Return (flattened) feature definitions.

        Parameters
        ----------
        flatten: boolean, optional, (default=True)
            If True, flattened feature definitions are returned. Otherwise,
            feture definitions learned by fitting are returned as they are.
        Return
        ----------
        feat_defs: list of (lists of) string
            Feature definitions.
        '''
        result = []
        if flatten:
            for ith_feat_defs in self.feat_defs:
                result += ith_feat_defs
        else:
            result = self.feat_defs
        return result

    def _comp_base_feat(self, g, base_feat_def):
        '''
        Compute and store base feature by using a method provided by graph-tool
        '''
        gt_measures = [
            # from graph-tool's centrarity
            'pagerank',
            'betweenness',
            'closeness',
            'eigenvector',
            'katz',
            'hits',
            # from graph-tool's topology
            'kcore_decomposition',
            'sequential_vertex_coloring',
            'max_independent_vertex_set',
            'label_components',
            'label_out_component',
            'label_largest_component'
        ]

        # judge whether weighted feature or not
        eweight = None
        b_feat = base_feat_def
        if len(base_feat_def) > 2 and base_feat_def[:2] == 'w_':
            eweight = g.edge_properties['weight']
            b_feat = base_feat_def[2:]

        if b_feat == 'in_degree' or b_feat == 'out_degree':
            g.vertex_properties[base_feat_def] = g.new_vertex_property(
                'double')
            for v in g.vertices():
                g.vertex_properties[base_feat_def][v] = eval("v." + b_feat)(
                    weight=eweight)
        elif b_feat == 'total_degree':
            g.vertex_properties[base_feat_def] = g.new_vertex_property(
                'double')
            for v in g.vertices():
                g.vertex_properties[base_feat_def][v] = v.in_degree(
                    weight=eweight) + v.out_degree(weight=eweight)
        elif b_feat in gt_measures and eweight is None:
            vals = eval('gt.' + b_feat)(g)
            if b_feat == "betweenness":
                vals = vals[0]
            elif b_feat == "eigenvector":
                vals = vals[1]
            elif b_feat == "hits":
                vals = vals[1]  # authority value TODO: hub value
            elif b_feat == "label_components":
                vals = vals[0]
            g.vertex_properties[base_feat_def] = vals
        elif b_feat in gt_measures and eweight is not None:
            vals = eval('gt.' + b_feat)(g, weight=eweight)
            if b_feat == "betweenness":
                vals = vals[0]
            elif b_feat == "eigenvector":
                vals = vals[1]
            elif b_feat == "hits":
                vals = vals[1]  # authority value TODO: hub value
            g.vertex_properties[base_feat_def] = vals
        else:
            try:
                g.vertex_properties[base_feat_def]
            except:
                print('base feature, ' + base_feat_def +
                      ', is set which is not supported in graph-tool. Load ' +
                      base_feat_def +
                      'as a vertex_properties before using MultiLens')

        return self

    def _prepare_base_feats(self, g):
        '''
        Compute and store all base features
        '''
        X = np.zeros((g.num_vertices(), len(self.base_feat_defs)))
        self.feat_defs = [self.base_feat_defs]
        for i, feat_def in enumerate(self.base_feat_defs):
            self._comp_base_feat(g, feat_def)
            X[:, i] = g.vertex_properties[feat_def].a

        return X

    def _gen_feat_def(self, rel_op, nbr_type, prev_feat_def):
        '''
        Generate feature definition string from inputs
        '''
        return rel_op + '^' + nbr_type + '-' + prev_feat_def

    def _comp_rel_op_feat(self, g, rel_op, nbr_type, prev_feat_def):
        '''
        Compute and store new features by applying relational operators
        '''
        new_feat_def = self._gen_feat_def(rel_op, nbr_type, prev_feat_def)

        g.vertex_properties[new_feat_def] = g.new_vertex_property('double')
        x = g.vertex_properties[prev_feat_def]

        for v in g.vertices():
            nbrs = eval('NeighborOp().' + nbr_type + '_nbr(g, v)')
            feat_val = eval('RelFeatOp().' + rel_op + '(nbrs, x)')
            # to avoid the result > inf (this happens when using hadamard)
            feat_val = min(feat_val, np.finfo(np.float64).max)
            g.vertex_properties[new_feat_def][v] = feat_val

        return new_feat_def

    def _search_rel_func_space(self, g, X):
        '''
        Searching the relational function space (Sec. 2.3, Rossi et al., 2018)
        '''
        n_rel_feat_ops = len(self.rel_feat_ops)
        n_nbr_types = len(self.nbr_types)

        for l in range(1, self.ego_dist):
            prev_feat_defs = self.feat_defs[l - 1]
            new_feat_defs = []

            for i, op in enumerate(self.rel_feat_ops):
                for j, nbr_type in enumerate(self.nbr_types):
                    for k, prev_feat_def in enumerate(prev_feat_defs):
                        new_feat_def = self._comp_rel_op_feat(
                            g, op, nbr_type, prev_feat_def)

                        new_feat = np.expand_dims(
                            g.vertex_properties[new_feat_def].a, axis=1)
                        X = np.concatenate((X, new_feat), axis=1)
                        new_feat_defs.append(new_feat_def)

            self.feat_defs.append(new_feat_defs)

        return X

    def _feat_def_to_feat_comps(self, feat_def):
        ''' From feature definition string, generate a list of pairs of
        a relational feature operator and a neigbor type.
        e.g., 'mean^in-mean^in-in_degree' => [['mean', 'in'], ['mean', 'in'], ['in_degree']]
        '''
        feat_comps = feat_def.split('-')
        for i, feat_comp in enumerate(feat_comps):
            feat_comps[i] = feat_comp.split('^')
        return feat_comps

    def _comp_rel_op_feat(self, g, rel_op, nbr_type, prev_feat_def):
        '''
        Compute and store new features by applying relational operators
        '''
        new_feat_def = self._gen_feat_def(rel_op, nbr_type, prev_feat_def)

        g.vertex_properties[new_feat_def] = g.new_vertex_property('double')
        x = g.vertex_properties[prev_feat_def]

        for v in g.vertices():
            nbrs = eval('NeighborOp().' + nbr_type + '_nbr(g, v)')
            feat_val = eval('RelFeatOp().' + rel_op + '(nbrs, x)')
            # to avoid the result > inf (this happens when using hadamard)
            feat_val = min(feat_val, np.finfo(np.float64).max)
            g.vertex_properties[new_feat_def][v] = feat_val

        return new_feat_def

    def _gen_hist_context_matrix(self, g, X, nbr_types, n_bins=5):
        if n_bins <= 0:
            print('n_bins must be greater than 0')
            n_bins = 1

        N, D = X.shape

        # TODO: maybe we should prepare differnt log binning methods
        # (e.g., based on fixed bases, etc)
        # log binning
        min_vals = X.min(axis=0)
        zero_min_dims = (min_vals == 0)
        # To handle, 0 values
        min_vals[min_vals == 0] = np.finfo(float).eps
        # this is to handle the case dtype is int
        min_vals[min_vals == 0] = 1

        max_vals = X.max(axis=0)
        zero_max_dims = (max_vals == 0)
        max_vals[max_vals == 0] = np.finfo(float).eps
        max_vals[max_vals == 0] = 1

        ranges = max_vals - min_vals
        bases = ranges**(1 / n_bins)
        bases[bases == 0] = np.finfo(float).eps
        bases[bases == 0] = 1

        start_exps = np.log(min_vals) / np.log(bases)
        stop_exps = np.log(max_vals) / np.log(bases)
        bins_for_all_dims = np.zeros((D, n_bins + 1))
        for d in range(D):
            bins = np.logspace(start_exps[d],
                               stop_exps[d],
                               num=n_bins + 1,
                               base=bases[d])
            if zero_min_dims[d]:
                bins[0] = 0

            bins_for_all_dims[d, :] = bins

        H = np.zeros((N, D * n_bins))
        for nbr_type in nbr_types:
            for v in g.vertices():
                nbrs = eval('NeighborOp().' + nbr_type + '_nbr(g, v)')
                for d in range(D):
                    nbr_feat_vals = X[nbrs, d]
                    hist, _ = np.histogram(nbr_feat_vals,
                                           bins=bins_for_all_dims[d])
                    H[int(v), d * n_bins:(d + 1) * n_bins] = hist

        return H
