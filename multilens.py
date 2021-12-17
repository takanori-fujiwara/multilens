import math

import graph_tool.all as gt
import numpy as np
from scipy.linalg import pinv
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler

from multilens_utils import NeighborOp, RelFeatOp

# TODO: use more matrix calc (right now following the psuedocode in the deepgl paper)
# TODO: use more sparse matrix to save memory usage
# TODO: do sanity check for input of base_feats


class MultiLens():
    '''
    Multi-LENS from Jin et al., 2019 (https://dl.acm.org/doi/abs/10.1145/3292500.3330992).

    Parameters
    ----------
    base_feat_defs: list of strings, optional, (default=['in_degree', 'out_degree', 'total_degree'])
        [Basic setting]
        Base features considered for network representation learning. Current
        implementation supports base features available in graph-tool:
        'in_degree', 'out_degree', 'total_degree', 'pagerank', 'betweenness',
        'closeness', 'eigenvector', 'katz', 'hits', 'kcore_decomposition',
        'sequential_vertex_coloring', 'max_independent_vertex_set',
        'label_components', 'label_out_component', 'label_largest_component'.
        [Weighted edges]
        Also, a weighted version of base feature is available by adding "w_" in
        front of a base feature name. For exaple, weighted in-degree can be set
        with "w_in_degree".
        [Node attributes]
        Node attributes can be included as well. To include node attributes,
        indicate vertex property names used in input graph-tool graph objects.
        For example, when graph objects have "gender" vertex property, you can
        include it as base_feat_defs with "gender" (see sample.py).
        [Edge filters]
        Edge filters can be added for base feature computation as well. Set
        efilts when using fit(), fit_transform(), or transform() (see fit())
        and indicate related edge filter with "@[efilt_key]".
        For example, when efilts={'filt1': xxx, 'filt2', xxx}, you can use
        "total_degree@filter1" (see sample.py).
    rel_feat_ops: list of strings, optional, (default=['minimum', 'maximum', 'sum', 'mean', 'variance', 'l1_norm', 'l2_norm'])
        Relational feature operators cosidered for learning. Current
        implmentation supports: 'mean', 'sum', 'maximum', 'hadamard', 'lp_norm',
        'rbf'. However, 'hadamard', 'lp_norm', 'rbf' are unstable to use.
    nbr_types: list of strings, optional, (default=['in', 'out', 'all'])
        Neighborhood types cosidered for learning. 'in', 'out', 'all' can be set.
    ego_dist: int, optional, (default=3)
        The maximum distance/# of hops to be used when computing egonet features.
    use_nbr_for_hist: bool, optional, (default=True)
        If True, when generating a histogram-based context matrix, for each
        node, generate histograms with (n_hist_bins) bins from its neighbor
        nodes' feature values (i.e., each cell of the histogram-based context
        matrix will have (n_hist_bins) values). See Sec. 3.3 in MultiLens by
        Jin et al., 2019.
        If False, generate a histogram-based context matrix by scaling each
        node's feature with log binning using log_binning_alpha (i.e., each
        cell of the histogram-based context matrix will have 1 value). See
        Sec.2.3 of DeepGL by Rossi et al., 2018.
    n_hist_bins: int, optional, (default=5)
        (This is used when use_nbr_for_hist=True) The number of histogram bins
        used when obtaining a histogram-based context matrix.
    log_binning_alpha: float, optional, (default=0.5)
        (This is used when use_nbr_for_hist=False) Ratio of bin width to the
        base bin width used for each bin. For example, if values are from 0 to
        100 and log_binning_alpha = 0.8, the first bin is from 0-80, the next
        bin is 80-96, and so on. log_binning_alpha must be
        0.0 < log_binning_alpha < 1.0.
    mat_fact_method: sklearn decomposition method, optional, (default=decomposition.PCA)
        Matrix factorization method used when computing summary representations.
        In default, SVD (PCA) is used. Other methods, such as decomposition.NMF,
        can be used as well.
    n_components: int, optional, (default=30)
        The number of components to be kept in summary representations.
    Attributes
    ----------
    S: summary representation, shape(n_features, n_components)
        Summary representation learned by fit().
    feat_defs: list of lists of string
        Learned features' definitions after applying fit(). i-th list of
        strings correspond to i-egonet features' definitions.
    base_feat_defs: list of strings
        Access to the parameter.
    rel_feat_ops: list of strings
        Access to the parameter.
    nbr_types: list of strings
        Access to the parameter.
    ego_dist: int
        Access to the parameter.
    n_hist_bins: int
        Access to the parameter.
    mat_fact_method: sklearn decomposition method
        Access to the parameter.
    n_components: int
        Access to the parameter.
    Examples
    --------
    '''
    def __init__(
            self,
            base_feat_defs=['in_degree', 'out_degree', 'total_degree'],
            rel_feat_ops=[
                'minimum', 'maximum', 'sum', 'mean', 'variance', 'l1_norm',
                'l2_norm'
            ],
            nbr_types=['in', 'out', 'all'],
            ego_dist=3,
            use_nbr_for_hist=True,
            n_hist_bins=5,
            log_binning_alpha=0.5,
            mat_fact_method=decomposition.PCA,  # decomposition.NMF
            n_components=30):
        self.base_feat_defs = base_feat_defs
        self.rel_feat_ops = rel_feat_ops
        self.nbr_types = nbr_types
        self.ego_dist = ego_dist
        self.use_nbr_for_hist = use_nbr_for_hist
        self.n_hist_bins = n_hist_bins
        self.log_binning_alpha = log_binning_alpha
        self.factorizer = mat_fact_method()
        self.factorizer.n_components = n_components

        self.feat_defs = None
        self.S = None

    def fit(self, g, efilts={}, return_hist=False):
        '''
        Apply fit (i.e., process learning procedures).

        Parameters
        ----------
        g: graph-tool graph object
            A graph to be extracted features.
        efilts: dictionary, optional, (default={})
            Dictionary of edge filter name and graph-tool's edge filter (e.g.,
            {'filt1': efilt1, 'filt2': efilt2}). This is used and must be
            provided when edge filters are set for self.base_feat_defs or
            self.nbr_type. Each edge filter (e.g., 'efilt1' above) must have
            the same length with g.num_edges(). As for graph-tool's edge filter,
            refer to class graph_tool.GraphView in https://graph-tool.skewed.de/static/doc/graph_tool.html?highlight=graphview#graph_tool.GraphView.
        return_hist: boolean, optional, default=False
            If True, return histogram-based context matrix H
        Return
        ----------
        self
        '''
        X = self._prepare_base_feats(g, efilts=efilts)
        X = self._search_rel_func_space(g=g, X=X, efilts=efilts)
        H = self._gen_hist_context_matrix(
            g=g,
            X=X,
            use_nbr_for_hist=self.use_nbr_for_hist,
            n_bins=self.n_hist_bins,
            nbr_types=self.nbr_types,
            efilts=efilts,
            log_binning_alpha=self.log_binning_alpha)

        if self.factorizer.n_components > H.shape[1]:
            print('n_components < # of cols of hist context matrix')
            print(f'n_components = {H.shape[1]} is used')
            self.factorizer.n_components = H.shape[1]

        self.factorizer.fit(H)
        self.S = self.factorizer.components_

        if return_hist:
            return self, H

        return self

    def fit_transform(self, g, efilts={}, return_hist=False):
        '''
        Apply fit (i.e., process learning procedures) and then return self.X.

        Parameters
        ----------
        g: graph-tool graph object
            A graph to be extracted features.
        efilts: dictionary, optional, (default={})
            Dictionary of edge filter name and graph-tool's edge filter (e.g.,
            {'filt1': efilt1, 'filt2': efilt2}). This is used and must be
            provided when edge filters are set for self.base_feat_defs or
            self.nbr_type. Each edge filter (e.g., 'efilt1' above) must have
            the same length with g.num_edges(). As for graph-tool's edge filter,
            refer to class graph_tool.GraphView in https://graph-tool.skewed.de/static/doc/graph_tool.html?highlight=graphview#graph_tool.GraphView.
        return_hist: boolean, optional, default=False
            If True, return histogram-based context matrix H
        Return
        ----------
        Y: array_like, shape(n_nodes, n_components)
            Node embedding after applying fit().
        '''
        X = self._prepare_base_feats(g, efilts=efilts)
        X = self._search_rel_func_space(g=g, X=X, efilts=efilts)
        H = self._gen_hist_context_matrix(
            g=g,
            X=X,
            use_nbr_for_hist=self.use_nbr_for_hist,
            n_bins=self.n_hist_bins,
            nbr_types=self.nbr_types,
            efilts=efilts,
            log_binning_alpha=self.log_binning_alpha)

        if self.factorizer.n_components > H.shape[1]:
            print('n_components < # of cols of hist context matrix')
            print(f'n_components = {H.shape[1]} is used')
            self.factorizer.n_components = H.shape[1]

        Y = self.factorizer.fit_transform(H)
        self.S = self.factorizer.components_

        if return_hist:
            return Y, H

        return Y

    def transform(self,
                  g,
                  feat_defs=None,
                  nbr_types=None,
                  efilts={},
                  use_nbr_for_hist=None,
                  n_hist_bins=None,
                  log_binning_alpha=None,
                  return_hist=False):
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
        nbr_types: list of strings, optional, (default=None)
            Neighbor types to be used for producing features. If None,
            self.nbr_types is used.
        efilts: dictionary, optional, (default={})
            Dictionary of edge filter name and graph-tool's edge filter (e.g.,
            {'filt1': efilt1, 'filt2': efilt2}). This is used and must be
            provided when edge filters are set for self.base_feat_defs or
            self.nbr_type. Each edge filter (e.g., 'efilt1' above) must have
            the same length with g.num_edges(). As for graph-tool's edge filter,
            refer to class graph_tool.GraphView in https://graph-tool.skewed.de/static/doc/graph_tool.html?highlight=graphview#graph_tool.GraphView.
        use_nbr_for_hist: bool, optional, (default=True)
            If True, when generatign a histogram-based context matrix, for each
            node, generate histograms with (n_hist_bins) bins from its neighbor
            nodes' feature values (i.e., each cell of the histogram-based context
            matrix will have (n_hist_bins) values). See Sec. 3.3 in MultiLens by
            Jin et al., 2019.
            If False, generate a histogram-based context matrix by scaling each
            node's feature with log binning using log_binning_alpha (i.e., each
            cell of the histogram-based context matrix will have 1 value). See
            Sec.2.3 of DeepGL by Rossi et al., 2018.
            If None, self.use_nbr_for_hist is used.
        n_hist_bins: int, optional, (default=None)
            The number of histogram bins to be used for producing features. If
            None, self.n_hist_bins is used.
        log_binning_alpha: float, optional, (default=None)
            (This is used when use_nbr_for_hist=False) Ratio of bin width to the
            base bin width used for each bin. For example, if values are from 0 to
            100 and log_binning_alpha = 0.8, the first bin is from 0-80, the next
            bin is 80-96, and so on. log_binning_alpha must be
            0.0 < log_binning_alpha < 1.0.
            If None, self.log_binning_alpha is used.
        Return
        ----------
        Y: array_like, shape(n_nodes, n_components)
            Node embedding obtained with transfer learning using self.S.
        '''
        if nbr_types is None:
            nbr_types = self.nbr_types
        if use_nbr_for_hist is None:
            use_nbr_for_hist = self.use_nbr_for_hist
        if n_hist_bins is None:
            n_hist_bins = self.n_hist_bins
        if log_binning_alpha is None:
            log_binning_alpha = self.log_binning_alpha

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
                        self._comp_base_feat(g, feat_op, efilts)
                # rel operators
                else:
                    tmp_feat_def = self._gen_feat_def(feat_op, nbr_type,
                                                      prev_tmp_feat_def)
                    if not tmp_feat_def in feat_defs_computed:
                        self._comp_rel_op_feat(g, feat_op, nbr_type,
                                               prev_tmp_feat_def, efilts)

                feat_defs_computed.append(tmp_feat_def)

        X = np.zeros((g.num_vertices(), len(feat_defs)))
        for i, feat_def in enumerate(feat_defs):
            X[:, i] = g.vertex_properties[feat_def].a

        H = self._gen_hist_context_matrix(g=g,
                                          X=X,
                                          use_nbr_for_hist=use_nbr_for_hist,
                                          n_bins=n_hist_bins,
                                          nbr_types=nbr_types,
                                          efilts=efilts,
                                          log_binning_alpha=log_binning_alpha)

        Y = H @ pinv(self.S)

        if return_hist:
            return Y, H
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

    def _comp_base_feat(self, g, base_feat_def, efilts):
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

        # to handle the case with edge filters
        gv = g
        if len(base_feat_def.split('@')) > 1:
            b_feat, efilt_key = base_feat_def.split('@')
            gv = gt.GraphView(g, efilt=efilts[efilt_key])

        if b_feat == 'in_degree' or b_feat == 'out_degree':
            g.vertex_properties[base_feat_def] = g.new_vertex_property(
                'double')
            for v in gv.vertices():
                g.vertex_properties[base_feat_def][v] = eval("v." + b_feat)(
                    weight=eweight)
        elif b_feat == 'total_degree':
            g.vertex_properties[base_feat_def] = g.new_vertex_property(
                'double')
            for v in gv.vertices():
                g.vertex_properties[base_feat_def][v] = v.in_degree(
                    weight=eweight) + v.out_degree(weight=eweight)
        elif b_feat in gt_measures and eweight is None:
            vals = eval('gt.' + b_feat)(gv)
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
            vals = eval('gt.' + b_feat)(gv, weight=eweight)
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
                print(f'graph-tool do not support base feat {base_feat_def}.')
                print(f'Set {base_feat_def} as v_prop before using fit')

        return self

    def _prepare_base_feats(self, g, efilts):
        '''
        Compute and store all base features
        '''
        X = np.zeros((g.num_vertices(), len(self.base_feat_defs)))
        self.feat_defs = [self.base_feat_defs]
        for i, feat_def in enumerate(self.base_feat_defs):
            self._comp_base_feat(g, feat_def, efilts)
            X[:, i] = g.vertex_properties[feat_def].a

        return X

    def _gen_feat_def(self, rel_op, nbr_type, prev_feat_def):
        '''
        Generate feature definition string from inputs
        '''
        return f'{rel_op}^{nbr_type}-{prev_feat_def}'

    def _search_rel_func_space(self, g, X, efilts):
        '''
        Searching the relational function space (Sec. 2.3, Rossi et al., 2018)
        '''
        for l in range(1, self.ego_dist):
            prev_feat_defs = self.feat_defs[l - 1]
            new_feat_defs = []

            for op in self.rel_feat_ops:
                for nbr_type in self.nbr_types:
                    for prev_feat_def in prev_feat_defs:
                        new_feat_def = self._comp_rel_op_feat(
                            g, op, nbr_type, prev_feat_def, efilts)

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

    def _get_nbr_direction_and_graphview(self, g, nbr_type, efilts):
        nbr_direction = nbr_type
        gv = g
        if len(nbr_type.split('@')) > 1:
            nbr_direction, nbr_efilt_key = nbr_type.split('@')
            gv = gt.GraphView(g, efilt=efilts[nbr_efilt_key])

        return nbr_direction, gv

    def _comp_rel_op_feat(self, g, rel_op, nbr_type, prev_feat_def, efilts):
        '''
        Compute and store new features by applying relational operators
        '''
        new_feat_def = self._gen_feat_def(rel_op, nbr_type, prev_feat_def)

        g.vertex_properties[new_feat_def] = g.new_vertex_property('double')
        x = g.vertex_properties[prev_feat_def]

        # TODO: by preparing graph view in advance, probably we can speed up
        # the process
        nbr_direction, gv = self._get_nbr_direction_and_graphview(
            g, nbr_type, efilts)

        for v in gv.vertices():
            nbrs = eval('NeighborOp().' + nbr_direction + '_nbr(gv, v)')
            feat_val = eval('RelFeatOp().' + rel_op + '(nbrs, x)')
            # to avoid the result > inf (this happens when using hadamard)
            feat_val = min(feat_val, np.finfo(np.float64).max)
            g.vertex_properties[new_feat_def][v] = feat_val

        return new_feat_def

    def _log_binning(self, X, alpha=0.5, copy=False):
        # note: this method overwrites X
        if alpha > 1.0 or alpha < 0.0:
            print('alpha must between 0.0 and 1.0')

        n, d = X.shape
        X_argsort = np.argsort(X, axis=0)

        bin_start = 0
        bin_width = math.ceil(alpha * n)
        bin_val = 0

        while bin_start <= n:
            bin_end = bin_start + bin_width

            for i in range(d):
                X[X_argsort[bin_start:bin_end, i], i] = bin_val

            bin_start = bin_end
            bin_width = math.ceil(alpha * bin_width)
            bin_val += 1

        return X

    def _gen_hist_context_matrix(self, g, X, use_nbr_for_hist, n_bins,
                                 nbr_types, efilts, log_binning_alpha):
        if n_bins <= 0:
            print('n_bins must be greater than 0')
            n_bins = 1

        N, D = X.shape

        # TODO: maybe we should prepare differnt log binning methods
        # (e.g., based on fixed bases, etc or no log binning (e.g., gender))
        # log binning
        # prepare bins in a range of [1, 2**n_bins]
        bins = np.logspace(0, n_bins, num=n_bins + 1, base=2)

        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1

        H = None
        if use_nbr_for_hist:
            H = np.zeros((N, D * n_bins))
            for nbr_type in nbr_types:
                # TODO: by preparing graph view in advance, probably we can speed up
                # the process
                nbr_direction, gv = self._get_nbr_direction_and_graphview(
                    g, nbr_type, efilts)
                for v in gv.vertices():
                    nbrs = eval('NeighborOp().' + nbr_direction +
                                '_nbr(gv, v)')
                    for d in range(D):
                        nbr_feat_vals = X[nbrs, d]
                        # scaling to [1, 2**n_bins]
                        nbr_feat_vals = ((nbr_feat_vals - min_vals[d]) /
                                         ranges[d]) * (2**n_bins - 1) + 1
                        hist, _ = np.histogram(nbr_feat_vals, bins=bins)
                        H[int(v), d * n_bins:(d + 1) * n_bins] = hist
        else:
            H = self._log_binning(X, alpha=log_binning_alpha)
        # else:
        #     H = np.zeros((N, D))
        #     for v in g.vertices():
        #         for d in range(D):
        #             feat_val = X[int(v), d]
        #             # scaling to [1, 2**n_bins]
        #             feat_val = ((feat_val - min_vals[d]) /
        #                         ranges[d]) * (2**n_bins - 1) + 1
        #             H[int(v), d] = np.log2(feat_val)

        return H
