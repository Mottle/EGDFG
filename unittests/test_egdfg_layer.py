import unittest

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter

from gps.layer.egdfg_layer import EnchantedViaDFGLayer, k_farthest_graph


def _k_farthest_graph_original(
    x, k, batch=None, loop=False, cosine=True, direction=True
):
    """Reference implementation for correctness verification."""
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)
    num_nodes = x.size(0)
    num_graphs = batch.max().item() + 1
    ptr = scatter(
        torch.ones(num_nodes, dtype=torch.long, device=x.device),
        batch,
        dim=0,
        dim_size=num_graphs,
        reduce="sum",
    ).cumsum(0)
    ptr = torch.cat([x.new_zeros(1, dtype=torch.long), ptr])
    all_edge_indices = []
    for i in range(num_graphs):
        start_idx = ptr[i]
        end_idx = ptr[i + 1]
        x_i = x[start_idx:end_idx]
        n_i = x_i.size(0)
        if n_i == 0:
            continue
        if cosine:
            x_norm = x_i / x_i.norm(dim=1, keepdim=True).clamp(min=1e-8)
            S = torch.mm(x_norm, x_norm.t())
            D_i = 1.0 - S
        else:
            x_sq = torch.sum(x_i**2, dim=1, keepdim=True)
            D_i = x_sq + x_sq.t() - 2 * torch.mm(x_i, x_i.t())
            D_i = D_i.clamp(min=0.0)
        k_adjusted = min(k + 1, n_i)
        _, indices = torch.topk(D_i, k=k_adjusted, dim=1, largest=True)
        source_nodes_local = torch.arange(n_i, device=x.device).repeat_interleave(
            k_adjusted
        )
        target_nodes_local = indices.flatten()
        edge_index_local = torch.stack([source_nodes_local, target_nodes_local], dim=0)
        if not loop:
            mask = edge_index_local[0] != edge_index_local[1]
            edge_index_local = edge_index_local[:, mask]
        edge_index_global = edge_index_local + start_idx
        if not direction:
            edge_index_global = torch.cat(
                [edge_index_global, edge_index_global.flip(0)], dim=1
            )
        all_edge_indices.append(edge_index_global)
    if len(all_edge_indices) == 0:
        return x.new_empty((2, 0), dtype=torch.long)
    return torch.cat(all_edge_indices, dim=1)


def _sets_equal(e1, e2):
    s1 = set(map(tuple, e1.t().tolist()))
    s2 = set(map(tuple, e2.t().tolist()))
    return s1 == s2


class TestKFarthestGraph(unittest.TestCase):

    def test_single_graph_loop_true(self):
        torch.manual_seed(42)
        x = torch.randn(10, 8)
        batch = torch.zeros(10, dtype=torch.long)
        e_orig = _k_farthest_graph_original(
            x, k=3, batch=batch, loop=True, cosine=True, direction=True
        )
        e_new = k_farthest_graph(
            x, k=3, batch=batch, loop=True, cosine=True, direction=True
        )
        self.assertTrue(_sets_equal(e_orig, e_new))

    def test_single_graph_loop_false(self):
        torch.manual_seed(42)
        x = torch.randn(10, 8)
        batch = torch.zeros(10, dtype=torch.long)
        e_orig = _k_farthest_graph_original(
            x, k=3, batch=batch, loop=False, cosine=True, direction=True
        )
        e_new = k_farthest_graph(
            x, k=3, batch=batch, loop=False, cosine=True, direction=True
        )
        self.assertTrue(_sets_equal(e_orig, e_new))

    def test_batch_varying_sizes(self):
        torch.manual_seed(42)
        x1 = torch.randn(5, 8)
        x2 = torch.randn(8, 8)
        x3 = torch.randn(3, 8)
        x = torch.cat([x1, x2, x3], dim=0)
        batch = torch.cat([torch.zeros(5), torch.ones(8), torch.full((3,), 2)]).long()
        e_orig = _k_farthest_graph_original(
            x, k=3, batch=batch, loop=True, cosine=True, direction=True
        )
        e_new = k_farthest_graph(
            x, k=3, batch=batch, loop=True, cosine=True, direction=True
        )
        self.assertTrue(_sets_equal(e_orig, e_new))

    def test_batch_loop_false(self):
        torch.manual_seed(42)
        x1 = torch.randn(5, 8)
        x2 = torch.randn(8, 8)
        x3 = torch.randn(3, 8)
        x = torch.cat([x1, x2, x3], dim=0)
        batch = torch.cat([torch.zeros(5), torch.ones(8), torch.full((3,), 2)]).long()
        e_orig = _k_farthest_graph_original(
            x, k=3, batch=batch, loop=False, cosine=True, direction=True
        )
        e_new = k_farthest_graph(
            x, k=3, batch=batch, loop=False, cosine=True, direction=True
        )
        self.assertTrue(_sets_equal(e_orig, e_new))

    def test_bidirectional(self):
        torch.manual_seed(42)
        x1 = torch.randn(5, 8)
        x2 = torch.randn(8, 8)
        x3 = torch.randn(3, 8)
        x = torch.cat([x1, x2, x3], dim=0)
        batch = torch.cat([torch.zeros(5), torch.ones(8), torch.full((3,), 2)]).long()
        e_orig = _k_farthest_graph_original(
            x, k=3, batch=batch, loop=True, cosine=True, direction=False
        )
        e_new = k_farthest_graph(
            x, k=3, batch=batch, loop=True, cosine=True, direction=False
        )
        self.assertTrue(_sets_equal(e_orig, e_new))

    def test_euclidean_distance(self):
        torch.manual_seed(42)
        x1 = torch.randn(5, 8)
        x2 = torch.randn(8, 8)
        x3 = torch.randn(3, 8)
        x = torch.cat([x1, x2, x3], dim=0)
        batch = torch.cat([torch.zeros(5), torch.ones(8), torch.full((3,), 2)]).long()
        e_orig = _k_farthest_graph_original(
            x, k=3, batch=batch, loop=True, cosine=False, direction=True
        )
        e_new = k_farthest_graph(
            x, k=3, batch=batch, loop=True, cosine=False, direction=True
        )
        self.assertTrue(_sets_equal(e_orig, e_new))

    def test_batch_none(self):
        torch.manual_seed(42)
        x = torch.randn(16, 8)
        e_orig = _k_farthest_graph_original(
            x, k=3, batch=None, loop=True, cosine=True, direction=True
        )
        e_new = k_farthest_graph(
            x, k=3, batch=None, loop=True, cosine=True, direction=True
        )
        self.assertTrue(_sets_equal(e_orig, e_new))

    def test_no_self_loops_removes_diagonal(self):
        torch.manual_seed(0)
        x = torch.randn(5, 8)
        batch = torch.zeros(5, dtype=torch.long)
        edge_index = k_farthest_graph(
            x, k=3, batch=batch, loop=False, cosine=True, direction=True
        )
        has_self_loop = (edge_index[0] == edge_index[1]).any().item()
        self.assertFalse(has_self_loop)

    def test_self_loops_not_removed_when_loop_true(self):
        torch.manual_seed(0)
        x = torch.randn(5, 8)
        batch = torch.zeros(5, dtype=torch.long)
        e_loop = k_farthest_graph(
            x, k=3, batch=batch, loop=True, cosine=True, direction=True
        )
        e_no_loop = k_farthest_graph(
            x, k=3, batch=batch, loop=False, cosine=True, direction=True
        )
        loop_self = (e_loop[0] == e_loop[1]).sum().item()
        no_loop_self = (e_no_loop[0] == e_no_loop[1]).sum().item()
        self.assertEqual(no_loop_self, 0)
        self.assertGreaterEqual(loop_self, no_loop_self)

    def test_bidirectional_edge_count(self):
        torch.manual_seed(0)
        x = torch.randn(5, 8)
        batch = torch.zeros(5, dtype=torch.long)
        e_directed = k_farthest_graph(
            x, k=3, batch=batch, loop=True, cosine=True, direction=True
        )
        e_bidir = k_farthest_graph(
            x, k=3, batch=batch, loop=True, cosine=True, direction=False
        )
        self.assertEqual(e_bidir.size(1), 2 * e_directed.size(1))

    def test_k_larger_than_nodes(self):
        torch.manual_seed(0)
        x = torch.randn(3, 8)
        batch = torch.zeros(3, dtype=torch.long)
        e_orig = _k_farthest_graph_original(
            x, k=5, batch=batch, loop=True, cosine=True, direction=True
        )
        e_new = k_farthest_graph(
            x, k=5, batch=batch, loop=True, cosine=True, direction=True
        )
        self.assertTrue(_sets_equal(e_orig, e_new))

    def test_output_dtype(self):
        torch.manual_seed(0)
        x = torch.randn(10, 8)
        batch = torch.zeros(10, dtype=torch.long)
        edge_index = k_farthest_graph(
            x, k=3, batch=batch, loop=True, cosine=True, direction=True
        )
        self.assertEqual(edge_index.dtype, torch.long)
        self.assertEqual(edge_index.dim(), 2)
        self.assertEqual(edge_index.size(0), 2)


class TestEnchantedViaDFGLayer(unittest.TestCase):

    def _make_data(self, num_nodes, dim=16):
        x = torch.randn(num_nodes, dim)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        return Data(x=x, edge_index=edge_index)

    def test_forward_output_shape(self):
        data = self._make_data(10, dim=16)
        layer = EnchantedViaDFGLayer(dim_in=16, dim_out=16, dropout=0.1, residual=False)
        out = layer(data)
        self.assertEqual(out.x.shape, (10, 16))

    def test_forward_with_residual(self):
        data = self._make_data(10, dim=16)
        layer = EnchantedViaDFGLayer(dim_in=16, dim_out=16, dropout=0.0, residual=True)
        out = layer(data)
        self.assertEqual(out.x.shape, (10, 16))

    def test_residual_adds_input(self):
        torch.manual_seed(42)
        data = self._make_data(10, dim=16)
        layer_res = EnchantedViaDFGLayer(
            dim_in=16, dim_out=16, dropout=0.0, residual=True
        )
        layer_no_res = EnchantedViaDFGLayer(
            dim_in=16, dim_out=16, dropout=0.0, residual=False
        )
        layer_no_res.load_state_dict(layer_res.state_dict())
        out_res = layer_res(Data(x=data.x.clone(), edge_index=data.edge_index.clone()))
        out_no_res = layer_no_res(
            Data(x=data.x.clone(), edge_index=data.edge_index.clone())
        )
        diff = (out_res.x - out_no_res.x - data.x).abs().sum().item()
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_gated_fusion_uses_both_branches(self):
        """Verify that fusion_x depends on both feat_x and stru_x."""
        torch.manual_seed(42)
        data = self._make_data(10, dim=16)
        layer = EnchantedViaDFGLayer(dim_in=16, dim_out=16, dropout=0.0, residual=False)
        out1 = layer(Data(x=data.x.clone(), edge_index=data.edge_index.clone()))

        data2 = self._make_data(10, dim=16)
        data2.edge_index = data.edge_index.clone()
        out2 = layer(data2)

        self.assertFalse(
            torch.allclose(out1.x, out2.x),
            "Output should change when node features change (feat_x branch used)",
        )

        data3 = Data(x=data.x.clone(), edge_index=torch.randint(0, 10, (2, 20)))
        out3 = layer(data3)

        self.assertFalse(
            torch.allclose(out1.x, out3.x),
            "Output should change when edge_index changes (stru_x branch used)",
        )

    def test_batch_forward(self):
        data_list = [self._make_data(5, dim=16) for _ in range(3)]
        batch = Batch.from_data_list(data_list)
        layer = EnchantedViaDFGLayer(dim_in=16, dim_out=16, dropout=0.1, residual=True)
        out = layer(batch)
        self.assertEqual(out.x.shape, (15, 16))

    def test_dropout_applied(self):
        torch.manual_seed(42)
        data = self._make_data(10, dim=16)
        layer = EnchantedViaDFGLayer(dim_in=16, dim_out=16, dropout=0.5, residual=False)
        layer.eval()
        out_eval = layer(Data(x=data.x.clone(), edge_index=data.edge_index.clone()))

        layer.train()
        out_train = layer(Data(x=data.x.clone(), edge_index=data.edge_index.clone()))

        self.assertFalse(
            torch.allclose(out_eval.x, out_train.x),
            "Output should differ between train and eval due to dropout",
        )


if __name__ == "__main__":
    unittest.main()
