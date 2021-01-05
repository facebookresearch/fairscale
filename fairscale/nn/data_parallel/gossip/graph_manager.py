# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Graph Manager Class

:description: Class provides an API for loading different peer-to-peer
    communication topologies, and cycling through peers.
"""

from math import log as mlog

import torch
import torch.distributed as dist


class Edge(object):
    def __init__(self, local_master_rank, dest, src, local_rank):
        self.src = src
        self.dest = dest
        self.process_group = dist.new_group([src, dest])
        if local_master_rank in [self.src, self.dest] and local_rank == 0:
            initializer_tensor = torch.Tensor([1]).cuda()
            dist.all_reduce(initializer_tensor, group=self.process_group)
            initializer_tensor = torch.Tensor([1]).cuda().half()
            dist.all_reduce(initializer_tensor, group=self.process_group)


class GraphManager(object):
    def __init__(
        self, rank, world_size, nprocs_per_node=1, local_rank=0, peers_per_itr=1
    ):
        assert int(peers_per_itr) >= 1
        self.rank = rank
        self.world_size = world_size
        self.phone_book = [[] for _ in range(self.world_size)]
        self._peers_per_itr = peers_per_itr
        self._group_indices = list(range(peers_per_itr))
        self.nprocs_per_node = nprocs_per_node
        self.local_rank = local_rank
        self._make_graph()

    @property
    def peers_per_itr(self):
        return self._peers_per_itr

    @peers_per_itr.setter
    def peers_per_itr(self, v):
        self._peers_per_itr = v
        # set group-indices attr. --- point to out-peers in phone-book
        self._group_indices = list(range(v))

    def _make_graph(self):
        """
        Returns a nested list of peers; the outer-list is indexed by rank,
        the inner list denotes the set of peers that 'rank' can send
        messages to at any point in time
        """
        raise NotImplementedError

    def _add_peers(self, rank, peers):
        for peer in peers:
            if peer not in self.phone_book[rank]:
                self.phone_book[rank].append(
                    Edge(
                        local_master_rank=(self.rank * self.nprocs_per_node),
                        dest=(peer * self.nprocs_per_node),
                        src=(rank * self.nprocs_per_node),
                        local_rank=self.local_rank,
                    )
                )

    def is_regular_graph(self):
        """ Whether each node has the same number of in-peers as out-peers """
        raise NotImplementedError

    def is_bipartite_graph(self):
        """ Whether graph is bipartite or not """
        raise NotImplementedError

    def is_passive(self, rank=None):
        """ Whether 'rank' is a passive node or not """
        raise NotImplementedError

    def is_dynamic_graph(self, graph_type=None):
        """ Whether the graph-type is dynamic (as opposed to static) """
        raise NotImplementedError

    def get_peers(self, rotate=False):
        """ Returns the out and in-peers corresponding to 'self.rank' """
        # cycle through in- and out-peers by updating group-index
        if rotate:
            self._rotate_group_indices()

        # get out- and in-peers using new group-indices
        out_peers, in_peers = [], []
        for group_index in self._group_indices:
            out_peers.append(self.phone_book[self.rank][group_index].dest)
            for rank, peers in enumerate(self.phone_book):
                if rank == self.rank:
                    continue
                if self.rank * self.nprocs_per_node == peers[group_index].dest:
                    in_peers.append(rank)
        return out_peers, in_peers

    def get_edges(self, rotate=False):
        """ Returns the pairwise process groups between rank and the out and
        in-peers corresponding to 'self.rank' """
        # cycle through in- and out-peers by updating group-index
        if rotate:
            self._rotate_group_indices()

        # get out- and in-peers using new group-indices
        out_edges, in_edges = [], []
        for group_index in self._group_indices:
            out_edges.append(self.phone_book[self.rank][group_index])
            for rank, edges in enumerate(self.phone_book):
                if rank == self.rank:
                    continue
                if self.rank * self.nprocs_per_node == edges[group_index].dest:
                    in_edges.append(self.phone_book[rank][group_index])
        return out_edges, in_edges

    def _rotate_group_indices(self):
        """ Incerement group indices to point to the next out-peer """
        increment = self.peers_per_itr
        for i, group_index in enumerate(self._group_indices):
            self._group_indices[i] = int(
                (group_index + increment) % len(self.phone_book[self.rank])
            )

    def _rotate_forward(self, r, p):
        """ Helper function returns peer that is p hops ahead of r """
        return (r + p) % self.world_size

    def _rotate_backward(self, r, p):
        """ Helper function returns peer that is p hops behind r """
        temp = r
        for _ in range(p):
            temp -= 1
            if temp < 0:
                temp = self.world_size - 1
        return temp


class DynamicDirectedExponentialGraph(GraphManager):
    def _make_graph(self):
        for rank in range(self.world_size):
            for i in range(0, int(mlog(self.world_size - 1, 2)) + 1):
                f_peer = self._rotate_forward(rank, 2 ** i)
                b_peer = self._rotate_backward(rank, 2 ** i)
                self._add_peers(rank, [f_peer, b_peer])

    def is_regular_graph(self):
        return True

    def is_bipartite_graph(self):
        return False

    def is_passive(self, rank=None):
        return False

    def is_dynamic_graph(self, graph_type=None):
        return True


class NPeerDynamicDirectedExponentialGraph(GraphManager):
    def _make_graph(self):
        for rank in range(self.world_size):
            for i in range(
                0, int(mlog(self.world_size - 1, self._peers_per_itr + 1)) + 1
            ):
                for j in range(1, self._peers_per_itr + 1):
                    distance_to_neighbor = j * ((self._peers_per_itr + 1) ** i)
                    f_peer = self._rotate_forward(rank, distance_to_neighbor)
                    self._add_peers(rank, [f_peer])

    def is_regular_graph(self):
        return True

    def is_bipartite_graph(self):
        return False

    def is_passive(self, rank=None):
        return False

    def is_dynamic_graph(self, graph_type=None):
        return True


class DynamicBipartiteExponentialGraph(GraphManager):
    def _make_graph(self):
        for rank in range(self.world_size):
            for i in range(0, int(mlog(self.world_size - 1, 2)) + 1):
                if i == 0:
                    f_peer = self._rotate_forward(rank, 1)
                    b_peer = self._rotate_backward(rank, 1)
                else:
                    f_peer = self._rotate_forward(rank, 1 + 2 ** i)
                    b_peer = self._rotate_backward(rank, 1 + 2 ** i)
                # create directory for non-passive peers
                if not self.is_passive(rank) and (
                    self.is_passive(f_peer) and self.is_passive(b_peer)
                ):
                    self._add_peers(rank, [f_peer, b_peer])
                # create directory for passive peers
                elif self.is_passive(rank) and (
                    not (self.is_passive(f_peer) or self.is_passive(b_peer))
                ):
                    self._add_peers(rank, [f_peer, b_peer])

    def is_regular_graph(self):
        return True

    def is_bipartite_graph(self):
        return True

    def is_passive(self, rank=None):
        rank = self.rank if rank is None else rank
        return (rank % 2) == 0

    def is_dynamic_graph(self, graph_type=None):
        return True


class DynamicDirectedLinearGraph(GraphManager):
    def _make_graph(self):
        for rank in range(self.world_size):
            for i in range(1, self.world_size):
                if i % 2 == 0:
                    continue
                f_peer = self._rotate_forward(rank, i)
                b_peer = self._rotate_backward(rank, i)
                self._add_peers(rank, [f_peer, b_peer])

    def is_regular_graph(self):
        return True

    def is_bipartite_graph(self):
        return False

    def is_passive(self, rank=None):
        return False

    def is_dynamic_graph(self, graph_type=None):
        return True


class DynamicBipartiteLinearGraph(GraphManager):
    def _make_graph(self):
        for rank in range(self.world_size):
            for i in range(1, self.world_size):
                f_peer = self._rotate_forward(rank, i)
                b_peer = self._rotate_backward(rank, i)
                # create directory for non-passive peers
                if not self.is_passive(rank) and (
                    self.is_passive(f_peer) and self.is_passive(b_peer)
                ):
                    self._add_peers(rank, [f_peer, b_peer])
                # create directory for passive peers
                elif self.is_passive(rank) and (
                    not (self.is_passive(f_peer) or self.is_passive(b_peer))
                ):
                    self._add_peers(rank, [f_peer, b_peer])

    def is_regular_graph(self):
        return True

    def is_bipartite_graph(self):
        return True

    def is_passive(self, rank=None):
        rank = self.rank if rank is None else rank
        return (rank % 2) == 0

    def is_dynamic_graph(self, graph_type=None):
        return True


class RingGraph(GraphManager):
    def _make_graph(self):
        for rank in range(self.world_size):
            f_peer = self._rotate_forward(rank, 1)
            b_peer = self._rotate_backward(rank, 1)
            self._add_peers(rank, [f_peer, b_peer])

    def is_regular_graph(self):
        return True

    def is_bipartite_graph(self):
        return False

    def is_passive(self, rank=None):
        return False

    def is_dynamic_graph(self, graph_type=None):
        return False