"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from itertools import chain, combinations
from typing import Callable, List
from typing import Dict

import numpy as np
import networkx as nx


def powerset(iterable, min_r: int = 1, max_r: int = None):
    if not isinstance(iterable, list):
        s = list(iterable)
    else:
        s = iterable
    if max_r is None:
        max_r = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(min_r, max_r + 1))


class Expression:
    """
    A base class for objects that represent an expression to be matched against a subgraph of a directed
    acyclic graph. Overloads certain Python operator to define a kind of a domain-specific mini-language for
    specifying a DAG subgraph in a single line. `Expression`-based subgraph specification is not expressive
    enough to cover all cases required by NNCF, e.g. in the case of the swish activation x * sigmoid(x), which
    cannot be unambiguosly specified using `Expression`s.
    """
    def _match(self, nodes, graph):
        return NotImplementedError

    def __add__(self, other):
        return ConcatExpression([self, other])

    def __or__(self, other):
        return AlternatingExpression([self, other])

    def __and__(self, other):
        return BranchingExpression([self, other])

    def _iterate_alternatives(self, nodes):
        return powerset(nodes, min_r=1)

    def _find_all_matches(self,  nodes: List[Dict], graph: nx.DiGraph):
        all_matches = []
        for n in self._iterate_alternatives(nodes):
            result = self._match(n, graph)
            if not result:
                continue
            if not isinstance(result, list):
                result = [result]
            for res in result:
                n, following = res
                following = list(following)
                if not isinstance(n, list):
                    n = [n]

                all_matches.append((n, following))
        return all_matches

    def all_matches(self, nodes: List[Dict], graph: nx.DiGraph):
        return self._find_all_matches(nodes, graph)

    def match(self, nodes: List[Dict], graph: nx.DiGraph):
        all_matches = self._find_all_matches(nodes, graph)
        if not all_matches:
            return None, None
        return max(all_matches, key=lambda x: len(x[0]))


class ConcatExpression(Expression):
    """
    A composite expression that matches to a node path in the graph which is a concatenation
    of the paths that each matches, in order, to the sub-expression of this expression.
    E.g. a subgraph (conv2d) -> (batch_norm) will be matched to the expression:
        ConcatExpression([NodeExpression('conv2d'), NodeExpression('batch_norm2d')])
    or, equivalently, using the overloaded operator syntax:
        NodeExpression('conv2d') + NodeExpression('batch_norm2d')
    """
    def __init__(self, expressions: List[Expression]):
        """
        :param expressions: A list of subexpressions to be matched in a concat-fashion
        """
        self.expressions = expressions

    def _match(self, nodes, graph):
        assert len(self.expressions) > 1
        full_match = []
        for ex in self.expressions:
            matches, following = ex.match(nodes, graph)

            if not matches:
                return None

            full_match += matches

            nodes = following
        return full_match, following

    def __add__(self, other):
        return ConcatExpression(self.expressions + [other])


class AlternatingExpression(Expression):
    """
    A composite expression that matches to any of the node path in the graph which matches to at least
     one of the subexpressions.
    E.g. in a graph (conv2d) -> (batch_norm) the expression:
        AlternatingExpression([NodeExpression('conv2d'), NodeExpression('batch_norm2d')]),
    or, equivalently,
        NodeExpression('conv2d') | NodeExpression('batch_norm2d')
    will match both to the (conv2d) node and the (batch_norm) node.
    """
    def __init__(self, expressions, greedy_match=False, greedy_consume=True):
        """
        :param expressions: A list of subexpressions to be matched in an alternating-fashion
        :param greedy_match:
        :param greedy_consume:
        """
        self.greedy_match = greedy_match
        self.greedy_consume = greedy_consume
        self.expressions = expressions

    def _match(self, nodes, graph):
        assert len(self.expressions) > 1
        all_matches = []
        for ex in self.expressions:
            matched, following = ex.match(nodes, graph)
            if not matched:
                continue

            if self.greedy_match:
                return matched, following

            all_matches.append((matched, following))

        if self.greedy_consume:
            if not all_matches:
                return None
            return all_matches
        return None

    def __or__(self, other):
        return AlternatingExpression(self.expressions + [other])


class BranchingExpression(Expression):
    """
    A composite expression that matches to a subgraph which is composed of separate branches, each matching
    against a subexpression; the branches are united by one and the same sourcing node.
    E.g. in a graph:
                |--> (max_pool2d)
         (conv2d)--> (batch_norm) the expression:
                |--> (RELU)

    the expression given by
        BranchingExpression([NodeExpression('max_pool2d'), NodeExpression('batch_norm'), NodeExpression('RELU')])
    or, alternatively,
        NodeExpression('max_pool2d') & NodeExpression('batch_norm') & NodeExpression('RELU')
    will match to the (max_pool2d), (batch_norm) and the (RELU) nodes.
    """
    def __init__(self, expressions):
        """
        :param expressions: A list of subexpressions to be matched as branches
        """
        self.expressions = expressions

    def _iterate_alternatives(self, nodes):
        return powerset(nodes, len(self.expressions), len(self.expressions))

    def _match(self, nodes, graph):
        assert len(self.expressions) > 1
        if len(nodes) != len(self.expressions):
            # need to try all possible assignments
            return None

        matches = [[] for _ in range(len(self.expressions))]
        for i, ex in enumerate(self.expressions):
            any_matched = False
            for node_name in nodes:
                matched, following = ex.match([node_name], graph)
                matches[i].append((matched, following))

                if matched:
                    any_matched = True

            if not any_matched:
                return None

        return self._assign_matches(matches)

    def _assign_matches(self, matches):
        """Assign every expression to some match"""
        assignments = np.full(len(matches[0]), -1)
        used = np.full(len(matches), False)

        def _find_assignment(i):
            if used[i]:
                return False
            used[i] = True
            for j in range(len(matches[0])):
                if not matches[i][j][0]:
                    continue
                if assignments[j] == -1 or _find_assignment(assignments[j]):
                    assignments[j] = i
                    return True
            return False

        for i in range(len(self.expressions)):
            used[...] = False
            _find_assignment(i)

        all_matches = set()
        all_followings = set()
        for i in range(len(matches[0])):
            if assignments[i] != -1:
                match, follow = matches[assignments[i]][i]
                all_matches.update(match)
                all_followings.update(follow)

        # assume matches dot not end in other match
        if all_matches & all_followings:
            return None
        return list(all_matches), list(all_followings)

    def __and__(self, other):
        return BranchingExpression(self.expressions + [other])


class NodeExpression(Expression):
    """
    A basic Expression that is matched against a single node; the node descriptor is expected to be a dict commonly
    seen in networkx graphs.
    """
    def __init__(self, node_type: str = None,
                 filter_fn: Callable[[Dict], bool] = None,
                 node_type_fn: Callable[[dict], str] = None):
        """
        :param node_type: A string value to be compared with a value returned by node_type_fn to determine
        an expression match.
        :param filter_fn: A predicate for nodes that are to be disregarded during matching.
        :param node_type_fn: A function that accepts a node dict and returns a value to be matched against node_type
        """
        self.filter = filter_fn
        self.node_type = node_type
        if node_type_fn is None:
            self.node_type_fn = lambda x: x['type']
        else:
            self.node_type_fn = node_type_fn

    def _iterate_alternatives(self, nodes):
        for node in nodes:
            yield [node]

    def _match(self, nodes: List[Dict], graph: nx.DiGraph):
        if len(nodes) != 1:
            return None

        node_name = nodes[0]
        node = graph.nodes[node_name]
        node_type = self.node_type_fn(node)
        if self.node_type == node_type:
            if self.filter and not self.filter(node):
                return None

            following = graph.successors(node_name)
            return node_name, following
        return None


def get_edge_boundaries(match: List[str], graph: nx.DiGraph):
    out_edge_boundary = list(nx.edge_boundary(graph, match, data=True))
    complement = list(filter(lambda x: x not in match, graph.nodes.keys()))
    in_edge_boundary = list(nx.edge_boundary(graph, complement, data=True))
    return sorted(in_edge_boundary), sorted(out_edge_boundary)


def find_whether_subgraph_has_inner_outgoing_edges(graph: nx.DiGraph, subgraph: List[str]) -> bool:
    """
    Check out whether the subgraph has outgoing edges starting not from the last node.
    Example:
    (conv2d + BN + ReLU pattern):
            ...
             |
          (conv2d)
             |------\
            (BN)    |
             |      |
           (RELU)   |
             |      |
           (cat)----/
             |
            ...
    :param graph: The model graph.
    :param subgraph: A subgraph of the model graph.
    :return: True if the subgraph contains outgoing edges starting not from the last node,
        False - otherwise.
    """
    for node_key in subgraph[:-1]:
        successors = list(graph.succ[node_key].keys())
        for successors_key in successors:
            if successors_key not in subgraph:
                return True

    # Breaking input edges
    for node_key in subgraph[1:]:
        predecessors = list(graph.pred[node_key].keys())
        for predecessors_key in predecessors:
            if predecessors_key not in subgraph:
                return True
    return False


def find_subgraphs_matching_expression(graph: nx.DiGraph, expression: Expression) -> List[List[str]]:
    """
    Find a list of subgraphs for the particular graph that match the pattern expression.
    :param graph: The model graph.
    :param expression: A pattern expression containing a logic of layer fusing.
    :return: A list of subgraphs for the particular graph, matching the pattern expression.
    """
    subgraphs = []
    subgraphs_nodes = set()
    nodes = nx.topological_sort(graph)
    for node in nodes:
        # If a node has already been added to any pattern skip this node
        if node in subgraphs_nodes:
            continue

        all_matches = expression.all_matches([node], graph)
        all_matches = sorted(all_matches, key=lambda x: len(x[0]), reverse=True)

        longest_valid_match = None
        for match in all_matches:
            # Find out the longest valid pattern
            if not find_whether_subgraph_has_inner_outgoing_edges(graph, match[0]):
                longest_valid_match = match
                break
        # If there is no pattern found, then skip this node
        if longest_valid_match is None:
            continue

        for mn in longest_valid_match[0]:
            subgraphs_nodes.add(mn)

        subgraphs.append(longest_valid_match[0])
    return subgraphs
