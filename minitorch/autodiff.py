from dataclasses import dataclass
from typing import Any, Iterable, Tuple, List

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    ls1: List[Variable] = []
    ls2: List[Variable] = []
    ls1.extend(vals[0:arg])
    ls1.append(vals[arg] + epsilon)
    ls1.extend(vals[arg + 1 :])
    ls2.extend(vals[0:arg])
    print(ls1)
    ls2.append(vals[arg] - epsilon)
    ls2.extend(vals[arg + 1 :])
    return (f(*ls1) - f(*ls2)) / (2 * epsilon)
    # TODO: Implement for Task 1.1.


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order

    # permMarks = {}
    # sorted_nodes: List[Variable] = []
    # Nodes = [variable]

    # def visit(node: Variable) -> None:
    #     if node.unique_id in permMarks:
    #         return
    #     for n in node.parents:
    #         visit(n)
    #     permMarks[node.unique_id] = 0
    #     sorted_nodes.insert(0, node)

    # while Nodes:
    #     visit(Nodes.pop())
    # return sorted_nodes

    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # sorted_nodes: Iterable[Variable] = topological_sort(variable)

    # derivatives = {}
    # derivatives[variable.unique_id] = deriv
    # for var in sorted_nodes:
    #     currDev = derivatives.pop(var.unique_id, 0)
    #     if var.is_leaf():
    #         var.accumulate_derivative(currDev)
    #         continue
    #     for v, d in var.chain_rule(currDev):
    #         v_id = v.unique_id
    #         if v_id in derivatives:
    #             derivatives[v_id] += d
    #         else:
    #             derivatives[v_id] = d

    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d

    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
