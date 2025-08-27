import pandas as pd
import jax
import numpy as np
import jax.numpy as jnp


def write_feather(df, filename):
    types_to_change = df.dtypes[df.dtypes == "object"]
    types_to_change = {x: "float64" for x in types_to_change.index}
    df = df.astype(types_to_change)
    df.to_feather(filename)


################################################################################
# Helper functions ------------------------------------------------------------
################################################################################


def dmap(func, d, is_leaf=lambda v: not isinstance(v, dict)):
    """
    map functions to leaves of nested dictionary
    """
    return {
        k: (func(v) if is_leaf(v) else dmap(func, v, is_leaf)) for k, v in d.items()
    }


# test = dict(a=2, b=3, c=dict(d=4, e=5))
# dmap(lambda x: x**2, test)


# these function are helpful for filtering down a datframe
def row_filter(df, key, op):
    """
    df is a dataframe we want to filter rows
    key = column of df
    op is a function that return boolean indicator to retain the row
    """
    return df[op(df[key].values)]


# pd.DataFrame.row_filter = row_filter


def geq(z):
    return lambda x: x >= z


def leq(z):
    return lambda x: x <= z


def between(z1, z2):
    return lambda x: (x >= z1) & (x <= z2)


def close(z):
    return lambda x: np.isclose(x, z)


def eq(z):
    return lambda x: x == z


# helpful for manipulating vmapped jax output
def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def tree_unstack(tree):
    leaves, treedef = jax.tree.flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


def extract(pytree, idx):
    return jax.tree.map(lambda x: x[idx] if jnp.atleast_1d(x).size > 1 else x, pytree)
