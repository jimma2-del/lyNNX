"""Utilities relating to handling functions."""

from typing import Callable, TypeVar, Any, ParamSpec

import inspect
import functools

P = ParamSpec('P')
TReturn = TypeVar('TReturn')

def try_call(possibly_callable: Callable[P, TReturn] | TReturn, *args: P.args, **kwargs: P.kwargs) -> TReturn:
    """Calls the input and returns the result if it is callable; returns the input directly otherwise."""
    if callable(possibly_callable): return possibly_callable(*args, **kwargs)
    return possibly_callable

def optionally_pass(func: Callable[..., TReturn], *opt_args: Any, **opt_kwargs: Any) -> Callable[..., TReturn]:
    """Returns `func` with arguments bound if they exist in `func`.
    
    Allows passing arguments to a function only if they function accepts them.
    """

    params = inspect.signature(func).parameters

    # check for *args or **kwargs in params
    accepts_all_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    accepts_all_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())

    # all positional params (POSITIONAL_ONLY & POSITIONAL_OR_KEYWORD)
    pos_params = [ (name, p) for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) ]

    if not accepts_all_kwargs: # only pass opt_kwargs if present; if accepts **kwargs, pass all
        opt_kwargs = {k: v for k, v in opt_kwargs.items() if k in params}
    
    def func_with_opt_args(*args: Any, **kwargs: Any) -> TReturn:
        all_named_args_keys = set(kwargs).union(set(opt_kwargs)) # set(dict) gives keys
    
        # final list of all positional args passed
        final_pos_args = list(args) # args are definitely passed, we may add more from opt_args
    
        if not accepts_all_args: # only pass opt_args if space; if accepts *args, pass all
            opt_args_i = 0

            # iterate through pos_params to find which slots are still open
            for slot, (name, p) in enumerate(pos_params):
                if opt_args_i >= len(opt_args): break  # no more optional args to place
                if slot < len(args): continue # slots 0 to len(args)-1 are filled by args already
                if name in all_named_args_keys: continue  # already covered by a kwarg
    
                # free slot found -> fill
                if p.kind == inspect.Parameter.POSITIONAL_ONLY:
                    final_pos_args.append(opt_args[opt_args_i])
                else: # POSITIONAL_OR_KEYWORD; pass as kwarg to avoid index confusion
                    opt_kwargs[name] = opt_args[opt_args_i]

                opt_args_i += 1
    
        else: # accepts *args; pass all
            final_pos_args.extend(opt_args)
    
        return func(*final_pos_args, **{**opt_kwargs, **kwargs})

    return func_with_opt_args

def override_signature(*args: P.args, **kwargs: P.kwargs) \
        -> Callable[[Callable[..., TReturn]], Callable[P, TReturn]]:
    """Factory for a decorator which overrides the function's signature to match `*args` and `**kwargs`.
        Overrides the `__annotations__` and `__signature__` fields.

    Values of `*args` and `**kwargs` can be either types, or values from which the type will be inferred.

    Positional parameters in the new signature will be named `arg0`, `arg1`, ...
    """

    def decorator(func: Callable[..., TReturn]) -> Callable[P, TReturn]:
        args_spec = { f'arg{i}': val if isinstance(val, type) else type(val) for i, val in enumerate(args) }
        kwargs_spec = { key: val if isinstance(val, type) else type(val) for key, val in kwargs.items() }
        func.__annotations__ = args_spec | kwargs_spec

        params = [ inspect.Parameter(key, inspect.Parameter.POSITIONAL_ONLY, annotation=type_hint)
            for key, type_hint in args_spec.items() ]
        params += [ inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY, annotation=type_hint)
            for key, type_hint in kwargs_spec.items() ]

        func.__signature__ = inspect.Signature(params)
        return func

    return decorator