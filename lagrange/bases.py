# bases.py

from typing import *


def all_subclasses(cls):
    "Generator. Wrap in list"
    for subcls in cls.__subclasses__():
        yield subcls
        yield from all_subclasses(subcls)


class Base(object):
    class LagrangeError(Exception):
        pass

    def __repr__(self):
        return f"<{self.__class__.__module__}.{self.__class__.__name__}: {{}}>"

    def __str__(self):
        return Base.__repr__(self).format('...')

    def _make_repr(self, *attrs: Union[str, Tuple[str, str]],
                   **kwargs: Any):
        """
        Make a repr
        
        :param attrs:
            attributes of self to be shown OR tuples of form (attr, format_specifier)
        :param kwargs: 
            plain {attribute_name: value passes} to be included
        :return: 
        """

        attrs = list(attrs)
        for index, attr in enumerate(attrs):
            if isinstance(attr, str):
                attrs[index] = (attr, '')

        assert all(all(isinstance(i, str) for i in t) for t in attrs), attrs
        assert all(len(attr) == 2 for attr in attrs), attrs

        fmt_attrs = [f'{attr}={getattr(self, attr):{fmt}}'
                     for attr, fmt in attrs]
        fmt_attrs.extend(f'{k}={v}' for k, v in kwargs.items())
        # order preserved in py > 3.6

        return Base.__repr__(self).format(', '.join(fmt_attrs))

    @property
    def _name(self):
        "Shortcut for self.__class__.__name__"
        return self.__class__.__name__

    subclasses = classmethod(all_subclasses)
