# bodies.py

from .bases import Base

G = 6.674e-11


class Body(Base):
    instances = {}

    def __init__(self, name: str, mass: float, sma: float, period: float):
        """
        :param name:
        :param mass: in kg
        :param sma: in m
        :param period: in s
        """
        self.name = name.lower().strip()
        self.instances[self.name] = self

        self.mass = mass
        self.sma = sma
        self.period = period

    def total_mass(self, mu: float):
        "Must be smaller mass"
        # mu = m2 / (m1 + m2)
        return self.mass/mu

    def make_mu(self, other):
        try:
            othermass = other.mass
        except AttributeError:
            othermass = other
        lower, higher = sorted((self.mass, othermass))
        return lower/(higher + lower)

    @classmethod
    def get_instance(cls, name):
        return cls.instances[name.lower().strip()]

        # inherit __repr__


earth = Body('earth', 5.97237e24, 149598023e3, 365.256363004*24*60**2)
moon = Body('moon', 7.342e22, 384399e3, 27.321661*224*60**2)
sun = Body('sun', 1.98855e30, None, None)


def get_body(name: str) -> Body:
    return Body.get_instance(name)
