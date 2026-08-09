class DesignError(ValueError):
    """Base class for invalid declarative designs."""


class DesignSchemaError(DesignError):
    """Raised when a design document violates schema rules."""


class DesignParameterError(DesignError):
    """Raised when parameter references or strict parameter checks fail."""


class DesignResolutionError(DesignError):
    """Raised when a hierarchical node or element path cannot be resolved."""


class DesignCollisionError(DesignError):
    """Raised when cursor ranges or element names collide."""
