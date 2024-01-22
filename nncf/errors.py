class ValidationError(Exception):
    """
        Raised when an internal NNCF validation check fails, for example, if the user supplied an invalid or inconsistent set of arguments.
    """
    pass

class InternalError(Exception):
    """
    Raised when an internal error occurs within the NNCF framework.

    This exception is raised when an unexpected internal error occurs during the execution
    of NNCF. It indicates a situation where the code encountered an unexpected condition.

    """
    pass

class UnsupportedDatasetError(Exception):
    """
    Raised when an unsupported dataset is encountered
    """
    pass

class InvalidPathError(Exception):
    """
    Raised when an invalid path is provided
    """
    pass

class UnsupportedBackendError(Exception):
    """
    Raised when an unsupported backend is specified
    """
    pass

class InvalidQuantizerGroupError(Exception):
    """
    Raised when an invalid quantizer group is encountered.
    """
    pass

class InstallationError(Exception):
    """
    Raised when an error occurs during installation.
    """
    pass

class UnsupportedModelError(Exception):
    """
    Raised when an unsupported model is encountered.
    """
    pass

class UnsupportedVersionError(Exception):
    """
    Raised when an unsupported version is encountered.
    """
    pass

class ModuleNotFoundError(Exception):
    """
    Raised when a required module is not found.
    """
    pass

class ParameterNotSupportedError(Exception):
    """
    Raised when an unsupported parameter is encountered.

    This error indicates that an unsupported parameter was provided, typically in the context
    of a compression algorithm's configuration or an argument passed to an API call.

    """
    pass

class InvalidCollectorTypeError(Exception):
    """
    Raised when an invalid collector type is encountered.
    """
    pass

class BufferFullError(Exception):
    """
    Raised when a buffer is full
    """
    pass

class UnknownDatasetError(Exception):
    """
    Raised when an unknown dataset is encountered.
    """
    pass