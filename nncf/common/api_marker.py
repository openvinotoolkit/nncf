class api:
    API_MARKER_ATTR = "_nncf_api_marker"

    def __init__(self):
        pass

    def __call__(self, obj):
        # The value of the marker will be useful in determining
        # whether we are handling a base class or a derived one.
        setattr(obj, api.API_MARKER_ATTR, obj.__name__)
        return obj
