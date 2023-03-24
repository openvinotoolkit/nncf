class api:
    API_MARKER_ATTR = "_nncf_api_marker"

    def __init__(self):
        pass

    def __call__(self, obj):
        setattr(obj, api.API_MARKER_ATTR, True)
