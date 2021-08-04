def _npy_dumps(data):
    """
    Serializes a numpy array into a stream of npy-formatted bytes.
    """
    from six import BytesIO
    import numpy as np
    buffer = BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()
