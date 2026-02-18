import pickle
import os


class StubManager:
    """Manages reading and writing of stub (cached) data to avoid re-computation."""

    @staticmethod
    def load(stub_path):
        """Load cached data from a stub file. Returns None if file doesn't exist."""
        if stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                data = pickle.load(f)
            print(f"[STUB] Loaded cached data from {stub_path}")
            return data
        return None

    @staticmethod
    def save(data, stub_path):
        """Save data to a stub file for future reuse."""
        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"[STUB] Saved cached data to {stub_path}")
