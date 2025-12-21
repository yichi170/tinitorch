"""Device abstraction for TiniTorch."""


class Device:
    def __init__(self, device_str: str):
        if device_str == "cpu":
            self.type = "cpu"
            self.index = None
        elif device_str == "cuda":
            self.type = "cuda"
            self.index = 0
        elif device_str.startswith("cuda:"):
            self.type = "cuda"
            self.index = int(device_str.split(":")[1])
        else:
            raise ValueError(f"Invalid device string: {device_str}")

    def __eq__(self, other):
        if isinstance(other, str):
            other = Device(other)
        return self.type == other.type and self.index == other.index

    def __repr__(self):
        if self.type == "cpu":
            return "device(type='cpu')"
        else:
            return f"device(type='cuda', index={self.index})"

    def __str__(self):
        if self.type == "cpu":
            return "cpu"
        else:
            return f"cuda:{self.index}"
