class TactileDataCollector:
    """Collects tactile sensor data during simulation."""

    def __init__(self, scene: InteractiveScene, sensor_names: List[str]):
        self.scene = scene
        self.sensor_names = sensor_names
        self.buffer = []

    def collect_step(self, step: int):
        """Collect data for current step."""
        step_data = {"step": step}

        for sensor_name in self.sensor_names:
            sensor = self.scene[sensor_name]
            step_data[sensor_name] = {
                "rgb": sensor.data.tactile_rgb_image,
                "depth": sensor.data.tactile_depth_image,
                "normal_force": sensor.data.tactile_normal_force,
                "shear_force": sensor.data.tactile_shear_force,
            }

        self.buffer.append(step_data)

    def get_buffer(self) -> List[Dict]:
        """Returns collected data."""
        return self.buffer

    def clear_buffer(self):
        """Clears collected"""
        self.buffer = []