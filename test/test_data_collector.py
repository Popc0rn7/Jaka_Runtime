from src.data_collector import LeRobotDataCollector


class _DatasetWithEpisodeBuffer:
    def __init__(self) -> None:
        self.episode_buffer = {"size": 2, "episode_index": 7}
        self.waited = False
        self.cleared = False

    def _wait_image_writer(self) -> None:
        self.waited = True

    def clear_episode_buffer(self) -> None:
        self.cleared = True
        self.episode_buffer = {"size": 0, "episode_index": 7}


def test_discard_episode_relies_on_lerobot_buffer_cleanup() -> None:
    collector = LeRobotDataCollector.__new__(LeRobotDataCollector)
    collector._closed = False
    collector.dataset = _DatasetWithEpisodeBuffer()

    collector.discard_episode()

    assert collector.dataset.waited
    assert collector.dataset.cleared
    assert not collector.has_pending_frames()
