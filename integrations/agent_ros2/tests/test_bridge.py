import os
import sys
from types import SimpleNamespace
from io import StringIO
from unittest.mock import patch

TESTS_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, "..", "..", ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from integrations.agent_ros2.bridge import InMemoryRos2ScaffoldTransport, AgentRos2Bridge
from integrations.agent_ros2 import cli
from integrations.agent_ros2.messages import ACTION_TOPIC_MAP, TOPIC_SIMULATOR_COMMAND, TOPIC_TASK, OpenClawAction
from integrations.agent_ros2.transport_rclpy import RclpyRos2Transport


def test_action_topic_map_matches_expected_surface():
    assert ACTION_TOPIC_MAP["move_forward"] == TOPIC_SIMULATOR_COMMAND
    assert ACTION_TOPIC_MAP["turn_left"] == TOPIC_SIMULATOR_COMMAND
    assert ACTION_TOPIC_MAP["turn_right"] == TOPIC_SIMULATOR_COMMAND
    assert ACTION_TOPIC_MAP["turn_around"] == TOPIC_SIMULATOR_COMMAND
    assert ACTION_TOPIC_MAP["stop"] == TOPIC_SIMULATOR_COMMAND
    assert ACTION_TOPIC_MAP["ask_human"] == TOPIC_TASK
    assert ACTION_TOPIC_MAP["wait"] == TOPIC_TASK


def test_scaffold_bridge_reports_not_ready_without_transport():
    bridge = AgentRos2Bridge()

    status = bridge.get_status()
    result = bridge.move_forward(1.0)

    assert status["ros2_host"] == "127.0.0.1"
    assert status["ros2_port"] == 10000
    assert status["transport_ready"] is False
    assert result["ok"] is False
    assert result["scaffolded"] is True
    assert result["topic"] == TOPIC_SIMULATOR_COMMAND


def test_ready_transport_publishes_and_updates_state():
    transport = InMemoryRos2ScaffoldTransport(ready=True)
    bridge = AgentRos2Bridge(transport=transport)

    result = bridge.perform_action(
        OpenClawAction(action="ask_human", parameters={"prompt": "Need directions"})
    )
    observation = bridge.get_observation()

    assert result["ok"] is True
    assert result["topic"] == TOPIC_TASK
    assert transport.last_published_topic == TOPIC_TASK
    assert transport.last_published_payload["parameters"]["prompt"] == "Need directions"
    assert observation["last_task"]["parameters"]["prompt"] == "Need directions"


def test_rclpy_transport_falls_back_cleanly_when_ros2_runtime_is_missing():
    with patch("integrations.agent_ros2.transport_rclpy.importlib.import_module", side_effect=ImportError("no rclpy")):
        transport = RclpyRos2Transport()

    status = transport.get_status()

    assert transport.is_ready() is False
    assert status["transport_live"] is False
    assert status["transport_ready"] is False
    assert "no rclpy" in status["transport_error"]
    assert transport.publish(TOPIC_SIMULATOR_COMMAND, {"action": "move_forward"}) is False


def test_rclpy_transport_can_publish_and_track_observations_with_fake_runtime():
    class FakeString:
        def __init__(self) -> None:
            self.data = ""

    class FakeImage:
        def __init__(self, width: int = 64, height: int = 48) -> None:
            self.width = width
            self.height = height

    class FakeOdometry:
        def __init__(self) -> None:
            self.pose = SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            )

    class FakePublisher:
        def __init__(self) -> None:
            self.messages = []

        def publish(self, message) -> None:
            self.messages.append(message.data)

    class FakeNode:
        def __init__(self, name: str) -> None:
            self._name = name
            self.publishers = {}
            self.subscriptions = {}

        def create_publisher(self, msg_type, topic: str, qos_depth: int):
            publisher = FakePublisher()
            self.publishers[topic] = (msg_type, qos_depth, publisher)
            return publisher

        def create_subscription(self, msg_type, topic: str, callback, qos_depth: int):
            self.subscriptions[topic] = (msg_type, qos_depth, callback)
            return callback

        def get_name(self) -> str:
            return self._name

        def destroy_node(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self) -> None:
            self.nodes = []

        def add_node(self, node) -> None:
            self.nodes.append(node)

        def spin_once(self, timeout_sec: float) -> None:
            return None

        def shutdown(self) -> None:
            return None

    class FakeRclpy:
        def __init__(self) -> None:
            self._ok = False

        def ok(self) -> bool:
            return self._ok

        def init(self, args=None) -> None:
            self._ok = True

        def shutdown(self) -> None:
            self._ok = False

    fake_modules = {
        "rclpy": FakeRclpy(),
        "node": SimpleNamespace(Node=FakeNode),
        "executors": SimpleNamespace(SingleThreadedExecutor=FakeExecutor),
        "std_msgs": SimpleNamespace(String=FakeString),
        "sensor_msgs": SimpleNamespace(Image=FakeImage),
        "nav_msgs": SimpleNamespace(Odometry=FakeOdometry),
    }

    transport = RclpyRos2Transport(_modules=fake_modules)
    published = transport.publish(TOPIC_SIMULATOR_COMMAND, {"action": "move_forward", "parameters": {"distance_m": 1.0}})
    transport._on_ack(SimpleNamespace(data='{"ok": true, "request_id": "abc"}'))
    transport._on_rgb(FakeImage())
    transport._on_depth(FakeImage())
    transport._on_odom(FakeOdometry())

    status = transport.get_status()
    observation = transport.get_observation()

    assert published is True
    assert transport.is_ready() is True
    assert status["transport_ready"] is True
    assert status["last_published_topic"] == TOPIC_SIMULATOR_COMMAND
    assert observation["last_ack"]["request_id"] == "abc"
    assert observation["rgb_available"] is True
    assert observation["depth_available"] is True
    assert observation["pose"]["position"]["x"] == 1.0

    transport.close()


def test_cli_observe_waits_before_reading_observation():
    class FakeBridge:
        def __init__(self) -> None:
            self.wait_calls = []

        def run_wait(self, seconds: float) -> None:
            self.wait_calls.append(seconds)

        def get_observation(self):
            return {"rgb_available": True, "depth_available": False}

        @property
        def transport(self):
            return None

    fake_bridge = FakeBridge()
    stdout = StringIO()

    with patch("integrations.agent_ros2.cli._build_bridge", return_value=fake_bridge):
        with patch("sys.stdout", stdout):
            exit_code = cli.main(["observe", "--wait-seconds", "3", "--output-json"])

    assert exit_code == 0
    assert fake_bridge.wait_calls == [3.0]
    assert '"rgb_available": true' in stdout.getvalue()
