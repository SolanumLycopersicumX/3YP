from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch


class FakeStreamlit(types.SimpleNamespace):
    def __init__(self):
        super().__init__(session_state={})


class FakeSource:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reset_called = False

    def reset(self):
        self.reset_called = True


class FakeArm:
    instances = []

    def __init__(self):
        self.closed = False
        self.reset_called = False
        FakeArm.instances.append(self)

    def close(self):
        self.closed = True

    def reset(self):
        self.reset_called = True


class ClosingFailureArm(FakeArm):
    def close(self):
        self.closed = True
        raise RuntimeError("close failed")


class TestDashboardAppLifecycle(unittest.TestCase):
    def setUp(self):
        self.streamlit = FakeStreamlit()
        self.original_streamlit = sys.modules.get("streamlit")
        sys.modules["streamlit"] = self.streamlit
        sys.modules.pop("dashboard_app", None)
        self.dashboard_app = importlib.import_module("dashboard_app")
        FakeArm.instances = []
        self.dashboard_app.st.session_state.clear()

    def tearDown(self):
        sys.modules.pop("dashboard_app", None)
        if self.original_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = self.original_streamlit

    def test_get_source_closes_previous_arm_when_key_changes(self):
        previous_arm = FakeArm()
        self.dashboard_app.st.session_state.update(
            {
                "source": FakeSource(),
                "source_key": ("Offline PhysioNet", 1, 0, 1, 1.0),
                "arm": previous_arm,
                "records": [{"old": "record"}],
                "last_frame": object(),
            }
        )

        with (
            patch.object(self.dashboard_app, "ArmVisualizer", FakeArm),
            patch.object(self.dashboard_app, "OfflinePhysioNetSource", FakeSource),
        ):
            source = self.dashboard_app.get_source(
                "Offline PhysioNet",
                subject=2,
                start_epoch=0,
                stop_epoch=1,
                duration_sec=1.0,
            )

        self.assertIsInstance(source, FakeSource)
        self.assertTrue(previous_arm.closed)
        self.assertIsNot(self.dashboard_app.st.session_state["arm"], previous_arm)
        self.assertTrue(self.dashboard_app.st.session_state["arm"].reset_called)
        self.assertEqual(self.dashboard_app.st.session_state["records"], [])
        self.assertIsNone(self.dashboard_app.st.session_state["last_frame"])

    def test_reset_run_closes_previous_arm_and_clears_run_state(self):
        source = FakeSource()
        previous_arm = ClosingFailureArm()
        self.dashboard_app.st.session_state.update(
            {
                "arm": previous_arm,
                "records": [{"old": "record"}],
                "last_frame": object(),
            }
        )

        with patch.object(self.dashboard_app, "ArmVisualizer", FakeArm):
            self.dashboard_app._reset_run(source)

        self.assertTrue(source.reset_called)
        self.assertTrue(previous_arm.closed)
        self.assertIsNot(self.dashboard_app.st.session_state["arm"], previous_arm)
        self.assertTrue(self.dashboard_app.st.session_state["arm"].reset_called)
        self.assertEqual(self.dashboard_app.st.session_state["records"], [])
        self.assertIsNone(self.dashboard_app.st.session_state["last_frame"])


if __name__ == "__main__":
    unittest.main()
