import unittest
from types import SimpleNamespace

import vanish_cam as app


def point(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def face_landmarks():
    points = [point(0.35, 0.2) for _ in range(11)]
    points[0] = point(0.35, 0.2)
    points[1] = point(0.65, 0.2)
    points[2] = point(0.65, 0.8)
    points[3] = point(0.35, 0.8)
    points[10] = point(0.50, 0.32)
    return points


def extended_face_landmarks(eye_center_x=0.45):
    points = [point(0.40, 0.20) for _ in range(478)]
    points[0] = point(0.20, 0.20)
    points[1] = point(0.80, 0.20)
    points[2] = point(0.80, 0.80)
    points[3] = point(0.20, 0.80)
    for index, x in zip((33, 133, 263, 362), (eye_center_x - 0.04,
                                               eye_center_x - 0.01,
                                               eye_center_x + 0.01,
                                               eye_center_x + 0.04)):
        points[index] = point(x, 0.45)
    return points


def goku_hand(tip_gap=0.0, y_offset=0.0):
    tip_left = 0.50 - tip_gap / 2
    tip_right = 0.50 + tip_gap / 2
    points = (
        [point(0.52, 0.65)]
        + [point(0.50, 0.60), point(0.48, 0.56), point(0.47, 0.52), point(0.46, 0.49)]
        + [point(0.45, 0.57), point(0.45, 0.48), point(0.45, 0.39), point(tip_left, 0.32)]
        + [point(0.55, 0.57), point(0.55, 0.48), point(0.55, 0.39), point(tip_right, 0.32)]
        + [point(0.62, 0.57), point(0.60, 0.60), point(0.57, 0.64), point(0.54, 0.66)]
        + [point(0.67, 0.58), point(0.64, 0.63), point(0.59, 0.67), point(0.55, 0.68)]
    )
    return [point(item.x, item.y + y_offset, item.z) for item in points]


class GokuPoseTests(unittest.TestCase):
    def test_side_view_uses_eye_center_not_face_box_center(self):
        anchor = app.forehead_anchor(extended_face_landmarks(eye_center_x=0.43))
        self.assertAlmostEqual(anchor[0], 0.43, places=3)
        self.assertAlmostEqual(anchor[1], 0.318, places=3)

    def test_side_facing_two_fingers_to_forehead_is_valid(self):
        valid, score, forehead, details = app.evaluate_goku_pose(
            goku_hand(), face_landmarks()
        )
        self.assertTrue(valid)
        self.assertGreater(score, 0.8)
        self.assertAlmostEqual(forehead[0], 0.5)
        self.assertAlmostEqual(forehead[1], 0.332, places=3)
        self.assertIn("contact=NEAR", details)

    def test_tip_spacing_does_not_block_forehead_contact(self):
        valid, _, _, details = app.evaluate_goku_pose(
            goku_hand(tip_gap=0.12), face_landmarks()
        )
        self.assertTrue(valid)
        self.assertIn("tips=not required", details)

    def test_hand_away_from_forehead_is_rejected(self):
        valid, _, _, details = app.evaluate_goku_pose(
            goku_hand(y_offset=0.25), face_landmarks()
        )
        self.assertFalse(valid)
        self.assertIn("contact=FAR", details)

    def test_missing_face_or_hand_is_rejected(self):
        valid, score, forehead, details = app.evaluate_goku_pose([], [])
        self.assertFalse(valid)
        self.assertEqual(score, 0.0)
        self.assertIsNone(forehead)
        self.assertIn("face not fully detected", details)

    def test_forehead_anchor_survives_missing_hand(self):
        valid, score, forehead, details = app.evaluate_goku_pose([], face_landmarks())
        self.assertFalse(valid)
        self.assertEqual(score, 0.0)
        self.assertAlmostEqual(forehead[0], 0.5)
        self.assertAlmostEqual(forehead[1], 0.332, places=3)
        self.assertEqual(details, "hand not fully detected")

    def test_hold_timer_triggers_once_until_pose_is_released(self):
        hand_result = SimpleNamespace(hand_landmarks=[goku_hand()])
        face_result = SimpleNamespace(face_landmarks=[face_landmarks()])

        class FakeDetector:
            def __init__(self, result):
                self.result = result

            def detect_for_video(self, image, timestamp_ms):
                return self.result

            def close(self):
                pass

        detector = app.GokuPoseDetector(
            FakeDetector(hand_result), FakeDetector(face_result)
        )
        self.assertFalse(detector.detect(object(), 0).trigger)
        self.assertFalse(detector.detect(object(), 100).trigger)
        self.assertTrue(detector.detect(object(), 250).trigger)
        self.assertFalse(detector.detect(object(), 300).trigger)

        detector.hand_detector.result = SimpleNamespace(hand_landmarks=[])
        detector.face_detector.result = SimpleNamespace(face_landmarks=[])
        self.assertFalse(detector.detect(object(), 400).trigger)

        detector.hand_detector.result = hand_result
        detector.face_detector.result = face_result
        self.assertFalse(detector.detect(object(), 500).trigger)
        self.assertTrue(detector.detect(object(), 750).trigger)


if __name__ == "__main__":
    unittest.main()
