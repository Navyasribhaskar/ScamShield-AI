
import unittest

from utils import analyze_message


class TestMessageAnalysis(unittest.TestCase):

    def test_urgent_message(self):
        result = analyze_message(
            "URGENT! Your account will be blocked immediately."
        )
        self.assertGreater(result["warning_count"], 0)

    def test_url_detection(self):
        result = analyze_message(
            "Please visit https://example.com to learn more."
        )
        self.assertTrue(result["has_url"])

    def test_otp_warning(self):
        result = analyze_message(
            "Please share your OTP to verify your account."
        )
        self.assertTrue(
            any("Sensitive information" in warning
                for warning in result["warnings"])
        )

    def test_ordinary_message(self):
        result = analyze_message(
            "Hey, are we still meeting at 5 pm today?"
        )
        self.assertEqual(result["warning_count"], 0)

    def test_empty_message(self):
        result = analyze_message("")
        self.assertEqual(result["warning_count"], 0)
        self.assertFalse(result["has_url"])


if __name__ == "__main__":
    unittest.main()
