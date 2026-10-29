
import re


def analyze_message(message):
    """
    Identify warning signs commonly found in scam messages.
    This is a rule-based helper, not the ML prediction model.
    """

    text = message.lower()
    warnings = []

    # 1. Urgent or threatening language
    urgency_words = [
        "urgent", "immediately", "act now",
        "account blocked", "account suspended",
        "expires today", "within 24 hours"
    ]

    for word in urgency_words:
        if word in text:
            warnings.append(
                f"Urgent or threatening language: '{word}'"
            )

    # 2. Requests for sensitive information
    sensitive_words = [
        "otp", "password", "pin",
        "bank details", "cvv", "verification code"
    ]

    for word in sensitive_words:
        if word in text:
            warnings.append(
                f"Sensitive information mentioned: '{word}'"
            )

    # 3. Financial promises or pressure
    financial_words = [
        "claim your prize", "you won",
        "free gift", "lottery winner",
        "limited offer", "send money"
    ]

    for word in financial_words:
        if word in text:
            warnings.append(
                f"Potential financial scam indicator: '{word}'"
            )

    # 4. Detect URLs
    urls = re.findall(r"https?://\S+|www\.\S+", text)

    if urls:
        warnings.append(
            "Contains a URL. Check the destination carefully."
        )

    # Return the results
    return {
        "warning_count": len(warnings),
        "warnings": warnings,
        "has_url": bool(urls)
    }
