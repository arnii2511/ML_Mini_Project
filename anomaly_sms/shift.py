from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SHIFT_TEMPLATES = [
    "URGENT: verify your UPI profile immediately to avoid payment block click {url}",
    "Crypto reward unlocked for your wallet claim now at {url}",
    "OTP fraud alert: confirm your banking device with code {code} at {url}",
    "FASTag KYC failed update PAN and Aadhaar details here {url}",
    "Your salary cashback is waiting approve the wallet request now",
    "Security notice: suspicious UPI collect request detected open {url}",
    "Limited token airdrop for premium users activate now and win {amount}",
    "Bank warning: your account access expires tonight verify mobile banking at {url}",
    "Instant loan preapproved submit card PIN and OTP for release",
    "Electricity bill overdue pay via UPI link now or service stops today",
    "Account under review due to crypto withdrawal confirm identity at {url}",
    "Insurance refund available enter debit card cvv to receive {amount}",
]


@dataclass
class SyntheticShiftConfig:
    synthetic_spam_count: int = 24


def generate_synthetic_shift_messages(count: int) -> list[str]:
    messages = []
    for index in range(count):
        template = SHIFT_TEMPLATES[index % len(SHIFT_TEMPLATES)]
        amount = 2000 + (index * 375)
        code = 410000 + index
        url = f"http://alert{index + 1}.secure-check.example"
        messages.append(template.format(url=url, amount=amount, code=code))
    return messages


def build_synthetic_shift_frame(count: int) -> pd.DataFrame:
    if count <= 0:
        return pd.DataFrame(columns=["label", "text", "is_anomaly", "split", "source"])

    texts = generate_synthetic_shift_messages(count)
    return pd.DataFrame(
        {
            "label": ["spam"] * len(texts),
            "text": texts,
            "is_anomaly": [1] * len(texts),
            "split": ["test"] * len(texts),
            "source": ["synthetic_shift"] * len(texts),
        }
    )
